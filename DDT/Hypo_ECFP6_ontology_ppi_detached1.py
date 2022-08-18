from geoopt import ManifoldParameter
from optim import RiemannianAdam, RiemannianSGD
from utils.data_read import read_original_files, adj_preprocess
from models.base_models import NCModel
from config import parser
import numpy as np
import torch
import logging
import time
from scipy.sparse import coo_matrix
import os

# for pre-training hyperbolic target embeddings using GO/PPI graph

# overall parameters
# if set SEPARATED[0] = True, which means that the model uses separated PPI graph and ontology graph, otherwise treats the both graphs as one homogeneous graph
# for SEPARATED[1], you could specify that the model uses the both graphs or one of both (1: ontology graph, 2: PPI graph, only available when SEPARATED[0] = True)
SEPARATED = [True, [1, 2]]
# DIRECTED = True: treat every graph to a directed graph, otherwise to an undirected graph
DIRECTED = True
# whether to add self loop to every graph
SELFLOOP = True

def run_model_predict_DDT(args):
    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # double_precision default = 0
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    selected_triples_, selected_drugs_abs2rel, selected_diseases_abs2rel, selected_targets_abs2rel, all_drug_morgan, target_similarity, \
    [anno_graph, anno_onto_graph, ppi_graph, anno_onto_ppi_graph], train_idx, val_idx, test_idx, manual_disease_absid_, rare_disease_absid_ = read_original_files(args.path_prefix)

    # 0: anno_graph, 1: anno_onto_graph, 2: ppi_graph, 3: anno_onto_ppi_graph
    graph_list = [anno_graph, anno_onto_graph, ppi_graph, anno_onto_ppi_graph]
    # abs2_rel_list = [selected_drugs_abs2rel, selected_diseases_abs2rel, selected_targets_abs2rel]
    # train_triples, val_triples, test_triples, manual_disease_triples, rare_disease_triples, all_drug_morgan = tensorize_original_files(selected_triples_, all_drug_morgan, train_idx, val_idx, test_idx, manual_disease_absid_, rare_disease_absid_)

    processed_adj = adj_preprocess(args, graph_list, normalize_adj=args.normalize_adj, directed=DIRECTED, separated=SEPARATED, self_loop=SELFLOOP)

    # initialize Euclidean node features
    target_features_list=[]
    target_in_dims=[]
    for adj in processed_adj:
        dim = adj.size(0)
        target_in_dims.append(dim)
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        values = np.ones(dim)
        feats = coo_matrix((values, indices), shape=(dim, dim))
        if target_similarity.sum() != 0:
            print('Current pre-training process uses target similarity as the node initial feature')
            feats = feats.todense()
            feats[:len(selected_targets_abs2rel), :len(selected_targets_abs2rel)] = target_similarity
            print('Current target feature size:', feats.shape, 'target similarity size:', target_similarity.shape)
            feats = coo_matrix(feats)
        feats_values = feats.data
        feats_indices = np.vstack((feats.row, feats.col))
        i = torch.LongTensor(feats_indices)
        v = torch.FloatTensor(feats_values)
        shape = feats.shape
        feats = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        target_features_list.append(feats.to(args.device))

    args.n_classes = len(selected_targets_abs2rel)
    Model = NCModel(target_features_list, selected_targets_abs2rel, processed_adj, args)
    Model.to(args.device)

    no_decay = ['bias', 'scale']
    optimizer_grouped_parameters = [
        {
        'params': [p for n, p in Model.named_parameters()
            if p.requires_grad and not any(nd in n for nd in no_decay) and not isinstance(p, ManifoldParameter)],
        'weight_decay': args.weight_decay},
        {
        'params': [p for n, p in Model.named_parameters()
            if p.requires_grad and any(nd in n for nd in no_decay) or isinstance(p, ManifoldParameter)],
        'weight_decay': 0.0 }]

    if args.optimizer == 'radam':
        opt = RiemannianAdam(params=optimizer_grouped_parameters,
                             lr=args.lr,
                             stabilize=10)
    elif args.optimizer == 'rsgd':
        opt = RiemannianSGD(params=optimizer_grouped_parameters,
                            lr=args.lr,
                            stabilize=10)
    else:
        raise ValueError('Wrong optimizer')

    tot_params = sum([np.prod(p.size()) for p in Model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")

    t_total = time.time()
    counter = 0
    best_val_metrics = Model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    compute_metrics_idx = [i for i in range(len(selected_targets_abs2rel))]
    compute_metrics_torch_idx = torch.LongTensor(compute_metrics_idx).to(args.device)
    for epoch in range(args.epochs):
        t = time.time()
        Model.train()
        opt.zero_grad()

        embeddings = Model.forward()
        train_metrics = Model.compute_metrics(embeddings, compute_metrics_idx, compute_metrics_torch_idx)
        train_metrics['loss'].backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(Model.parameters(), max_norm=args.grad_clip)
        opt.step()

        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join([
                'Epoch: {:04d}'.format(epoch + 1),
                format_metrics(train_metrics, 'train'),
                'time: {:.4f}s'.format(time.time() - t)]))

        with torch.no_grad():
            if (epoch + 1) % args.valid_steps == 0:
                Model.eval()
                embeddings = Model.forward()
                val_metrics = Model.compute_metrics(embeddings, compute_metrics_idx, compute_metrics_torch_idx)
                if (epoch + 1) % args.log_freq == 0:
                    logging.info(" ".join([
                        'Epoch: {:04d}'.format(epoch + 1),
                        format_metrics(val_metrics, 'val')]))

                if Model.has_improved(best_val_metrics, val_metrics):
                    best_test_metrics = Model.compute_metrics(embeddings, compute_metrics_idx, compute_metrics_torch_idx)
                    best_emb = embeddings.cpu()
                    best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    if counter == args.early_stop and epoch > args.min_epochs:
                        logging.info("Early stopping")
                        break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    if not best_test_metrics:
        Model.eval()
        best_emb = Model.forward()
        best_test_metrics = Model.compute_metrics(best_emb, compute_metrics_idx, compute_metrics_torch_idx)
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:",format_metrics(best_test_metrics, 'test')]))

    np.save(os.path.join('./data/target_pretrained_storage/', 'target_pretrained_embeddings.npy'), best_emb.cpu().detach().numpy())
    # torch.save(Model.state_dict(), os.path.join('./data/', 'model.pth'))

def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])


if __name__=='__main__':
    args=parser.parse_args()

    # add or modify some hyper-parameters imported from config.py file
    # 1: common part
    args.seed = 1234
    args.cuda = 0  # default: -1
    args.path_prefix = r'./data/Cross_validation_split/fold4/' # your original file path
    args.data = 'DTINet'
    args.dim = 128  # overall embedding dimension in the model
    args.epochs = 300
    args.lr = 0.005
    args.weight_decay = 0
    # margin is used to control the margin hyper-parameter in KG completion score function
    args.margin = 8.0
    # grad_clip is used to clip the gradient in optimization
    args.grad_clip = 0.5
    # early stop is executed based on valid_steps
    args.early_stop = 10
    args.valid_steps = 10  # default: 10
    args.optimizer = 'radam'

    # 2: hyperbolic GCN part
    # link prediction task
    args.task = 'lp'
    # name of target graph processing model
    args.model = 'HyboNet'
    args.num_layers = 2
    args.bias = 1
    args.normalize_adj = 1
    args.manifold = 'Lorentz'
    args.target_graph_mapping = ['expmap']
    # this dropout is only used to control the dropout degree in Lorentz Linears of the HyboNet encoder
    args.dropout = 0
    # Lorentz GCN layer for implementing addition operation in hyperbolic space
    # initialize GCN_add operation when GCN_add_layers is larger than 0, if GCN_add_layers=0, means that do not use GCN_add operation
    # what actually decides whether GCN_add is used is how many groups of target embeddings in target_embedding_list
    args.GCN_add_layers = 1  # >= 0 Int, only serve for GCN_add operation
    # compared with EXP_MAX_NORM used in lmath to control 'a' in the expmap function, it is used as an extra parameter in lorentz projx/lmath project function to further control the norm of fused target embeddings (if exists)
    args.max_norm = 2.5 # do not use max_norm: None/0, only serve for GCN_add operation
    # the frequency of printing evaluation results during the training/val phase
    args.log_freq = 5

    args.SEPARATED = SEPARATED
    args.DIRECTED = DIRECTED
    args.SELFLOOP = SELFLOOP

    for name,value in vars(args).items():
            print(name,value)
    print('*** Above Are All Hyper Parameters ***')

    run_model_predict_DDT(args)