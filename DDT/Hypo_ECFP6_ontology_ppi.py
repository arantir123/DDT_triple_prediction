from geoopt import ManifoldParameter
from optim import RiemannianAdam, RiemannianSGD
from utils.data_read import read_original_files, tensorize_original_files, adj_preprocess, get_er_vocab
from models.base_models import HyboNet4KG
from config import parser
import numpy as np
import torch
import logging
from tqdm import tqdm
from copy import deepcopy
import time
from scipy.sparse import coo_matrix

# for hyperbolic based variants using ECFP6 + GO/PPI graph

# overall parameters
# if set SEPARATED[0] = True, which means that the model uses separated PPI graph and ontology graph, otherwise treats the both graphs as one homogeneous graph
# for SEPARATED[1], you could specify that the model uses the both graphs or one of both (1: ontology graph, 2: PPI graph, only available when SEPARATED[0] = True)
SEPARATED = [True, [1, 2]]
# DIRECTED = True: treat every graph to a directed graph, otherwise to an undirected graph
DIRECTED = True
# whether to add self loop to every graph
SELFLOOP = True

DRUG_NUM = 708
TARGET_NUM = 1493
DISEASE_NUM = 5603

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

    # ONE
    selected_triples_, selected_drugs_abs2rel, selected_diseases_abs2rel, selected_targets_abs2rel, all_drug_morgan, target_similarity, \
    [anno_graph, anno_onto_graph, ppi_graph, anno_onto_ppi_graph], train_idx, val_idx, test_idx, manual_disease_absid_, rare_disease_absid_ = read_original_files(args.path_prefix)

    # 0: anno_graph, 1: anno_onto_graph, 2: ppi_graph, 3: anno_onto_ppi_graph
    graph_list = [anno_graph, anno_onto_graph, ppi_graph, anno_onto_ppi_graph]
    abs2_rel_list = [selected_drugs_abs2rel, selected_diseases_abs2rel, selected_targets_abs2rel]

    train_triples, val_triples, test_triples, manual_disease_triples, rare_disease_triples, all_drug_morgan = tensorize_original_files(selected_triples_, all_drug_morgan, train_idx, val_idx, test_idx, manual_disease_absid_, rare_disease_absid_)

    processed_adj = adj_preprocess(args, graph_list, normalize_adj=args.normalize_adj, directed=DIRECTED, separated=SEPARATED, self_loop=SELFLOOP)

    # initalize Euclidean node features
    target_features_list=[]
    target_in_dims=[]
    for adj in processed_adj:
        dim = adj.size(0)
        target_in_dims.append(dim)
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        values = np.ones(dim)
        feats = coo_matrix((values, indices), shape=(dim, dim))
        if target_similarity.sum() != 0:
            feats = feats.todense()
            feats[:len(selected_targets_abs2rel), :len(selected_targets_abs2rel)] = target_similarity
            print('***************************************************************************')
            print('Current target feature size:', feats.shape, 'target similarity size:', target_similarity.shape)
            print('***************************************************************************')
            feats = coo_matrix(feats)
        feats_values = feats.data
        feats_indices = np.vstack((feats.row, feats.col))
        i = torch.LongTensor(feats_indices)
        v = torch.FloatTensor(feats_values)
        shape = feats.shape
        feats = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        target_features_list.append(feats.to(args.device))

    # TWO
    train_triples = train_triples.to(args.device)
    val_triples = val_triples.to(args.device)
    test_triples = test_triples.to(args.device)

    all_drug_morgan = all_drug_morgan.to(args.device)

    hyponet4kg = HyboNet4KG(target_features_list, processed_adj, all_drug_morgan, abs2_rel_list, args, args.dim, args.max_scale, args.margin)
    hyponet4kg.to(args.device)

    no_decay = ['bias', 'scale']
    optimizer_grouped_parameters = [
        {
        'params': [p for n, p in hyponet4kg.named_parameters()
            if p.requires_grad and not any(nd in n for nd in no_decay) and not isinstance(p, ManifoldParameter)],
        'weight_decay': args.weight_decay},
        {
        'params': [p for n, p in hyponet4kg.named_parameters()
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

    tot_params = sum([np.prod(p.size()) for p in hyponet4kg.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")

    # THREE
    train_triples_np = train_triples.cpu().numpy()
    train_order = list(range(len(train_triples)))
    targets = np.zeros((args.batch_size, args.nneg + 1))
    targets[:, 0] = 1
    targets = torch.FloatTensor(targets).to(args.device)
    max_mrr = 0.0
    max_it = 0
    mrr = 0
    bad_cnt = 0
    print('Starting training...')
    # create a dict with set as its value type, storing the known targets for given drug + disease in the dataset
    # for ensuring every negative sample in the training phase is the real unknown sample
    sr_vocab = get_er_vocab(selected_triples_)

    bar = tqdm(range(1, args.epochs + 1),
                desc='Best:%.3f@%d,curr:%.3f,loss:%.3f' %
                (max_mrr, max_it, 0., 0.),
                ncols=75)
    best_model = None
    for it in bar:
        epoch_start_time = time.time()
        hyponet4kg.train()
        losses = []
        np.random.shuffle(train_order)
        # batch wise training
        for j in range(0, len(train_triples), args.batch_size):
            data_batch = train_triples[train_order[j:j + args.batch_size]]
            data_batch_np = train_triples_np[train_order[j:j + args.batch_size]]
            if j + args.batch_size > len(train_triples):
                continue

            negsamples = np.random.randint(low=0, high=len(selected_targets_abs2rel), size=(data_batch.size(0), args.nneg), dtype=np.int32)
            if args.real_neg:
                # Filter out the fake negative samples.
                candidate = np.random.randint(low=0, high=len(selected_targets_abs2rel), size=(data_batch.size(0)), dtype=np.int32)
                p_candidate = 0
                e1_idx_np = data_batch_np[:, 0]
                r_idx_np = data_batch_np[:, 1]
                for index in range(len(negsamples)):
                    filt = sr_vocab[(e1_idx_np[index], r_idx_np[index])]
                    for index_ in range(len(negsamples[index])):
                        while negsamples[index][index_] in filt:
                            negsamples[index][index_] = candidate[p_candidate]
                            p_candidate += 1
                            if p_candidate == len(candidate):
                                candidate = np.random.randint(0, len(selected_targets_abs2rel), size=(args.batch_size))
                                p_candidate = 0
            negsamples = torch.LongTensor(negsamples).to(args.device)
            opt.zero_grad()

            e1_idx = data_batch[:, 0]
            r_idx = data_batch[:, 1]
            e2_idx = torch.cat([data_batch[:, 2:3], negsamples], dim=-1)

            predictions = hyponet4kg(e1_idx, r_idx, e2_idx)
            loss = hyponet4kg.loss(predictions, targets)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad.clip_grad_norm_(hyponet4kg.parameters(), max_norm=args.grad_clip)

            opt.step()
            losses.append(loss.item())

        bar.set_description('Best:%.4f@%d,cur:%.4f,loss:%.3f' % (max_mrr, max_it, mrr, np.mean(losses)))
        hyponet4kg.eval()
        with torch.no_grad():
            if not it % args.valid_steps:
                # np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0]), np.mean(1. / np.array(ranks))
                hit10, hit3, hit1, mrr = evaluate(hyponet4kg, selected_triples_, val_triples, selected_targets_abs2rel, args)
                if mrr > max_mrr:
                    # max_mrr = 0.0
                    # max_it = 0
                    # mrr = 0
                    # bad_cnt = 0
                    max_mrr = mrr
                    max_it = it
                    bad_cnt = 0
                    best_model = deepcopy(hyponet4kg.state_dict())
                else:
                    bad_cnt += 1
                    if bad_cnt == args.early_stop:
                        break
                bar.set_description('Best:%.4f@%d,cur:%.4f,loss:%.3f' % (max_mrr, max_it, mrr, loss.item()))

        epoch_end_time = time.time()
        print('Current Epoch %d Time Cost' % (it), epoch_end_time - epoch_start_time, 'Average Loss %.4f' % (np.mean(losses)))

    with torch.no_grad():
        hyponet4kg.load_state_dict(best_model)
        hit10, hit3, hit1, mrr = evaluate(hyponet4kg, selected_triples_, test_triples, selected_targets_abs2rel, args)

    # print('Manual Disease Test Result\nBest it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f' % (max_it, manual_hit10, manual_hit3, manual_hit1, manual_mrr))
    # print('Rare Disease Test Result\nBest it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f' % (max_it, rare_hit10, rare_hit3, rare_hit1, rare_mrr))
    print('Overall Disease Test Result\nBest it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f' % (max_it, hit10, hit3, hit1, mrr))

    file = './outputs/' + 'Hypo_ECFP6_ontology_ppi_log_%d.%s.txt' % (args.dim, args.data)
    with open(file, 'a') as f:
        f.write(str(args) + '\n')
        # f.write('Manual disease results: Best it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f\n' % (max_it, manual_hit10, manual_hit3, manual_hit1, manual_mrr))
        # f.write('Rare disease results: Best it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f\n' % (max_it, rare_hit10, rare_hit3, rare_hit1, rare_mrr))
        f.write('Overall disease results: Best it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f\n' % (max_it, hit10, hit3, hit1, mrr))


# evaluate(hyponet4kg, selected_triples_, val_triples, selected_targets_abs2rel, args)
def evaluate(model, all_data, data, tail_entity_set, args, batch=30):
    hits = []
    ranks = []
    for i in range(10):
        hits.append([])

    test_data_idxs = data.cpu().numpy()
    sr_vocab = get_er_vocab(all_data)

    tt = torch.Tensor(np.array(range(len(tail_entity_set)), dtype=np.int64)).to(args.device)
    tt = tt.long().repeat(batch, 1)

    for i in range(0, len(test_data_idxs), batch):
        data_point = test_data_idxs[i:i + batch]
        e1_idx = torch.tensor(data_point[:, 0])
        r_idx = torch.tensor(data_point[:, 1])
        e2_idx = torch.tensor(data_point[:, 2])
        # if args.cuda:
        e1_idx = e1_idx.to(args.device)
        r_idx = r_idx.to(args.device)
        e2_idx = e2_idx.to(args.device)

        predictions_s = model.forward(e1_idx, r_idx, tt[:min(batch, len(test_data_idxs) - i)])

        for j in range(min(batch, len(test_data_idxs) - i)):
            filt = list(sr_vocab[(data_point[j][0], data_point[j][1])])
            target_value = predictions_s[j][e2_idx[j]].item()
            predictions_s[j][filt] = -np.Inf
            predictions_s[j][e2_idx[j]] = target_value

            rank = (predictions_s[j] >= target_value).sum().item() - 1
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

    return np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0]), np.mean(1. / np.array(ranks))


if __name__=='__main__':
    args=parser.parse_args()

    # add or modify some hyper-parameters imported from config.py file
    # 1: common part
    args.seed = 1234
    args.cuda = 0  # default: -1
    args.path_prefix = r'./data/Cross_validation_split/fold4/' # your original file path
    args.data = 'DTINet'
    args.dim = 128  # overall embedding dimension in the model
    args.epochs = 500  # default: 500
    args.batch_size = 1000  # default: 1000
    args.lr = 0.005
    args.weight_decay = 0
    # margin is used to control the margin hyper-parameter in the KGC score function
    args.margin = 8.0
    # grad_clip is used to clip the gradient in optimization
    args.grad_clip = 0.5
    # early stop is executed based on valid_steps
    args.early_stop = 15
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
    # this dropout is only used to control the dropout degree in Lorentz Linears of the HyboNet encoder
    args.dropout = 0
    # Lorentz GCN layer for implementing addition operation in hyperbolic space
    # initialize GCN_add operation when GCN_add_layers is larger than 0, if GCN_add_layers=0, means that do not use GCN_add operation
    # what actually decides whether GCN_add is used is how many groups of target embeddings in target_embedding_list
    args.GCN_add_layers = 1  # >= 0 Int, only serve for GCN_add operation
    # compared with EXP_MAX_NORM used in lmath to control 'a' in the expmap function, it is used as an extra parameter in lorentz projx/lmath project function to further control the norm of fused target embeddings (if exists)
    args.max_norm = 2.5 # do not use max_norm: None/0, only serve for GCN_add operation
    args.target_graph_mapping = ['expmap'] # an extra parameter, to control whether to use expmap to map initial graph node features from Eulidean space to hyperbolic space, ['direct', 'expmap'], default: 'expmap'

    # 3: knowledge graph completion part
    # number of negative samples in graph completion training
    args.nneg = 50
    # max_scale is used to control the scale between time dimension and space dimension in Lorentz Linear of decoder (which is slightly different from that in Lorentz GCN encoder)
    # in Lorentz GCN encoder, it is trainable parameter with the initialization 10, while in KG decoder, it is a fixed scaler determined by this hyper-parameter
    args.max_scale = 2.5
    args.real_neg = True

    args.SEPARATED = SEPARATED
    args.DIRECTED = DIRECTED
    args.SELFLOOP = SELFLOOP

    for name,value in vars(args).items():
            print(name,value)
    print('*** Above Are All Hyper Parameters ***')

    run_model_predict_DDT(args)