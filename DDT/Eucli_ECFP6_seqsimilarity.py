from models.base_models_ import EucliNet4KG
import torch.optim as optim
from utils.data_read import read_original_files, tensorize_original_files, get_er_vocab
from config import parser
import numpy as np
import torch
import logging
from tqdm import tqdm
from copy import deepcopy
import time

# for Euclidean based variants using domain knowledge

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
    [anno_graph, anno_onto_graph, ppi_graph, anno_onto_ppi_graph], \
    train_idx, val_idx, test_idx, manual_disease_absid_, rare_disease_absid_ = read_original_files(args.path_prefix)

    abs2_rel_list = [selected_drugs_abs2rel, selected_diseases_abs2rel, selected_targets_abs2rel]
    train_triples, val_triples, test_triples, manual_disease_triples, rare_disease_triples, all_drug_morgan = tensorize_original_files(selected_triples_, all_drug_morgan, train_idx, val_idx, test_idx, manual_disease_absid_, rare_disease_absid_)

    # TWO
    train_triples = train_triples.to(args.device)
    val_triples = val_triples.to(args.device)
    test_triples = test_triples.to(args.device)

    all_drug_morgan = all_drug_morgan.to(args.device)

    euclinet4kg = EucliNet4KG(abs2_rel_list, args, args.dim, args.margin, args.extra_bias, args.drug_pretrained, args.target_pretrained, all_drug_morgan)
    euclinet4kg.to(args.device)

    if args.optimizer == 'adam':
        opt = optim.Adam(params=euclinet4kg.parameters(), lr=args.lr)

    elif args.optimizer == 'sgd':
        opt = optim.SGD(params=euclinet4kg.parameters(), lr=args.lr)
    else:
        raise ValueError('Wrong optimizer')

    tot_params = sum([np.prod(p.size()) for p in euclinet4kg.parameters()])
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
        euclinet4kg.train()
        losses = []
        np.random.shuffle(train_order)
        # batch wise training
        for j in range(0, len(train_triples), args.batch_size):
            data_batch = train_triples[train_order[j:j + args.batch_size]]
            data_batch_np = train_triples_np[train_order[j:j + args.batch_size]]
            # remove the last batch
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

            predictions = euclinet4kg(e1_idx, r_idx, e2_idx)
            # BCE loss + sigmoid
            loss = euclinet4kg.loss(predictions, targets)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad.clip_grad_norm_(euclinet4kg.parameters(), max_norm=args.grad_clip)

            opt.step()
            losses.append(loss.item())

        bar.set_description('Best:%.4f@%d,cur:%.4f,loss:%.3f' % (max_mrr, max_it, mrr, np.mean(losses)))
        euclinet4kg.eval()
        with torch.no_grad():
            if not it % args.valid_steps:
                # np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0]), np.mean(1. / np.array(ranks))
                hit10, hit3, hit1, mrr = evaluate(euclinet4kg, selected_triples_, val_triples, selected_targets_abs2rel, args)
                if mrr > max_mrr:
                    # max_mrr = 0.0
                    # max_it = 0
                    # mrr = 0
                    # bad_cnt = 0
                    max_mrr = mrr
                    max_it = it
                    bad_cnt = 0
                    best_model = deepcopy(euclinet4kg.state_dict())
                else:
                    bad_cnt += 1
                    if bad_cnt == args.early_stop:
                        break

                bar.set_description('Best:%.4f@%d,cur:%.4f,loss:%.3f' % (max_mrr, max_it, mrr, loss.item()))
        epoch_end_time = time.time()
        print('Current Epoch %d Time Cost' % (it), epoch_end_time - epoch_start_time, 'Average Loss %.4f' % (np.mean(losses)))

    with torch.no_grad():
        euclinet4kg.load_state_dict(best_model)
        hit10, hit3, hit1, mrr = evaluate(euclinet4kg, selected_triples_, test_triples, selected_targets_abs2rel, args)

    # print('Manual Disease Test Result\nBest it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f' % (max_it, manual_hit10, manual_hit3, manual_hit1, manual_mrr))
    # print('Rare Disease Test Result\nBest it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f' % (max_it, rare_hit10, rare_hit3, rare_hit1, rare_mrr))
    print('Overall Disease Test Result\nBest it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f' % (max_it, hit10, hit3, hit1, mrr))

    file = './outputs/' + 'Eucli_ECFP6_seqsimilarity_log_%d.%s.txt' % (args.dim, args.data)
    with open(file, 'a') as f:
        f.write(str(args) + '\n')
        # f.write('Manual disease results: Best it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f\n' % (max_it, manual_hit10, manual_hit3, manual_hit1, manual_mrr))
        # f.write('Rare disease results: Best it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f\n' % (max_it, rare_hit10, rare_hit3, rare_hit1, rare_mrr))
        f.write('Overall disease results: Best it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f\n' % (max_it, hit10, hit3, hit1, mrr))
        if args.save_model:
            torch.save(euclinet4kg, './outputs/' + 'Eucli_ECFP6_seqsimilarity_{}.pt'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))


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
    args.save_model = True  # whether to save the model
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
    args.early_stop = 10
    args.valid_steps = 10  # default: 10
    args.optimizer = 'adam'

    # 2: knowledge graph completion part
    # number of negative samples in graph completion training
    args.nneg = 50
    args.real_neg = True
    # choose to use different Euclidean KGC variants：[None, 'vector', 'matrix']
    # None: DDTE, 'vector': DDTE_bias, 'matrix': fully Euclidean counterpart of FLNE
    args.extra_bias = 'matrix'
    # choose to use ECFP6
    args.drug_pretrained = [True]
    # choose to use target sequence similarity: [True/False, 'similarity']
    args.target_pretrained = [True, 'similarity']

    for name,value in vars(args).items():
            print(name,value)
    print('*** Above Are All Hyper Parameters ***')

    run_model_predict_DDT(args)