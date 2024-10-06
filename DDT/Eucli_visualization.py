from utils.data_read import read_original_files, tensorize_original_files, get_er_vocab
from config import parser
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import pickle

DRUG_NUM = 708
TARGET_NUM = 1493
DISEASE_NUM = 5603

# visualize the spatial layout of target embeddings under given drug and disease based on Euclidean KGC methods
# currently only for the fully Euclidean counterpart of FLONE

def run_model_visualize_DDT(args):
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

    train_triples, val_triples, test_triples, manual_disease_triples, rare_disease_triples, all_drug_morgan = tensorize_original_files(selected_triples_, all_drug_morgan, train_idx, val_idx, test_idx, manual_disease_absid_, rare_disease_absid_)

    test_triples = test_triples.to(args.device)
    print('Starting visualization...')

    with torch.no_grad():
        euclinet4kg = torch.load('./outputs/' + args.model_name)
        tot_params = sum([np.prod(p.size()) for p in euclinet4kg.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")

        visualized_positive_triple = test_triples[(test_triples[:, 0] == args.selected_drug_disease[0]) & (test_triples[:, 1] == args.selected_drug_disease[1])]
        print('visualized_positive_triple:\n', visualized_positive_triple)
        drug_disease = torch.LongTensor(args.selected_drug_disease).repeat(len(selected_targets_abs2rel), 1)
        target = torch.arange(len(selected_targets_abs2rel)).unsqueeze(1)
        visualized_all_triple = torch.cat((drug_disease, target), 1) # all triples based on candidate targets under given drug and disease

        drug_entity = euclinet4kg.drug_transform(euclinet4kg.drug_entity)
        target_entity = euclinet4kg.target_transform(euclinet4kg.target_entity)

        euclidean_drug_embedding = drug_entity[visualized_all_triple[:, 0]]
        euclidean_target_embedding = target_entity[visualized_all_triple[:, 2]]
        r_bias = euclinet4kg.relation_bias[visualized_all_triple[:, 1]]
        r_transform = euclinet4kg.relation_transform[visualized_all_triple[:, 1]]
        euclidean_drug_embedding_transformed = euclidean_drug_embedding.unsqueeze(1) @ r_transform.transpose(-2, -1)
        euclidean_drug_embedding_transformed = (euclidean_drug_embedding_transformed + r_bias.unsqueeze(1)).squeeze(1)

        # reduce high-dimension coordinates to 2-dimension
        # group1: drug embedding before translation + target embedding
        ei_x1, ei_y1 = poincare_dimension_reduction(euclidean_drug_embedding, euclidean_target_embedding)
        # group2：drug embedding after translation + target embedding
        ei_x2, ei_y2 = poincare_dimension_reduction(euclidean_drug_embedding_transformed, euclidean_target_embedding)

        pos_idx = visualized_positive_triple[:, -1].cpu().numpy() + 1
        neg_idx = np.array(list(set(np.arange(len(selected_targets_abs2rel)) + 1) - set(pos_idx)))
        neg_idx.sort()
        print('positive triple number:', len(pos_idx), '({})'.format(pos_idx), 'unknown triple number:', len(neg_idx))
        # for group1
        ei_x1 = ei_x1.cpu().numpy() # x coordinate
        ei_y1 = ei_y1.cpu().numpy() # y coordinate
        pos_ei_x1 = ei_x1[pos_idx]
        pos_ei_y1 = ei_y1[pos_idx]
        neg_ei_x1 = ei_x1[neg_idx]
        neg_ei_y1 = ei_y1[neg_idx]
        # for group2
        ei_x2 = ei_x2.cpu().numpy()
        ei_y2 = ei_y2.cpu().numpy()
        pos_ei_x2 = ei_x2[pos_idx]
        pos_ei_y2 = ei_y2[pos_idx]
        neg_ei_x2 = ei_x2[neg_idx]
        neg_ei_y2 = ei_y2[neg_idx]

        fig = plt.figure(figsize=(12, 12))

        plt.scatter(ei_x1[0], ei_y1[0], c='springgreen') # given drug
        plt.scatter(neg_ei_x1, neg_ei_y1, c='paleturquoise') # unknown candidate targets
        plt.scatter(pos_ei_x1, pos_ei_y1, c='lightcoral') # known candidate targets in the test set

        plt.scatter(ei_x2[0], ei_y2[0], c='g')
        plt.scatter(neg_ei_x2, neg_ei_y2, c='b')
        plt.scatter(pos_ei_x2, pos_ei_y2, c='r')

        plt.show()

    eucli_coordinate_set_original = {'drug_x': ei_x1[0], 'drug_y': ei_y1[0],
                                    'neg_target_x': neg_ei_x1, 'neg_target_y': neg_ei_y1,
                                    'pos_target_x': pos_ei_x1, 'pos_target_y': pos_ei_y1}
    eucli_coordinate_set_transformed = {'drug_x': ei_x2[0], 'drug_y': ei_y2[0],
                                    'neg_target_x': neg_ei_x2, 'neg_target_y': neg_ei_y2,
                                    'pos_target_x': pos_ei_x2, 'pos_target_y': pos_ei_y2}

    with open('./outputs/' + 'eucli_coordinate_set_original_{}_{}_{}.pickle'.format(args.fold_name, args.selected_drug_disease[0], args.selected_drug_disease[1]), 'wb') as out_file:
        pickle.dump(eucli_coordinate_set_original, out_file)
    with open('./outputs/' + 'eucli_coordinate_set_transformed_{}_{}_{}.pickle'.format(args.fold_name, args.selected_drug_disease[0], args.selected_drug_disease[1]), 'wb') as out_file:
        pickle.dump(eucli_coordinate_set_transformed, out_file)


def poincare_dimension_reduction(poincare_drug_embedding, poincare_target_embedding):
    es = poincare_drug_embedding[0]
    es_norm = torch.norm(es, p=2, dim=0)
    ei = torch.cat((es.unsqueeze(0), poincare_target_embedding), 0)
    ei_x = ((es.mul(ei).sum(1)) / es_norm).unsqueeze(1)
    ei_norm_square = torch.square(torch.norm(ei, p=2, dim=1))
    ei_x_norm_square = torch.square(torch.norm(ei_x, p=2, dim=1))
    ei_y = torch.sqrt(ei_norm_square - ei_x_norm_square)
    return ei_x.squeeze(1), ei_y


if __name__=='__main__':
    args=parser.parse_args()

    # add or modify some hyper-parameters imported from config.py file
    # 1: common part
    args.seed = 1234
    args.cuda = 0  # default: -1
    args.data = 'DTINet'
    args.path_prefix = r'./data/Cross_validation_split/fold4/' # your original file path
    args.fold_name = 'fold4' # for specifying the storage name of coordinate files (i.e., used to further determine the fold name during visualization)
    args.model_name = 'Eucli_ECFP6_seqsimilarity_fold4.pt'
    args.selected_drug_disease = [515, 449] # based on the relative ids in abs2rel dicts

    for name,value in vars(args).items():
            print(name,value)
    print('*** Above Are All Hyper Parameters ***')

    run_model_visualize_DDT(args)
