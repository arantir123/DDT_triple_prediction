import pandas as pd
import networkx as nx
import numpy as np
import pickle
import scipy
import torch
import scipy.sparse as sp
from collections import defaultdict
import os


def read_original_files(path_prefix):
    selected_triples = pd.read_csv(path_prefix + 'selected_triples.csv') # based on absolute ids
    # print('selected_triples numbers:', len(selected_triples))

    in_file = open(path_prefix + 'selected_drugs_abs2rel.pickle', 'rb')
    selected_drugs_abs2rel = pickle.load(in_file)
    in_file.close()
    in_file = open(path_prefix + 'selected_diseases_abs2rel.pickle', 'rb')
    selected_diseases_abs2rel = pickle.load(in_file)
    in_file.close()
    in_file = open(path_prefix + 'selected_targets_abs2rel.pickle', 'rb')
    selected_targets_abs2rel = pickle.load(in_file)
    in_file.close()

    # transform the absolute ids in the triple set to relative ids
    selected_triples_ = []
    for row in np.array(selected_triples):
        temp0 = selected_drugs_abs2rel[row[0]]
        temp1 = selected_diseases_abs2rel[row[1]]
        temp2 = selected_targets_abs2rel[row[2]]
        selected_triples_.append([temp0, temp1, temp2])

    # ECFP6
    if os.path.exists(path_prefix + 'ECFP6_coomatrix.npz'):
        all_drug_morgan = scipy.sparse.load_npz(path_prefix + 'ECFP6_coomatrix.npz')
    else:
        all_drug_morgan = scipy.sparse.coo_matrix(np.zeros((len(selected_drugs_abs2rel), 1024)))

    # target similarity data stored in ./data/
    if os.path.exists('./data/' + 'target_similarity_embeddings.npy'):
        target_similarity = np.load('./data/' + 'target_similarity_embeddings.npy')
    else:
        target_similarity = np.zeros((len(selected_targets_abs2rel), len(selected_targets_abs2rel)))

    # GO and PPI graphs
    if os.path.exists(path_prefix + 'ECFP6_coomatrix.npz'):
        anno_graph = nx.read_gml(path_prefix + 'anno_graph.gml')
        anno_onto_graph = nx.read_gml(path_prefix + 'anno_onto_graph.gml')
        ppi_graph = nx.read_gml(path_prefix + 'ppi_graph.gml')
        anno_onto_ppi_graph = nx.read_gml(path_prefix + 'anno_onto_ppi_graph.gml')
    else:
        anno_graph = nx.DiGraph()
        anno_onto_graph = nx.DiGraph()
        ppi_graph = nx.DiGraph()
        anno_onto_ppi_graph = nx.DiGraph()

    train_idx = np.load(path_prefix + 'train_idx.npy')
    val_idx = np.load(path_prefix + 'val_idx.npy')
    test_idx = np.load(path_prefix + 'test_idx.npy')

    if os.path.exists(path_prefix + 'manual_disease_absid.npy') and os.path.exists(path_prefix + 'rare_disease_absid.npy'):
        manual_disease_absid = np.load(path_prefix + 'manual_disease_absid.npy')
        rare_disease_absid = np.load(path_prefix + 'rare_disease_absid.npy')
        manual_disease_absid_ = [selected_diseases_abs2rel[i] for i in manual_disease_absid]
        rare_disease_absid_ = [selected_diseases_abs2rel[i] for i in rare_disease_absid]
    else:
        manual_disease_absid_ = [0]
        rare_disease_absid_ = [0]

    return selected_triples_, selected_drugs_abs2rel, selected_diseases_abs2rel, selected_targets_abs2rel, all_drug_morgan, target_similarity, \
           [anno_graph, anno_onto_graph, ppi_graph, anno_onto_ppi_graph], train_idx, val_idx, test_idx, manual_disease_absid_, rare_disease_absid_


def tensorize_original_files(selected_triples_, all_drug_morgan, train_idx, val_idx, test_idx, manual_disease_absid_, rare_disease_absid_):
    morgan_values = all_drug_morgan.data
    morgan_indices = np.vstack((all_drug_morgan.row, all_drug_morgan.col))
    i = torch.LongTensor(morgan_indices)
    v = torch.FloatTensor(morgan_values)
    shape = all_drug_morgan.shape
    all_drug_morgan = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    # selected_triples_: list, train_idx: numpy
    train_triples = np.array(selected_triples_)[train_idx]
    val_triples = np.array(selected_triples_)[val_idx]
    test_triples = np.array(selected_triples_)[test_idx]

    test_triples_pd = pd.DataFrame(test_triples, columns=['drug', 'disease', 'target'])
    manual_disease_triples = np.array(test_triples_pd[test_triples_pd['disease'].isin(manual_disease_absid_)])
    rare_disease_triples = np.array(test_triples_pd[test_triples_pd['disease'].isin(rare_disease_absid_)])
    print('selected rare diseases in the training set:', sorted(manual_disease_absid_), sorted(rare_disease_absid_))
    print('selected rare diseases in the test set:', sorted(set(manual_disease_triples[:, 1])), sorted(set(rare_disease_triples[:, 1])))

    train_triples = torch.LongTensor(train_triples)
    val_triples = torch.LongTensor(val_triples)
    test_triples = torch.LongTensor(test_triples)

    manual_disease_triples = torch.LongTensor(manual_disease_triples)
    rare_disease_triples = torch.LongTensor(rare_disease_triples)

    print('train/val/test idx:', train_idx.shape, val_idx.shape, test_idx.shape)
    print('train/val/test triples:', train_triples.size(), val_triples.size(), test_triples.size())
    print('manual/rare disease triples:', manual_disease_triples.size(), rare_disease_triples.size())

    # data split distribution test
    print('data split distribution test results:')
    temp1 = np.array(train_triples)
    temp2 = np.array(val_triples)
    temp3 = np.array(test_triples)
    temp1_r_set = set(temp1[:, 1]) # disease
    temp2_r_set = set(temp2[:, 1])
    temp3_r_set = set(temp3[:, 1])
    temp1_e1_set = set(temp1[:, 0]) # drug
    temp2_e1_set = set(temp2[:, 0])
    temp3_e1_set = set(temp3[:, 0])
    temp1_e2_set = set(temp1[:, 2]) # target
    temp2_e2_set = set(temp2[:, 2])
    temp3_e2_set = set(temp3[:, 2])

    print('Compare Relation:')
    print(len(temp1_r_set), len(temp2_r_set), len(temp3_r_set))
    print('extra relations in val/test sets:', len(temp2_r_set - temp1_r_set), len(temp3_r_set - temp1_r_set))
    print(len(temp1_r_set - temp2_r_set), len(temp1_r_set - temp3_r_set))
    print('Compare Drug/Target Entity:\ndrug entity not included in training set:', len(temp2_e1_set - temp1_e1_set), len(temp3_e1_set - temp1_e1_set))
    print(len(temp1_e1_set - temp2_e1_set), len(temp1_e1_set - temp3_e1_set), len(temp3_e1_set - temp2_e1_set))
    print('target entity not included in training set:', len(temp2_e2_set - temp1_e2_set), len(temp3_e2_set - temp1_e2_set))
    print(len(temp1_e2_set - temp2_e2_set), len(temp1_e2_set - temp3_e2_set), len(temp3_e2_set - temp2_e2_set))
    return train_triples, val_triples, test_triples, manual_disease_triples, rare_disease_triples, all_drug_morgan


def adj_preprocess(args, graph_list, normalize_adj, directed = False, separated = [True, [1,2]], self_loop = True):

    # 0: anno_graph, 1: anno_onto_graph, 2: ppi_graph, 3: anno_onto_ppi_graph
    anno_graph, anno_onto_graph, ppi_graph, anno_onto_ppi_graph = graph_list
    # change the direction of original graphs
    anno_adj = nx.adjacency_matrix(anno_graph).T.todense()
    anno_onto_adj = nx.adjacency_matrix(anno_onto_graph).T.todense()
    # PPI is an undirected graph, while GO is a directed graph
    ppi_adj = nx.adjacency_matrix(ppi_graph).todense()
    anno_onto_ppi_adj = nx.adjacency_matrix(anno_onto_ppi_graph).T.todense()

    # default: undirected graph (i.e., directed = False)
    if directed == False:
        anno_adj = (anno_adj + anno_adj.T)
        anno_onto_adj = (anno_onto_adj + anno_onto_adj.T)
        anno_onto_ppi_adj = (anno_onto_ppi_adj + anno_onto_ppi_adj.T)
        anno_onto_ppi_adj = np.where(anno_onto_ppi_adj <= 1, anno_onto_ppi_adj, 1)

    # adjacency matrix row normalization
    if normalize_adj:
        # process ppi_adj+anno_ontology
        if separated[0] == True:
            if self_loop == True:
                anno_onto_adj = normalize(anno_onto_adj + np.eye(anno_onto_adj.shape[0]))
                ppi_adj = normalize(ppi_adj + np.eye(ppi_adj.shape[0]))
                ppi_adj[ppi_adj > 1] = 1
            else:
                anno_onto_adj = normalize(anno_onto_adj)
                ppi_adj = normalize(ppi_adj)
        else:
            # process anno_onto_ppi_adj
            if self_loop == True:
                anno_onto_ppi_adj = normalize(anno_onto_ppi_adj + np.eye(anno_onto_ppi_adj.shape[0]))
                anno_onto_ppi_adj[anno_onto_ppi_adj > 1] = 1
            else:
                anno_onto_ppi_adj = normalize(anno_onto_ppi_adj)

    adj_list = [anno_adj, anno_onto_adj, ppi_adj, anno_onto_ppi_adj]

    if separated[0] == True:
        return [sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_list[i])).to(args.device) for i in separated[1]]
    else:
        return [sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(anno_onto_ppi_adj)).to(args.device)]

# normalize adj矩阵
def normalize(mx):
    """Row-normalize sparse matrix."""
    # rowsum = np.array(mx.sum(1))
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_er_vocab(data, idxs=[0, 1, 2]):
    """ Return the valid tail entities for (head, relation) pairs
    Can be used to guarantee that negative samples are true negative.
    """
    er_vocab = defaultdict(set)
    for triple in data:
        er_vocab[(triple[idxs[0]], triple[idxs[1]])].add(triple[idxs[2]])
    return er_vocab


# test
if __name__ == '__main__':
    path_prefix = r'./original_data_luo/fold_1/'

    selected_triples_, selected_drugs_abs2rel, selected_diseases_abs2rel, selected_targets_abs2rel, all_drug_morgan, [
        anno_graph, anno_onto_graph, ppi_graph,
        anno_onto_ppi_graph], train_idx, val_idx, test_idx = read_original_files(path_prefix)

    graph_list = [anno_graph, anno_onto_graph, ppi_graph, anno_onto_ppi_graph]

    train_triples, val_triples, test_triples, all_drug_morgan = tensorize_original_files(selected_triples_, all_drug_morgan, train_idx, val_idx, test_idx)

    processed_adj = adj_preprocess(graph_list, normalize_adj=1, directed=False)
