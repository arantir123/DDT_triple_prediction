import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import manifolds
import models.encoders as encoders
import scipy.sparse as sp
from layers.hyp_layers import LorentzLinear
from utils.data_read import normalize, sparse_mx_to_torch_sparse_tensor
from manifolds.lorentz_ import Lorentz
import math
from geoopt import ManifoldParameter
from utils.eval_utils import acc_f1


# the base model of graph encoders in which different GCN models can be selected
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """
    # n_nodes: node number, feat_dim: node embedding dimension
    def __init__(self, args, n_nodes, feat_dim):
        super(BaseModel, self).__init__()
        # Lorentz
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))

        self.feat_dim = feat_dim
        # The getattr() method returns the value of the named attribute of an object. If not found, it returns the default value provided to the function.
        self.manifold = getattr(manifolds, self.manifold_name)()

        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            self.feat_dim = self.feat_dim

        self.n_nodes = n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args, self.n_nodes, self.feat_dim)

    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class Target_Model(nn.Module):
    # target_model = Target_Model(target_features_list, self.selected_targets_abs2rel, processed_adj, self.args)
    def __init__(self, target_features_list, selected_targets_abs2rel, processed_adj, args):
        super(Target_Model, self).__init__()
        self.target_graph_mapping = args.target_graph_mapping
        self.processed_adj = processed_adj
        self.GCN_add_layers = args.GCN_add_layers
        self.args = args
        self.dim = args.dim
        self.model = args.model

        # currently node features are still in the Euclidean space
        assert len(target_features_list) == len(self.processed_adj)
        self.manifold = getattr(manifolds, args.manifold)(max_norm=args.max_norm)

        # map Euclidean features to Lorentz space
        if self.target_graph_mapping[-1] == 'expmap':
            print('Use expmap to transform Euclidean graph node initial features to the Lorentz space')
            target_features_list_ = []
            for target_feature in target_features_list:
                target_feature = target_feature.to_dense()
                o = torch.zeros_like(target_feature)
                target_feature = torch.cat([o[:, 0:1], target_feature], dim=1)
                target_feature = self.manifold.expmap0(target_feature)
                # target_feature = target_feature.to_sparse()
                target_features_list_.append(target_feature)
            print('Current target initial embedding dimension (n+1):', target_features_list_[0].size(1), target_features_list_[-1].size(1), 'Current specified model hidden state dimension:', self.dim)
            self.target_features_list = target_features_list_
        else:
            print('Use direct function to transform Euclidean graph node initial features to the Lorentz space')
            print('Current target initial embedding dimension (n):', target_features_list[0].size(1), target_features_list[-1].size(1), 'Current specified model hidden state dimension:', self.dim)
            self.target_features_list = target_features_list

        self.target_layers = nn.ModuleList()
        for target_feature in self.target_features_list:
            n_nodes = target_feature.size(0)
            feat_dim = target_feature.size(1)
            self.target_layers.append(BaseModel(args, n_nodes, feat_dim))
        self.target_num = torch.LongTensor([len(selected_targets_abs2rel.keys())])

        assert self.GCN_add_layers >= 0, 'GCN_add_layers is an int larger than 0.'
        # set GCN_add_layers = 0: do not initialize GCN_add, set >= 1, initialize GCN_add with the layer number specified by 'GCN_add_layers'
        # while what actually determines whether GCN_add is used is the number of target embedding groups in target_embedding_list (len(target_embedding_list) == 2: use, == 1: not use)
        # >=1 (GCN_add exists and has at least one layer), in this case, GCN_add_adj should be initialized with GCN_add layer
        if self.GCN_add_layers:
            # fix the curvature of all HypoNet to -1
            if args.c is not None:
                self.c = torch.tensor([args.c])
                if not args.cuda == -1:
                    self.c = self.c.to(args.device)
            else:
                self.c = nn.Parameter(torch.Tensor([1.]))

            # extra Lorentz GCN based layer for fusing target embeddings under GO and under PPI network
            self.GCN_add = getattr(encoders, self.model)(self.c, self.args, int(self.target_num) * 2, self.dim, self.GCN_add_layers)
            # the adjacent matrix generated for fusing these two groups of target embeddings
            GCN_add_adj = normalize(np.tile(np.eye(int(self.target_num)), (2,2)))
            self.GCN_add_adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(GCN_add_adj))
            if not args.cuda == -1:
                self.GCN_add_adj = self.GCN_add_adj.to(args.device)

    def forward(self):
        target_embedding_list = []
        for i in range(len(self.target_features_list)):
            target_layer = self.target_layers[i]
            target_embeddings = target_layer.encode(self.target_features_list[i], self.processed_adj[i])
            target_embedding_list.append(target_embeddings)

        # start to process generated embedding list
        if len(target_embedding_list) == 1:
            return target_embedding_list[-1]
        # fuse two groups of target embeddings using defined GCN_add
        elif len(target_embedding_list) > 1:
            # [sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(anno_onto_adj)).to(args.device), sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ppi_adj)).to(args.device)]
            ontology_embedding = target_embedding_list[0][:self.target_num, :]
            protein_embedding = target_embedding_list[-1]
            assert ontology_embedding.size(0) == protein_embedding.size(0), 'The shape of ontology embeddings is different from that of protein embeddings.'
            assert self.GCN_add_layers >= 1, 'Need to specify at least one GCN_add layer.'
            # start to fuse embeddings
            # encode input: (self, x, adj)
            fused_embedding = self.GCN_add.encode(torch.cat((ontology_embedding, protein_embedding), 0), self.GCN_add_adj)
            # print(fused_embedding.size(), torch.cat((ontology_embedding, protein_embedding), 0).size(), self.GCN_add_adj.size())
            return self.manifold.projx_with_norm(fused_embedding.reshape(2, self.target_num, -1)[0]) # projx_with_norm: make generated embeddings exactly in the Lorentz manifold
        else:
            raise Exception('Fail to generate target_embedding_list correctly.')


class HyboNet4KG(nn.Module):
    # max scale: hyper-parameter adjusting the scale of time and spatial dimensions
    # hyponet4kg = HyboNet4KG(target_features_list, processed_adj, all_drug_morgan, abs2_rel_list, args, args.dim, args.max_scale, args.margin)
    def __init__(self, target_features_list, processed_adj, all_drug_morgan, abs2_rel_list, args, dim, max_scale, margin):
        super(HyboNet4KG, self).__init__()
        # default curvature: -1
        self.manifold = getattr(manifolds, args.manifold)()
        self.selected_drugs_abs2rel = abs2_rel_list[0]
        self.selected_diseases_abs2rel = abs2_rel_list[1]
        self.selected_targets_abs2rel = abs2_rel_list[2]

        # initialize drug fixed embedding generation model
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(all_drug_morgan)
            all_drug_morgan = torch.cat([o[:, 0:1], all_drug_morgan], dim=1)
            if self.manifold.name == 'Lorentz':
                # This module is often used to retrieve word embeddings using indices. The input to the module is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.
                self.drug_entity = self.manifold.expmap0(all_drug_morgan)
                # linear = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
                self.drug_transform = LorentzLinear(self.manifold, all_drug_morgan.size(1), args.dim, args.bias, args.dropout, nonlin=None)

        # initialize target fixed embedding generation model
        self.target_model = Target_Model(target_features_list, self.selected_targets_abs2rel, processed_adj, args)
        print('*Target Model Structure*')
        print(self.target_model)

        self.relation_bias = nn.Parameter(torch.zeros((len(self.selected_diseases_abs2rel), dim)))
        # Returns a tensor filled with uninitialized data. The shape of the tensor is defined by the variable argument size.
        self.relation_transform = nn.Parameter(torch.empty(len(self.selected_diseases_abs2rel), dim, dim))
        nn.init.kaiming_uniform_(self.relation_transform)
        # https://blog.csdn.net/nyist_yangguang/article/details/120738406
        self.scale = nn.Parameter(torch.ones(()) * max_scale, requires_grad=False)
        self.margin = margin

        # used to control the drug/target entity specific bias in the distance score function
        self.bias_head = torch.nn.Parameter(torch.zeros(len(self.selected_drugs_abs2rel)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(self.selected_targets_abs2rel)))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        target_entity = self.target_model()
        drug_entity = self.drug_transform(self.drug_entity)

        head = drug_entity[u_idx]
        tail = target_entity[v_idx]
        r_bias = self.relation_bias[r_idx]
        r_transform = self.relation_transform[r_idx]
        head = lorentz_linear4decoder(head.unsqueeze(1), r_transform, self.scale, r_bias.unsqueeze(1)).squeeze(1)

        # L model link prediction decoder
        neg_dist = (self.margin + 2 * self.manifold.cinner(head.unsqueeze(1), tail).squeeze(1))
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]


class HyboNet4KG_with_pretrained(nn.Module):
    # hyponet4kg = HyboNet4KG(target_features_list, processed_adj, all_drug_morgan, abs2_rel_list, args, args.dim, args.max_scale, args.margin)
    def __init__(self, target_features_list, processed_adj, all_drug_morgan, abs2_rel_list, args, dim, max_scale, margin):
        super(HyboNet4KG_with_pretrained, self).__init__()
        # default curvature: -1
        self.manifold = getattr(manifolds, args.manifold)()
        self.selected_drugs_abs2rel = abs2_rel_list[0]
        self.selected_diseases_abs2rel = abs2_rel_list[1]
        self.selected_targets_abs2rel = abs2_rel_list[2]

        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            # for drug embeddings
            o = torch.zeros_like(all_drug_morgan)
            all_drug_morgan = torch.cat([o[:, 0:1], all_drug_morgan], dim=1)
            if self.manifold.name == 'Lorentz':
                # This module is often used to retrieve word embeddings using indices. The input to the module is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.
                self.drug_entity = self.manifold.expmap0(all_drug_morgan)
                self.drug_transform = LorentzLinear(self.manifold, self.drug_entity.size(1), args.dim, args.bias, args.dropout, nonlin=None)

            # for target embedddings
            self.target_pretrained = args.target_pretrained
            if self.target_pretrained[0] == True:
                target_entity = np.load('./data/target_pretrained_storage/' + 'target_{}_embeddings.npy'.format(self.target_pretrained[1]))
                # assert target_entity.shape[1] == args.dim, 'The pretrained target embeddings do
                # not have the same dimension with that of the specified dimension hyper-parameter'
                target_entity = torch.FloatTensor(target_entity).to(args.device)

                if self.target_pretrained[1] == 'similarity' and self.target_pretrained[2] == 'expmap':
                    o = torch.zeros_like(target_entity)
                    self.target_entity = torch.cat([o[:, 0:1], target_entity], dim=1)
                else: # for the case in which time dimension is not needed
                    self.target_entity = target_entity
                print('Name of current loaded target embedding file:', 'target_{}_embeddings.npy'.format(self.target_pretrained[1]))
                print('Current loaded target embedding dimension:', self.target_entity.shape[1], 'Current specified model hidden state dimension:', args.dim) # 此时target的维度已经不会发生变化

                if self.manifold.name == 'Lorentz': # for the extension to other manifold
                    if self.target_pretrained[1] == 'similarity' and self.target_pretrained[2] == 'expmap':
                        self.target_entity = self.manifold.expmap0(self.target_entity)
                    self.target_transform = LorentzLinear(self.manifold, self.target_entity.size(1), args.dim, args.bias, args.dropout, nonlin=None)

            else: # for the case target_pretrained[0] == True
                self.target_transform = Target_Model(target_features_list, self.selected_targets_abs2rel, processed_adj, args)

            print('*Target Model Structure*')
            print(self.target_transform)

        else:
            raise NotImplementedError

        self.relation_bias = nn.Parameter(torch.zeros((len(self.selected_diseases_abs2rel), dim)))
        # Returns a tensor filled with uninitialized data. The shape of the tensor is defined by the variable argument size. 产生一些非常接近于0的值
        self.relation_transform = nn.Parameter(torch.empty(len(self.selected_diseases_abs2rel), dim, dim))
        nn.init.kaiming_uniform_(self.relation_transform)
        self.scale = nn.Parameter(torch.ones(()) * max_scale, requires_grad=False)
        self.margin = margin

        # used to control the drug/target entity specific bias in the distance score function
        self.bias_head = torch.nn.Parameter(torch.zeros(len(self.selected_drugs_abs2rel)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(self.selected_targets_abs2rel)))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        # generating drug and target embedding look-up table
        if self.target_pretrained[0] == True:
            target_entity = self.target_transform(self.target_entity)
        else:
            target_entity = self.target_transform()
        drug_entity = self.drug_transform(self.drug_entity)

        head = drug_entity[u_idx]
        tail = target_entity[v_idx]
        r_bias = self.relation_bias[r_idx]
        r_transform = self.relation_transform[r_idx]
        head = lorentz_linear4decoder(head.unsqueeze(1), r_transform, self.scale, r_bias.unsqueeze(1)).squeeze(1)

        # L model link prediction decoder
        neg_dist = (self.margin + 2 * self.manifold.cinner(head.unsqueeze(1), tail).squeeze(1))
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]


# implementation of general format of fully Lorentz linear layer
def lorentz_linear4decoder(x, weight, scale, bias=None):
    x = x @ weight.transpose(-2, -1)
    # args.max_scale = 2.5
    time = x.narrow(-1, 0, 1).sigmoid() * scale + 1.1
    if bias is not None:
        x = x + bias
    x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
    x_narrow = x_narrow / ((x_narrow * x_narrow).sum(dim=-1, keepdim=True) / (time * time - 1)).sqrt()
    x = torch.cat([time, x_narrow], dim=-1)
    return x


# model = HyboNetnoEncoder(abs2_rel_list, args.dim, args.max_scale, args.max_norm, args.margin)
class HyboNetnoEncoder(torch.nn.Module):

    def __init__(self, abs2_rel_list, dim, max_scale, max_norm, margin, all_drug_morgan=None, bias=None):
        super(HyboNetnoEncoder, self).__init__()
        self.manifold = Lorentz(max_norm=max_norm)

        self.selected_drugs_abs2rel = abs2_rel_list[0]
        self.selected_diseases_abs2rel = abs2_rel_list[1]
        self.selected_targets_abs2rel = abs2_rel_list[2]

        self.all_drug_morgan = all_drug_morgan
        if self.all_drug_morgan != None:
            o = torch.zeros_like(all_drug_morgan)
            all_drug_morgan = torch.cat([o[:, 0:1], all_drug_morgan], dim=1)
            # This module is often used to retrieve word embeddings using indices. The input to the module is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.
            # *for comparison with the complete model using ECFP6 and target graphs, change the norm for expmap to 10.0 at first*
            self.drug_entity = self.manifold.expmap0_without_norm(all_drug_morgan)
            self.drug_transform = LorentzLinear(self.manifold, all_drug_morgan.size(1), dim, bias, dropout=0.0, nonlin=None)
        else:
            # ManifoldParameter: Same as torch.nn.Parameter that has information about its manifold.
            # random_normal: Create a point on the manifold, measure is induced by Normal distribution on the tangent space of zero.
            self.drug_entity = ManifoldParameter(self.manifold.random_normal((len(self.selected_drugs_abs2rel), dim), std=1. / math.sqrt(dim)), manifold=self.manifold)

        self.target_entity = ManifoldParameter(self.manifold.random_normal((len(self.selected_targets_abs2rel), dim), std=1. / math.sqrt(dim)), manifold=self.manifold)
        print('drug entity size:', self.drug_entity.size())
        print('target entity size:', self.target_entity.size())
        self.relation_bias = nn.Parameter(torch.zeros((len(self.selected_diseases_abs2rel), dim)))
        self.relation_transform = nn.Parameter(torch.empty(len(self.selected_diseases_abs2rel), dim, dim))
        print('disease matrices sizes:', self.relation_transform.size(), self.relation_bias.size())

        nn.init.kaiming_uniform_(self.relation_transform)
        self.scale = nn.Parameter(torch.ones(()) * max_scale, requires_grad=False) # control the scale of time and spatial dimensions
        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(self.selected_drugs_abs2rel)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(self.selected_targets_abs2rel)))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        if self.all_drug_morgan != None:
            drug_entity = self.drug_transform(self.drug_entity)
        else:
            drug_entity = self.drug_entity
        target_entity = self.target_entity

        head = drug_entity[u_idx]
        tail = target_entity[v_idx]
        r_bias = self.relation_bias[r_idx]
        r_transform = self.relation_transform[r_idx]
        head = lorentz_linear4decoder(head.unsqueeze(1), r_transform, self.scale, r_bias.unsqueeze(1)).squeeze(1)
        neg_dist = (self.margin + 2 * self.manifold.cinner(head.unsqueeze(1), tail).squeeze(1))
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]


class NCModel(Target_Model):
    def __init__(self, target_features_list, selected_targets_abs2rel, processed_adj, args):
        super(NCModel, self).__init__(target_features_list, selected_targets_abs2rel, processed_adj, args)
        self.decoder = LorentzDecoder(self.c, args)
        self.margin = args.margin
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, idx):
        output = self.decoder.decode(h)
        return output[idx]

    def compute_metrics(self, embeddings, idx, torch_idx):
        output = self.decode(embeddings, idx)
        if self.manifold.name == 'Lorentz':
            correct = output.gather(1, torch_idx.unsqueeze(-1))
            loss = F.relu(self.margin - correct + output).mean()
        else:
            loss = F.cross_entropy(output, torch_idx, self.weights)
        acc, f1 = acc_f1(output, torch_idx, average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """
    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class LorentzDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """
    def __init__(self, c, args):
        super(LorentzDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        self.cls = ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim), std=1./math.sqrt(args.dim)), manifold=self.manifold)
        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.n_classes))
        self.decode_adj = False

    # rewrite the decode function in inherited Decoder
    def decode(self, x, adj = None):
        return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias











