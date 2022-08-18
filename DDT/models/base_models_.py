import numpy as np
import torch
import torch.nn as nn
from layers.hyp_layers import LorentzLinear
from manifolds.lorentz_ import Lorentz
import math
from geoopt import ManifoldParameter
from models.base_models import Target_Model, lorentz_linear4decoder

class HyboNet4KG_(nn.Module):
    def __init__(self, target_features_list, processed_adj, abs2_rel_list, args, dim, max_scale, max_norm, margin):
        super(HyboNet4KG_, self).__init__()
        self.manifold = Lorentz(max_norm=max_norm)
        self.selected_drugs_abs2rel = abs2_rel_list[0]
        self.selected_diseases_abs2rel = abs2_rel_list[1]
        self.selected_targets_abs2rel = abs2_rel_list[2]

        self.drug_entity = ManifoldParameter(self.manifold.random_normal((len(self.selected_drugs_abs2rel), dim), std=1. / math.sqrt(dim)), manifold=self.manifold)

        # for target entity
        self.target_pretrained = args.target_pretrained
        if self.target_pretrained[0] == True:
            target_entity = np.load('./data/target_pretrained_storage/' + 'target_{}_embeddings.npy'.format(self.target_pretrained[1]))
            # assert target_entity.shape[1] == args.dim, 'The pretrained target embeddings do
            # not have the same dimension with that of the specified dimension hyper-parameter'
            target_entity = torch.FloatTensor(target_entity).to(args.device)

            if self.target_pretrained[1] == 'similarity' and self.target_pretrained[2] == 'expmap':
                o = torch.zeros_like(target_entity)
                self.target_entity = torch.cat([o[:, 0:1], target_entity], dim=1)
            else:
                self.target_entity = target_entity
            print('Name of current loaded target embedding file:', 'target_{}_embeddings.npy'.format(self.target_pretrained[1]))
            print('Current loaded target embedding dimension:', self.target_entity.shape[1], 'Current specified model hidden state dimension:', args.dim)  # 此时target的维度已经不会发生变化

            if self.manifold.name == 'Lorentz':
                if self.target_pretrained[1] == 'similarity' and self.target_pretrained[2] == 'expmap':
                    self.target_entity = self.manifold.expmap0_without_norm(self.target_entity)
                self.target_transform = LorentzLinear(self.manifold, self.target_entity.size(1), args.dim, args.bias, args.dropout, nonlin=None)

        else:
            self.target_transform = Target_Model(target_features_list, self.selected_targets_abs2rel, processed_adj, args)

        print('*Target Model Structure*')
        print(self.target_transform)

        # initialize target fixed embedding generation model
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
        drug_entity = self.drug_entity

        head = drug_entity[u_idx]
        tail = target_entity[v_idx]
        r_bias = self.relation_bias[r_idx]
        r_transform = self.relation_transform[r_idx]
        head = lorentz_linear4decoder(head.unsqueeze(1), r_transform, self.scale, r_bias.unsqueeze(1)).squeeze(1)
        neg_dist = (self.margin + 2 * self.manifold.cinner(head.unsqueeze(1), tail).squeeze(1))
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]


class EucliNet4KG(nn.Module):
    def __init__(self, abs2_rel_list, args, dim, margin, extra_bias, drug_pretrained = None, target_pretrained = None, all_drug_morgan = None):
        super(EucliNet4KG, self).__init__()
        self.extra_bias = extra_bias
        self.margin = margin
        self.selected_drugs_abs2rel = abs2_rel_list[0]
        self.selected_diseases_abs2rel = abs2_rel_list[1]
        self.selected_targets_abs2rel = abs2_rel_list[2]
        self.drug_pretrained = drug_pretrained
        self.target_pretrained = target_pretrained
        # for controlling whether need normalization for drug and target entities
        self.drug_norm = False
        self.target_norm = False

        if self.drug_pretrained: # for Eucli_ECFP6_seqsimilarity
            if self.drug_pretrained[0] == True:
                self.drug_entity = all_drug_morgan
                self.drug_transform = nn.Linear(self.drug_entity.size(1), dim, bias = True)
                stdv = 1. / math.sqrt(dim)
                nn.init.uniform_(self.drug_transform.weight, -stdv, stdv)
                nn.init.constant_(self.drug_transform.bias, 0) # bias = True
            else:
                self.drug_entity = torch.nn.Embedding(num_embeddings=len(self.selected_drugs_abs2rel), embedding_dim=dim)
                self.drug_norm = True
        else: # for Eucli_noECFP6_noontology_noppi
            self.drug_entity = torch.nn.Embedding(num_embeddings=len(self.selected_drugs_abs2rel), embedding_dim=dim)
            self.drug_norm = True

        # for target embeddings
        if self.target_pretrained: # Eucli_ECFP6_seqsimilarity
            if self.target_pretrained[0] == True:
                target_entity = np.load('./data/target_pretrained_storage/' + 'target_{}_embeddings.npy'.format(self.target_pretrained[1]))
                self.target_entity = torch.FloatTensor(target_entity).to(args.device)
                self.target_transform = nn.Linear(self.target_entity.size(1), dim, bias=True)
                stdv = 1. / math.sqrt(dim)
                nn.init.uniform_(self.target_transform.weight, -stdv, stdv)
                nn.init.constant_(self.target_transform.bias, 0)
                print('*Target Model Structure*')
                print(self.target_transform)
            else:
                self.target_entity = torch.nn.Embedding(num_embeddings=len(self.selected_targets_abs2rel), embedding_dim=dim)
                self.target_norm = True
        else: # for Eucli_noECFP6_noontology_noppi
            self.target_entity = torch.nn.Embedding(num_embeddings=len(self.selected_targets_abs2rel), embedding_dim=dim)
            self.target_norm = True

        # None: transE, vector: Euclidean counterpart with vector disease representation, matrix: Euclidean counterpart with matrix disease representation
        assert self.extra_bias in [None, 'vector', 'matrix'], 'Extra_bias should be None/vector/matrix.'
        # use the totally same initialization with the Lorentz decoder
        if self.extra_bias == 'matrix':
            self.relation_bias = nn.Parameter(torch.zeros((len(self.selected_diseases_abs2rel), dim)))
            self.relation_transform = nn.Parameter(torch.empty(len(self.selected_diseases_abs2rel), dim, dim))
            nn.init.kaiming_uniform_(self.relation_transform)
        else: # for the case that disease representation is embedding vector instead of matrix
            self.relation_transform = torch.nn.Embedding(num_embeddings=len(self.selected_diseases_abs2rel), embedding_dim=dim)

        # for extra_bias = matrix/vector
        if self.extra_bias != None:
            # used to control the drug/target entity specific bias in the distance score function
            self.bias_head = torch.nn.Parameter(torch.zeros(len(self.selected_drugs_abs2rel)))
            self.bias_tail = torch.nn.Parameter(torch.zeros(len(self.selected_targets_abs2rel)))

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.__data_init()

    def __data_init(self):
        # entity look up table norm
        if self.drug_norm == True:
            nn.init.xavier_uniform_(self.drug_entity.weight.data) # Xavier initalization
        if self.target_norm == True:
            nn.init.xavier_uniform_(self.target_entity.weight.data)
        self.normalization_ent_embedding()

        if self.extra_bias != 'matrix':
            nn.init.xavier_uniform_(self.relation_transform.weight.data)
            self.normalization_rel_embedding()

    def normalization_ent_embedding(self):
        if self.drug_norm == True:
            drug_norm = self.drug_entity.weight.detach().cpu().numpy()
            norm_weight = np.sqrt(np.sum(np.square(drug_norm), axis=1, keepdims=True))
            drug_norm = drug_norm / norm_weight
            self.drug_entity.weight.data.copy_(torch.from_numpy(drug_norm))

        if self.target_norm == True:
            target_norm = self.target_entity.weight.detach().cpu().numpy()
            norm_weight = np.sqrt(np.sum(np.square(target_norm), axis=1, keepdims=True))
            target_norm = target_norm / norm_weight
            self.target_entity.weight.data.copy_(torch.from_numpy(target_norm))

    def normalization_rel_embedding(self):
        norm = self.relation_transform.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.relation_transform.weight.data.copy_(torch.from_numpy(norm))

    def forward(self, u_idx, r_idx, v_idx):
        # for drug embeddings
        if self.drug_pretrained:
            if self.drug_pretrained[0] == True:
                # for the case: drug_pretrained exists & drug_pretrained[0] == True
                drug_entity = self.drug_transform(self.drug_entity)
                head = drug_entity[u_idx]
            else:
                drug_entity = self.drug_entity
                head = drug_entity(u_idx)
        else:
            drug_entity = self.drug_entity
            head = drug_entity(u_idx)

        # for target embeddings
        if self.target_pretrained:
            if self.target_pretrained[0] == True:
                target_entity = self.target_transform(self.target_entity)
                tail = target_entity[v_idx]
            else:
                target_entity = self.target_entity
                tail = target_entity(v_idx)
        else:
            target_entity = self.target_entity
            tail = target_entity(v_idx)

        if self.extra_bias == 'matrix':
            r_bias = self.relation_bias[r_idx]
            r_transform = self.relation_transform[r_idx]
            head = head.unsqueeze(1)@ r_transform.transpose(-2, -1)
            head = (head + r_bias.unsqueeze(1)).squeeze(1)
        else:
            r_transform = self.relation_transform(r_idx)
            head = head + r_transform

        neg_dist = self.margin - torch.sqrt(torch.square(head.unsqueeze(1) - tail).sum(-1))

        if self.extra_bias == None:
            return neg_dist
        else: # drug and target type specific biases for extra_bias = matrix/vector
            return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]










