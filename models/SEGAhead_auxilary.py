import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from pdb import set_trace as breakpoint

import os
import pickle

class LinearDiag(nn.Module):
    def __init__(self, num_features, bias=False):
        super(LinearDiag, self).__init__()
        weight = torch.FloatTensor(num_features).fill_(1) # initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out


class FeatExemplarAvgBlock(nn.Module):
    def __init__(self, nFeat):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        # features_train [batch_size, num_train_examples, num_features]
        # labels_train [batch_size, num_train_examples, nKnovel]
        labels_train_transposed = labels_train.transpose(1,2)
        # labels_train_transposed [batch_size, nKnovel, num_train_examples]
        weight_novel = torch.bmm(labels_train_transposed, features_train)
        # weight_novel [batch_size, nKnovel, num_features]
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))
        return weight_novel

class SenmanticBlock(nn.Module): # KEY MODEL OF OURS SEGA
    def __init__(self, nFeat, semantic_path, sem_aug_level=0.0, droprate=0.7):
        super(SenmanticBlock, self).__init__()
        self.nFeat = nFeat
        label2vec_np = np.load(semantic_path)
        assert(label2vec_np.ndim == 2)
        self.Sem_nFeat = label2vec_np.shape[1]
        self.label2vec = nn.Parameter(torch.from_numpy(label2vec_np), requires_grad=False)
        print("SemLayers is an MLP: FC+LeakyReLU+Dropout+FC+Sigmoid")
        self.SemLayers = nn.Sequential(nn.Linear(self.Sem_nFeat, np.min([self.Sem_nFeat, nFeat])), nn.LeakyReLU(0.1), \
                                       nn.Dropout(droprate), \
                                       nn.Linear(np.min([self.Sem_nFeat, nFeat]), nFeat), nn.Sigmoid())
        
    def get_baseweight_att(self, Kbase_ids):
        batch_size, nKbase = Kbase_ids.size()
        baseweight_label2vec = self.label2vec[Kbase_ids.view(-1)].view(batch_size, nKbase, self.Sem_nFeat)
        baseweight_sematt = self.SemLayers(baseweight_label2vec)
        return baseweight_sematt

    def forward(self, weight_novel_in, Knovel_ids, labels_train):
        # choose the corresponding label2vec
        batch_size, _, nKnovel = labels_train.size()
        episodelabel2vec = self.label2vec[Knovel_ids.view(-1)]
        episodelabel2vec = episodelabel2vec.view(batch_size, nKnovel, self.Sem_nFeat)
        sem_att = self.SemLayers(episodelabel2vec)
        assert(sem_att.size()[-1] == weight_novel_in.size()[-1])

        return sem_att


class TransferBasedBlock(nn.Module):
    # Adapted from AttentionBasedBlock of Gidaris & Komodakis, CVPR 2018: 
    # https://github.com/gidariss/FewShotWithoutForgetting/architectures/ClassifierWithFewShotGenerationModule.py
    def __init__(self, nFeat, nK, scale_att=10.0):
        super(TransferBasedBlock, self).__init__()
        self.nFeat = nFeat
        self.queryLayer = nn.Linear(nFeat, nFeat)
        self.queryLayer.weight.data.copy_(
            torch.eye(nFeat, nFeat) + torch.randn(nFeat, nFeat)*0.001)
        self.queryLayer.bias.data.zero_()

        self.scale_att = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_att), requires_grad=True)
        wkeys = torch.FloatTensor(nK, nFeat).normal_(0.0, np.sqrt(2.0/nFeat))
        self.wkeys = nn.Parameter(wkeys, requires_grad=True)


    def forward(self, features_train, labels_train, weight_base, Kbase):
        batch_size, num_train_examples, num_features = features_train.size()
        nKbase = weight_base.size(1) # [batch_size x nKbase x num_features]
        labels_train_transposed = labels_train.transpose(1,2)
        nKnovel = labels_train_transposed.size(1) # [batch_size x nKnovel x num_train_examples]

        features_train = features_train.view(
            batch_size*num_train_examples, num_features)
        Qe = self.queryLayer(features_train)
        Qe = Qe.view(batch_size, num_train_examples, self.nFeat)
        Qe = F.normalize(Qe, p=2, dim=Qe.dim()-1, eps=1e-12)

        if Kbase is not None:
            wkeys = self.wkeys[Kbase.view(-1)] # the keys of the base categoreis
        else:
            wkeys = self.wkeys.repeat(batch_size,1,1)
        wkeys = F.normalize(wkeys, p=2, dim=wkeys.dim()-1, eps=1e-12)
        # Transpose from [batch_size x nKbase x nFeat] to
        # [batch_size x self.nFeat x nKbase]
        wkeys = wkeys.view(batch_size, nKbase, self.nFeat).transpose(1,2)

        # Compute the attention coeficients
        # batch matrix multiplications: AttentionCoeficients = Qe * wkeys ==>
        # [batch_size x num_train_examples x nKbase] =
        #   [batch_size x num_train_examples x nFeat] * [batch_size x nFeat x nKbase]
        AttentionCoeficients = self.scale_att * torch.bmm(Qe, wkeys)
        AttentionCoeficients = F.softmax(
            AttentionCoeficients.view(batch_size*num_train_examples, nKbase), dim=1)
        AttentionCoeficients = AttentionCoeficients.view(
            batch_size, num_train_examples, nKbase)

        # batch matrix multiplications: weight_novel = AttentionCoeficients * weight_base ==>
        # [batch_size x num_train_examples x num_features] =
        #   [batch_size x num_train_examples x nKbase] * [batch_size x nKbase x num_features]
        weight_novel = torch.bmm(AttentionCoeficients, weight_base)
        # batch matrix multiplications: weight_novel = labels_train_transposed * weight_novel ==>
        # [batch_size x nKnovel x num_features] =
        #   [batch_size x nKnovel x num_train_examples] * [batch_size x num_train_examples x num_features]
        weight_novel = torch.bmm(labels_train_transposed, weight_novel)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))

        return weight_novel