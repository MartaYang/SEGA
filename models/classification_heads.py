import os
import sys

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

from models.SEGAhead_auxilary import LinearDiag, FeatExemplarAvgBlock, SenmanticBlock, TransferBasedBlock
class SEGAhead(nn.Module):
    def __init__(self, weight_generator_type, semantic_path, nFeat, nKall):
        super(SEGAhead, self).__init__()
        print("$$$$$$$$$$$$SEGAhead$$$$$$$$$$$$")
        self.weight_generator_type = weight_generator_type
        self.nFeat = nFeat
        self.nKall = nKall
        self.semantic_path = semantic_path
        print('semantic_path:', semantic_path)
        weight_base = torch.FloatTensor(nKall, nFeat).normal_(0.0, np.sqrt(2.0/nFeat))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)
        scale_cls = 10.0 # cosine similarity temperature t
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)

        print('weight_generator_type:', weight_generator_type)
        if self.weight_generator_type == 'none':
            self.favgblock = FeatExemplarAvgBlock(nFeat)
        elif self.weight_generator_type=='feature_averaging':
            self.favgblock = FeatExemplarAvgBlock(nFeat)
            self.wnLayerFavg = LinearDiag(nFeat)
        elif self.weight_generator_type=='transfer_base_weight':
            scale_att = 10.0
            self.favgblock = FeatExemplarAvgBlock(nFeat)
            self.transferbaseblock = TransferBasedBlock(
                nFeat, nKall, scale_att=scale_att)
            self.wnLayerFavg = LinearDiag(nFeat)
            self.wnLayerWatt = LinearDiag(nFeat)
        elif self.weight_generator_type=='semantic_guided_attention_on_final_visproto':
            scale_att = 10.0
            self.favgblock = FeatExemplarAvgBlock(nFeat)
            self.transferbaseblock = TransferBasedBlock(
                nFeat, nKall, scale_att=scale_att)
            self.semanticblock = SenmanticBlock(nFeat, semantic_path=semantic_path)
            self.lamda = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True)
            self.lamda2 = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True)
        else:
            raise ValueError('Not supported / recognized type {0}'.format(
                self.weight_generator_type))

    def get_classification_weights(self, Kbase_ids, Knovel_ids, features_train=None, labels_train=None):
        if Kbase_ids is not None:
            batch_size, nKbase = Kbase_ids.size()
            weight_base = self.weight_base[Kbase_ids.view(-1)]
            weight_base = weight_base.view(batch_size, nKbase, -1)
        else:
            batch_size, _ = Knovel_ids.size()
            weight_base = self.weight_base.repeat(batch_size,1,1)
        #***********************************************************************

        if features_train is None or labels_train is None:
            # If training data for the novel categories are not provided then
            # return only the classification weights of the base categories.
            return weight_base

        _, num_train_examples, num_channels = features_train.size()
        nKnovel = labels_train.size(2)

        # features_train normalization for generalization
        features_train = F.normalize(features_train, p=2, dim=features_train.dim()-1, eps=1e-12)

        if self.weight_generator_type=='none':
            weight_novel = self.favgblock(features_train, labels_train)
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        elif self.weight_generator_type=='feature_averaging':
            weight_novel_avg = self.favgblock(features_train, labels_train)
            weight_novel = self.wnLayerFavg(
                weight_novel_avg.view(batch_size * nKnovel, num_channels)
            )
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        elif self.weight_generator_type=='transfer_base_weight':
            weight_novel_avg = self.favgblock(features_train, labels_train)
            weight_novel_avg = self.wnLayerFavg(
                weight_novel_avg.view(batch_size * nKnovel, num_channels)
            )
            weight_base_tmp = F.normalize(
                weight_base, p=2, dim=weight_base.dim()-1, eps=1e-12)

            weight_transfer_from_base = self.transferbaseblock(
                features_train, labels_train, weight_base_tmp, Kbase_ids)
            weight_transfer_from_base = self.wnLayerWatt(
                weight_transfer_from_base.view(batch_size * nKnovel, num_channels)
            )
            weight_novel = weight_novel_avg + weight_transfer_from_base
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        elif self.weight_generator_type=='semantic_guided_attention_on_final_visproto':
            weight_novel_avg = self.favgblock(features_train, labels_train)
            weight_base_tmp = F.normalize(
                weight_base, p=2, dim=weight_base.dim()-1, eps=1e-12)
            weight_transfer_from_base = self.transferbaseblock(
                features_train, labels_train, weight_base_tmp, Kbase_ids)
            weight_novel_sensor = self.lamda * weight_novel_avg + self.lamda2 * weight_transfer_from_base
            novelweight_att = self.semanticblock(weight_novel_sensor, Knovel_ids, labels_train)
            weight_novel = weight_novel_sensor * novelweight_att
            if Kbase_ids is not None:
                weight_base = weight_base_tmp * self.semanticblock.get_baseweight_att(Kbase_ids)
        else:
            raise ValueError('Not supported / recognized type {0}'.format(
                self.weight_generator_type))

        # Concatenate the base and novel classification weights and return them.
        if Kbase_ids is not None:
            weight_both = torch.cat([weight_base, weight_novel], dim=1)
        else:
            weight_both = weight_novel
        # weight_both shape: [batch_size x (nKbase + nKnovel) x num_channels]

        return weight_both

    def apply_classification_weights(self, features, cls_weights, l2_norm=True):
        if l2_norm:
            features = F.normalize(
                features, p=2, dim=features.dim()-1, eps=1e-12)
            cls_weights = F.normalize(
                cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)
            cls_scores = self.scale_cls * torch.bmm(features, cls_weights.transpose(1,2))
        else:
            cls_scores = torch.bmm(features, cls_weights.transpose(1,2))

        return cls_scores

    def forward(self, query, support, support_labels, n_way, n_shot, Kall, nKbase, normalize=True, l2_norm=True, *args, **kwargs):
        tasks_per_batch = query.size(0)
        n_query = query.size(1)
        d = query.size(2)
        assert(query.dim() == 3)
        if support is not None:
            n_support = support.size(1)
            assert(support.dim() == 3)
            assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
            assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
        
            support_labels_one_hot = one_hot((support_labels-nKbase).view(tasks_per_batch * n_support), n_way)
            support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
        else:
            support_labels_one_hot = None
        
        # get base and novel class IDs to identify corresponding semantic embeddings
        Kall = Kall.long()
        Kbase_ids = (None if (nKbase==0) else Variable(Kall[:,:nKbase].contiguous(), requires_grad=False))
        Knovel_ids = Variable(Kall[:,nKbase:].contiguous(), requires_grad=False)

        cls_weights = self.get_classification_weights(
            Kbase_ids, Knovel_ids, support, support_labels_one_hot)
        logits = self.apply_classification_weights(
            query, cls_weights, l2_norm)

        return logits



def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))

def CosineClassfier(query, support, support_labels, n_way, n_shot, normalize=True, *args, **kwargs):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and 
    returns the classification score (=L2 distance to each class prototype) on the query set.
    
    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)
    
    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    
    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    #************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1,2)
    # Batch matrix multiplication:
    #   prototypes = labels_train_transposed * features_train ==>
    #   [batch_size x nKnovel x num_channels] =
    #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    # Cosine similarity
    query = F.normalize(query, p=2, dim=query.dim()-1, eps=1e-12)
    prototypes = F.normalize(prototypes, p=2, dim=prototypes.dim()-1, eps=1e-12)
    logits = torch.bmm(query, prototypes.transpose(1,2))
    
    # if normalize:
    #     logits = logits / d

    return logits

def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True, *args, **kwargs):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and 
    returns the classification score (=L2 distance to each class prototype) on the query set.
    
    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)
    
    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    
    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    #************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1,2)
    # Batch matrix multiplication:
    #   prototypes = labels_train_transposed * features_train ==>
    #   [batch_size x nKnovel x num_channels] =
    #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits
    
    if normalize:
        logits = logits / d

    return logits

class ClassificationHead(nn.Module):
    def __init__(self, base_learner='SEGA', semantic_path=None, nfeat=640, enable_scale=True, weight_generator_type=None, nKall=None):
        super(ClassificationHead, self).__init__()
        if ('SEGA' == base_learner):
            self.head = SEGAhead(weight_generator_type, semantic_path, nfeat, nKall)
        elif ('CosineClassfier' == base_learner):
            self.head = CosineClassfier
        elif ('ProtoNet' == base_learner):
            self.head = ProtoNetHead
        else:
            print ("Cannot recognize the base learner type")
            assert(False)
        
        # Add a learnable scale
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        
    def forward(self, query, support, support_labels, n_way, n_shot, *args, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels, n_way, n_shot, *args, **kwargs)
        else:
            return self.head(query, support, support_labels, n_way, n_shot, *args, **kwargs)
