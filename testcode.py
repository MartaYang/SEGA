# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tqdm import tqdm

from models.ResNet12_embedding import resnet12
from models.ConvNet_embedding import conv4
from models.classification_heads import ClassificationHead

from utils import pprint, set_gpu, Timer, count_accuracy, log

import numpy as np
import os

def get_model(opt):
    # Choose the embedding network
    if opt.network == 'ResNet12':
        if opt.dataset == 'CIFAR_FS' or opt.dataset == 'FC100':
            network = resnet12(avg_pool=opt.avg_pool, drop_rate=0.1, dropblock_size=2).cuda()
        else:
            network = resnet12(avg_pool=opt.avg_pool, drop_rate=0.1, dropblock_size=5).cuda()
        # network = torch.nn.DataParallel(network, device_ids=[0, 1])
    elif opt.network == 'ConvNet':
        network = conv4(in_planes=3, userelu=False, num_stages=4).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if opt.head == 'SEGA':
        cls_head = ClassificationHead(base_learner='SEGA', semantic_path=opt.semantic_path, nfeat=opt.nfeat, \
                                      weight_generator_type=opt.weight_generator_type, nKall=opt.nKall, enable_scale=False).cuda()
    elif opt.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()    
    elif opt.head == 'CosineClassfier':
        cls_head = ClassificationHead(base_learner='CosineClassfier').cuda()    
    else:
        print ("Cannot recognize the classification head type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_test = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CUB':
        from data.CUB_FS import CUB_FS, FewShotDataloader
        dataset_test = CUB_FS(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_test, data_loader)

def test(opt):
    (dataset_test, data_loader) = get_dataset(opt)

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    
    log_file_path = os.path.join(os.path.dirname(opt.load), "test_log.txt")
    log(log_file_path, str(vars(opt)))

    # Define the models
    (embedding_net, cls_head) = get_model(opt)
    
    # Load saved model checkpoints
    if opt.load != 'pretrian-features':
        saved_models = torch.load(opt.load)
        embedding_net.load_state_dict(saved_models['embedding'])
        embedding_net.eval()
        cls_head.load_state_dict(saved_models['head'])
        cls_head.eval()
    
    # Evaluate on test set
    test_accuracies = []
    for i, batch in enumerate(tqdm(dloader_test()), 1):
        data_support, labels_support, data_query, labels_query, Kall, nKbase = [x.cuda() for x in batch]

        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query
        with torch.no_grad():
            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, n_query, -1)

        logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot, Kall=Kall, nKbase=nKbase, l2_norm=True)

        acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())
        
        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)
        
        if i % 50 == 0:
            log(log_file_path, 'Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'\
                  .format(i, opt.episode, avg, ci95, acc))
