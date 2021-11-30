import argparse

def argparse_config_train():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--num-epoch', type=int, default=30,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--nfeat', type=int, default=640,
                            help='number of feature dimension')
    parser.add_argument('--nKall', type=int, default=-1,
                            help='number of all classes')
    parser.add_argument('--nKbase', type=int, default=0,
                            help='number of base classes')
    parser.add_argument('--nTestBase', type=int, default=0,
                            help='number of query examples per testing class')
    parser.add_argument('--epoch_size', type=int, default=1000,
                            help='number of episodes per epoch')
    parser.add_argument('--avg-pool', default=False, action='store_true',
                            help='whether to do average pooling in the last layer of ResNet models')
    parser.add_argument('--milestones', type=list, default=[10,20,25,30],
                            help='learning rate decay milestones (in epoches number)')
    parser.add_argument('--lambdalr', type=list, default=[0.1,0.006,0.0012,0.00024],
                            help='learning rates to be used between above milestones')
    parser.add_argument('--save-path', default='./experiments/tmp',
                            help='path to save log outputs')
    parser.add_argument('--gpu', default='0',
                            help='choose which gpu to be used')
    parser.add_argument('--network', type=str, default='Pretrain',
                            help='choose which embedding network to use. ResNet, ConvNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                            help='choose which classification head to use. SEGA, ProtoNet')
    parser.add_argument('--weight_generator_type', type=str, default='feature_averaging',
                            help='choose which weight generator type to use (for SEGA)')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose dataset to use. miniImageNet, tieredImageNet, CIFAR_FS, CUB')
    parser.add_argument('--semantic_path', type=str, default='No semantic to be used',
                            help='semantic path for current dataset.')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='batch size')
    parser.add_argument('--embnet_pretrainedandfix', default=False, action='store_true',
                            help='whether to load the feature extractor and fix it during the second stage.')
    parser.add_argument('--pretrian_embnet_path', type=str, default=None,
                            help='feature extractor path.')

    args = parser.parse_known_args()[0]
    return args

