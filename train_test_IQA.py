import os
import argparse
import random
import numpy as np
from HyerIQASolver import HyperIQASolver

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(config):

    # 数据集路径
    folder_path = {
        'live': '/home/ssl/Database/databaserelease2/',
        'csiq': '/root/data/iqa/CSIQ/',
        'tid2013': '/home/ssl/Database/TID2013/',
        'livec': '/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/',
        'koniq-10k': '/home/ssl/Database/koniq-10k/',
        'bid': '/home/ssl/Database/BID/',
        'multilevel': '/root/data/iqa/multilevel/'
    }

    # 原始图片数量
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
        'multilevel': list(range(0, 393)),  # multilevel共有多少组图片
    }
    sel_num = img_num[config.dataset]   # number length of select dataset

    srcc_all = np.zeros(config.train_test_num, dtype=np.float)  # 存放spearman系数
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)  # 存放pearson系数

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        # Randomly select 90% images for training and the rest for testing
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.9 * len(sel_num)))]
        test_index = sel_num[int(round(0.9 * len(sel_num))):len(sel_num)]

        solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index)
        srcc_all[i], plcc_all[i] = solver.train(train_test_num=i)

    srcc_med = np.median(srcc_all)  # 计算spearman中位数
    plcc_med = np.median(plcc_all)  # 计算pearson中位数

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='multilevel', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013|multileve_(mine dataset)')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate。主干网络的学习率')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network。主干网络*lr_ratio是hypernet的学习率')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='Batch size。 batchsize大小')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches。 定义输入backbone网络的图片大小')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')
    parser.add_argument('--resume', dest='resume', type=str, default='./pretrained/multilevel.pth', help='weight from other dataset')

    config = parser.parse_args()
    main(config)

