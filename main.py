import os
import sys
current_path=os.path.dirname(os.path.abspath(__file__))
father_path=os.path.split(current_path)[0]
grand_path=os.path.split(father_path)[0]
sys.path.append(current_path)
sys.path.append(father_path)
sys.path.append(grand_path)
import random
import argparse

import torch.cuda
from torch.backends import cudnn
from utils import *

from solver import Solver

def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    datasetpath=config.model_save_path+config.dataset+'/'
    if (not os.path.exists(datasetpath)):
        mkdir(datasetpath)
    for r in range(3):
        roundpath=datasetpath+str(r)+'/'
        if (not os.path.exists(roundpath)):
            mkdir(roundpath)
        solver = Solver(vars(config),roundpath)

        if config.mode == 'train':
            solver.train_stage1()
            solver.train_stage2()
        elif config.mode == 'test':
            solver.test()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num2_epochs', type=int, default=10)

    parser.add_argument('--w_o', type=float, default=1)
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--c_in', type=int, default=25)
    parser.add_argument('--c_out', type=int, default=25)
    parser.add_argument('--d_model', type=int,default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='PSM')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--data_path', type=str, default='./datasets/')
    parser.add_argument('--model_save_path', type=str, default='./DisTI/')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--ngpu', type=int,default=1)
    parser.add_argument('--data_num', type=int,default=0)
    parser.add_argument('--form', type=str, default="trend")

    config = parser.parse_args()
    args = vars(config)

    fix_seed = args['random_seed']  # args是字典形式
    random.seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)


