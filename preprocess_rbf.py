from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse
import h5py
import pickle
from glob import glob
import numpy as np
import scipy.io as sio
import re
from data import *

def RBF(point, data, sigma = 1.0, op = np.max):
    data = data - point
    data = data * data
    distance = np.sum(data, axis = 1)
    return op(np.exp(-distance / 2.0 / pow(sigma, 2)))

def _get_coord(num, beg = -1.0, end = 1.0):

    return  np.linspace(beg, end, num = num, retstep = False)

def _get_grid_coord(data, num):
    '''
    data : numpy.ndarray (N, 3)
    num : [int, int, int]
    '''
    beg = np.min(data, axis = 0)
    end = np.max(data, axis = 0)
    coords = [_get_coord(num[ind], beg = beg[ind], end = end[ind]) for ind in range(3)]
    return coords


def _cal_voxel(coord, data, fn = RBF, sigma = 1):
    '''
    coord : return value of func _get_grid_coord
    '''
    res = [[[ 
        fn(np.array([coord[0][ii], coord[1][jj], coord[2][kk]]), data, sigma = sigma)
        for kk in range(len(coord[2]))]
        for jj in range(len(coord[1]))]
        for ii in range(len(coord[0]))]
    
    res = np.array(res)
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--in_dir',
            default = './data/shapenet_part_seg_hdf5_data/hdf5_data',
            help = 'path to h5 file which containes only one training example')
    parser.add_argument(
            '--out_dir',
            default = './data/shapenet_part_seg_hdf5_data/rbf',
            help = 'path to pkl file which contains the result of rbf kernel')
    parser.add_argument(
            '--split_num',
            type = int,
            default = [16, 16, 16],
            nargs = '+',
            help = 'split number along axis')
    parser.add_argument(
            '--inner_num',
            default = 4,
            help = 'inner voxel num')
    parser.add_argument(
            '--sigma',
            default = 0.5,
            help = 'sigma of RBF kernel')
    parser.add_argument(
            '--op_fn',
            default = 'max',
            help = 'function to get rbf result')
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.in_dir, "*.h5")))
    split_num = np.array(args.split_num)*args.inner_num
    sigma = float(args.sigma)
    print(split_num)
    mkdir(args.out_dir) 
    for f in files:
        data = load_data(f)
        data = data['data']
        rbf = []
        num = 1
        for m in data:
            print('processing %i point cloud data' % num)
            coord = _get_grid_coord(m, split_num)
            res = _cal_voxel(coord, m, sigma = sigma)
            rbf.append(res)
            num = num + 1
            if num % 100 == 0:
                save_data(os.path.join(args.out_dir, re.split('[\\\\/]', f)[-1]),{'rbf':np.array(rbf)})
        save_data(os.path.join(args.out_dir, re.split('[\\\\/]', f)[-1]),{'rbf':np.array(rbf)})
        

        

if __name__ == "__main__":
    main()