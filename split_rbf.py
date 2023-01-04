from __future__ import absolute_import 
from __future__ import print_function
from __future__ import division

import os
import glob
import argparse
import pickle
import numpy as np
import tensorflow as tf
import h5py

def printout(log_fn, data):
    print(data + '\n')
    log_fn.write(data + '\n')


def split_voxel(data,split_num):
    '''
    data : np.ndarray of shape (N_x, N_y, N_z)
    split_num : list of 3 ints
    '''
    shape = data.shape
    num = (shape / np.array(split_num)).astype(np.int32)
    res = [[[ data[ii * num[0] : (ii + 1) * num[0], \
                   jj * num[1] : (jj + 1) * num[1], \
                   kk * num[2] : (kk + 1) * num[2]] \
                   for kk in range(split_num[2])] \
                   for jj in range(split_num[1])] \
                   for ii in range(split_num[0])]
    res = np.array(res)
    return res

def _floats_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))


parser = argparse.ArgumentParser()
parser.add_argument(
        '--in_dir',
        default = './data/shapenet_part_seg_hdf5_data/rbf',
        help = 'path to rbf voxel')
parser.add_argument(
        '--out_dir',
        default = './data/shapenet_part_seg_hdf5_data/rbf',
        help = 'path to output tfrecord data dir')
parser.add_argument(
        '--fn_format',
        default = 'train_%d.tfrecord',
        help = 'output tfrecord file format')
parser.add_argument(
        '--num',
        type = int,
        default =  100,
        help = 'number of examples per tfrecord file')
parser.add_argument(
        '--split_num',
        nargs = '+',
        default = [16, 16, 16],
        help = 'split number along axis')
parser.add_argument(
        '--log_fn',
        default = 'log_rbf.txt',
        help = 'path to log file')

args = parser.parse_args()
if not os.path.isdir(args.out_dir):
    print('\t\nMkdir {}'.format(args.out_dir))
    os.mkdir(args.out_dir)
log_fn = open(os.path.join(args.out_dir, args.log_fn), 'w')
files = glob.glob(os.path.join(args.in_dir, "*train*.h5"))
printout(log_fn, '\t\n# files = {}'.format(len(files)))

# seen file count
cnt = 0
# used tfrecord file count
idx = 0
# output tfecord file name
out_record = lambda index : os.path.join(args.out_dir, args.fn_format % index)

record_fname = out_record(idx)
printout(log_fn, '\t[INFO] : Saving to {}'.format(record_fname))
tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)

for ff in files:
    
    data = h5py.File(ff)['rbf']
    print(data.shape)
    for m in data:
        cnt += 1
        print(cnt)
        result = split_voxel(m, args.split_num)
        for ii in range(args.split_num[0]):
            for jj in range(args.split_num[1]):
                for kk in range(args.split_num[2]):
                    example = tf.train.Example(
                            features = tf.train.Features(
                                feature = {
                                    'voxel_rbf' : \
                                            _floats_feature(result[ii, jj, kk, ...].reshape(-1).astype(np.float32))
                                            }
                                )
                            )
                    tfrecord_writer.write(example.SerializeToString())
        if cnt % args.num == 0:
            tfrecord_writer.close()
            idx += 1
            record_fname = out_record(idx)
            tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)
            printout(log_fn, '\t\n[INFO] : Saving to {}'.format(record_fname))

tfrecord_writer.close()


