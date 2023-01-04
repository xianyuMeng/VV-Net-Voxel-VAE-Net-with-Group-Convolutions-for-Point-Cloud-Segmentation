from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

# import user defined package
import provider
import pointnet_part_seg as model

def read_object_cat(fname):
    f = open(fname, 'r')
    lines = [line.split('\n') for line in f.readlines()]
    lines = [list(filter(None, line)) for line in lines]
    lines = [line[0].split('\t') for line in lines]
    objcats = [line[1] for line in lines]
    objnames = [line[0] for line in lines]
    return (objcats, objnames)

def object2id(fname):
    oid2cpid = json.load(open(fname, 'r'))
    object2setid = {}
    for idx in range(0, len(oid2cpid)):
        objid, pid = oid2cpid[idx]
        if not objid in object2setid.keys():
            object2setid[objid] = []
        object2setid[objid].append(idx)
    return object2setid

def convert_label_to_one_hot(labels, num_part_cat):
    label_one_hot = np.zeros((labels.shape[0], num_part_cat))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot


def read_color_map(fname):
    color_map = json.load(open(fname, 'r'))
    color_map = np.array(color_map)
    return color_map

def output_color_point_cloud(data, seg, color_map, out_fn):
    f = open(out_fn, 'w')
    [f.write('v %f %f %f %f %f %f\n' % \
            (data[i][0], data[i][1], data[i][2], color_map[seg[i]][0], color_map[seg[i]][1], color_map[seg[i]][2])) for i in range(0, len(seg))]
    f.close()
    return

def output_color_point_cloud_red_blue(data, mask, out_fn):
    num = data.shape[0]
    color = np.zeros((num, 3))
    print(mask.shape)
    for idx in range(num):
        if mask[idx] == 1:
            color[idx, ...] = [0, 0, 1]
        elif mask[idx] == 0:
            color[idx, ...] = [1, 0, 0]
        else:
            color[idx, ...] = [0, 0, 0]
    
    with open(out_fn, 'w') as out:
        [out.write('v %f %f %f %f %f %f\n' % \
                (data[i][0], data[i][1], data[i][2], \
                color[i][0], color[i][1], color[i][2])) for i in range(num)]
        return



    

def placeholder_inputs(batch_size, point_num, num_obj_cat):
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, num_obj_cat))
    voxel_ph = tf.placeholder(tf.float32, shape = ( batch_size, 16, 16, 16, 8))
    #voxel_finer_ph = tf.placeholder(tf.float32, shape = (batch_size, 32, 32, 32))
    return pointclouds_ph, input_label_ph, voxel_ph

def printout(f_log, data):
    print(data)
    f_log.write(data + '\n')

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
        '--checkpoint',
        default = './xxx/epoch_xx.ckpt',
        help = 'path to checkpoint file'
        )
parser.add_argument(
        '--out_dir',
        default = './eval',
        help = 'output directory'
        )
parser.add_argument(
        '--hdf5_data_dir',
        default = './hdf5_data/',
        help = 'path to hdf5 data dir'
        )
parser.add_argument(
        '--gpu_to_use',
        default = 1)
parser.add_argument(
        '--file_list',
        default = 'test_hdf5_file_list.txt',
        help = 'file containing filename'
        )
parser.add_argument(
        '--num_obj_cat',
        default = 16,
        help = 'number of categories'
        )
parser.add_argument(
        '--num_part_cat',
        default = 50)
parser.add_argument(
        '--batch_size',
        default = 1)
parser.add_argument(
        '--num_point',
        type = int,
        default = 2048)
parser.add_argument(
        '--obj_cat_fn',
        default = 'all_object_categories.txt',
        help = 'a list of catogories'
        )
parser.add_argument(
        '--color_map_fn',
        default = 'part_color_mapping.json',
        help = 'color map in json format')
args = parser.parse_args()
 
def main():
    is_training = False
    objcats, objnames = read_object_cat(os.path.join(args.hdf5_data_dir, args.obj_cat_fn))

    object2setid = object2id(os.path.join(args.hdf5_data_dir, 'overallid_to_catid_partid.json'))
    color_map = read_color_map(os.path.join(args.hdf5_data_dir, args.color_map_fn))
    print('\t\b{}\n'.format(object2setid))
    if not os.path.exists(os.path.join(BASE_DIR, args.out_dir)):
        print('\b\nMkdir {}\n'.format(os.path.join(BASE_DIR, args.out_dir)))
        os.mkdir(os.path.join(BASE_DIR,args.out_dir))

    with tf.device('/gpu:' + str(args.gpu_to_use)):
        pointclouds_ph, input_label_ph, voxel_ph = placeholder_inputs(
                args.batch_size,
                args.num_point,
                args.num_obj_cat
                )
        is_training_ph = tf.placeholder(tf.bool, shape=())

        # get model
        pred, seg_pred, end_points = model.get_model(
                pointclouds_ph, 
                input_label_ph,
                voxel_ph,
                cat_num = args.num_obj_cat,
                part_num = args.num_part_cat,
                is_training = is_training_ph, 
                batch_size = args.batch_size,
                num_point = args.num_point,
                weight_decay = 0.0,
                bn_decay = None)

    saver = tf.train.Saver()
    config = tf.ConfigProto(
            allow_soft_placement = True)
    config.gpu_options.allow_growth = True

    test_file_list = provider.getDataFiles(os.path.join(args.hdf5_data_dir,args.file_list))

    with tf.Session(config=config) as sess:
        batch_size = int(args.batch_size)
        f_log = open(os.path.join(args.out_dir, 'log.txt'), 'w')
        printout(f_log, 'Loading model %s' % args.checkpoint)
        saver.restore(sess, args.checkpoint)
        printout(f_log, '\t\bModel Restored\n')

        batch_data = np.zeros(
                [args.batch_size, args.num_point, 3]).astype(np.float32)

        total_acc_iou = 0.0
        total_seen = 0

        total_per_cat_iou = np.zeros((int(args.num_obj_cat))).astype(np.float32)
        total_per_cat_seen = np.zeros((int(args.num_obj_cat))).astype(np.int32)
        out_color_point_cloud_fn = lambda fn : os.path.join(args.out_dir, fn)

        for idx in range(len(test_file_list)):
            cur_test_filename = test_file_list[idx]
            printout(f_log, 'Loading test file ' + cur_test_filename)
            cur_data, cur_labels, cur_seg, cur_voxel = \
                    provider.loadDataFile_with_seg_voxel( cur_test_filename)
            print('\ncur_voxel_shape = {}'.format(cur_voxel.shape))
            out_fn = out_color_point_cloud_fn(cur_test_filename.split('/')[-1])
            printout(f_log, '\tSaving to {}'.format(out_fn))

            cur_labels = np.squeeze(cur_labels).astype(np.int32)
            cur_labels_one_hot = convert_label_to_one_hot(cur_labels, int(args.num_obj_cat))
            num_data = len(cur_labels)
            num_batch = num_data // batch_size
            for j in range(num_batch):
                beg_idx = j * batch_size
                end_idx = (j + 1) * batch_size
                feed_dict = {
                        pointclouds_ph : cur_data[beg_idx : end_idx, ...],
                        input_label_ph : cur_labels_one_hot[beg_idx : end_idx, ...],
                        voxel_ph : cur_voxel[beg_idx : end_idx, ...],
                        is_training_ph : is_training,
                        }
                pred_val, seg_pred_val = sess.run(
                        [pred, seg_pred],
                        feed_dict = feed_dict)
                seg_pred_val = np.argmax(seg_pred_val, axis = -1)


                cur_batch_label = cur_labels[beg_idx : end_idx, ...]
                
                output_color_point_cloud(
                        cur_data[beg_idx : end_idx, ...].reshape(args.num_point, 3),
                        seg_pred_val.reshape(-1),
                        color_map,
                        out_fn + str(j) + '.obj')

                output_color_point_cloud(
                        cur_data[beg_idx : end_idx, ...].reshape(args.num_point, 3),
                        cur_seg[beg_idx : end_idx, ...].reshape(-1).astype(np.int32),
                        color_map,
                        out_fn + str(j) + '.gt.obj')

                output_color_point_cloud_red_blue(
                        cur_data[beg_idx : end_idx, ...].reshape(args.num_point, 3),
                        np.int32(seg_pred_val.reshape(-1) == cur_seg[beg_idx : end_idx, ...].reshape(-1)),
                        out_fn + str(j) + '.diff.obj')


                printout(f_log, '\t\nExample {} {} {}'.format(j, cur_batch_label, float(np.sum(seg_pred_val == cur_seg[beg_idx : end_idx, ...]) / args.num_point)))

                cur_batch_part_cats = [object2setid[objcats[cur_batch_label[ind]]]for ind in range(batch_size)]
                cur_batch_seg = cur_seg[beg_idx : end_idx, ...]
                
                total_iou = 0.0
                printout(f_log, "\t\bseg_pred_val {}\n".format(np.unique(seg_pred_val)))
                printout(f_log, '\t\bcur_batch_part_cats {}\n'.format(cur_batch_part_cats))
                total_intersect = 0.0
                total_union = 0.0
                for cat in cur_batch_part_cats[0]:
                    n_pred = np.sum(seg_pred_val == cat)
                    n_ground = np.sum(cur_batch_seg == cat)
                    n_intersect = np.sum(np.int32(cur_batch_seg == cat) * np.int32(seg_pred_val == cur_batch_seg))
                    n_union = n_pred + n_ground - n_intersect
                    #printout(f_log, '\t\bn_pred %f n_ground %f n_intersect %f\n' % ( n_pred, n_ground, n_intersect))
#                    if n_union == 0:
#                        total_iou += 1.0
#                    else:
#                        total_iou += n_intersect * 1.0 / n_union
                    total_intersect += n_intersect
                    total_union += n_union

                total_seen += 1.0
#                ave_iou = total_iou / len(cur_batch_part_cats[0])
                ave_iou = total_intersect / total_union
                total_acc_iou += ave_iou
                for ind in range(batch_size):
                    total_per_cat_seen[cur_batch_label[ind]] += 1
                    total_per_cat_iou[cur_batch_label[ind]] += ave_iou


            printout(f_log, '\t\bIoU: %f\n' % (total_acc_iou / total_seen))


        for cat_idx in range(int(args.num_obj_cat)):
            printout(f_log, '\t\b' + objnames[cat_idx] + ' Total Number : ' + str(total_per_cat_seen[cat_idx]))
            printout(f_log, '\t\b' + objnames[cat_idx] + ' IoU : ' + \
                    str(total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]))
                






with tf.Graph().as_default():
    main()

