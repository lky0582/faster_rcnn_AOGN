#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN with OHEM
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Abhinav Shrivastava
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb


import caffe
import argparse
import pprint
import time, os, sys
import re
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=1, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--det_thresh', dest='det_thresh',
                        help='detection score threshold',
                        default=0.05, type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print(('Waiting for {} to exist...'.format(args.caffemodel)))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis, thresh=args.det_thresh)

    #遍历所有caffemodel进行测试mAP
    # min_num = 30000
    # max_num = 60000
    # interval = 500
    # model_dir = "/data/li/output/faster_rcnn_senet/"
    # for file in os.listdir(model_dir):
    #     if file.endswith(".caffemodel"):
    #         file_number = int(re.findall('\d+', file)[0])
    #         if min_num <= file_number <= max_num:
    #             if file_number % interval == 0:
    #                 model_path = os.path.join(model_dir, file)
    #                 print ("Testing model:", file)
    #                 print('Called with args:')
    #                 print(args)
    #                 imdb = get_imdb(args.imdb_name)
    #                 imdb.competition_mode(args.comp_mode)
    #                 if not cfg.TEST.HAS_RPN:
    #                      imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    #                 caffe.set_mode_gpu()
    #                 caffe.set_device(args.gpu_id)
    #                 net = caffe.Net(args.prototxt, model_path, caffe.TEST)
    #                 net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    #                 test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis, thresh=args.det_thresh)
    min_num = 10000
    max_num = 50000
    interval = 2500
    model_dir_ls = os.listdir("/data/li/output/faster_rcnn/")
    model_dir = "/data/li/output/faster_rcnn/"
    files_with_num = []
    for file in model_dir_ls:
        if file.endswith(".caffemodel"):
            num = re.findall(r'\d+', file)
            if int(num[0]) % interval  == 0:
                files_with_num.append((file, int(num[0])))
    files_with_num_sorted = sorted(files_with_num, key=lambda x: x[1])
    for file_with_num in files_with_num_sorted:
        file_number = int(re.findall('\d+', file_with_num[0])[0])
        if min_num <= file_number <= max_num:
            model_path = os.path.join(model_dir, file_with_num[0])
            print ("Testing model:", file_with_num[0])
            print('Called with args:')
            print(args)
            imdb = get_imdb(args.imdb_name)
            imdb.competition_mode(args.comp_mode)
            if not cfg.TEST.HAS_RPN:
                    imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
            caffe.set_mode_gpu()
            caffe.set_device(args.gpu_id)
            net = caffe.Net(args.prototxt, model_path, caffe.TEST)
            net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
            test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis, thresh=args.det_thresh)

