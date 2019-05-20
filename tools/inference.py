# -*- coding:utf-8 -*-





import os, sys
sys.path.append("../")
import tensorflow as tf
import time
import cv2
import numpy as np
import argparse

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from help_utils.tools import *
from libs.box_utils import draw_box_in_img
from help_utils import tools
from libs.box_utils import coordinate_convert
from tensorflow.core.protobuf import config_pb2

def inference(det_net, data_dir):
    start = time.time()
    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN)# , is_resize = False)

    det_boxes_h, det_scores_h, det_category_h, \
    det_boxes_r, det_scores_r, det_category_r = det_net.build_whole_detection_network(input_img_batch=img_batch,
                                                                                      gtboxes_h_batch=None,
                                                                                      gtboxes_r_batch=None)
    print("img_batch : " + str(img_batch))
    print("det_boxes_h : " + str(det_boxes_h))
    print("det_scores_h : " + str(det_scores_h))
    print("det_category_h : " + str(det_category_h))
    print("det_boxes_r : " + str(det_boxes_r))
    print("det_scores_r : " + str(det_scores_r))
    print("det_category_r : " + str(det_category_r))

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    #restore_ckpt : resnet_v1_101.ckpt no path
    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print("------start session-----")
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            print("restore begin")
            restorer.restore(sess, restore_ckpt)
            print("restore finish")
            # print('restore model')
            # print(restore_ckpt)

        imgs = os.listdir(data_dir)
        print("imgs is : " + str(imgs))
        for i, a_img_name in enumerate(imgs):
            print("\n")
            print("[step : " + str(i) + "\t\timg : " + a_img_name + "]")

            # f = open('./res_icdar_r/res_{}.txt'.format(a_img_name.split('.jpg')[0]), 'w')
            #print("imread : " + a_img_name)
            raw_img = cv2.imread(os.path.join(data_dir,
                                              a_img_name))
            # raw_img = raw_img + np.array(cfgs.PIXEL_MEAN)

            # raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            start = time.time()
            #print("start sess.run")
            #この下のscores達は、ボックスの何かしらのスコアだから　　　つかえねえ！！！
            resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
            det_boxes_r_, det_scores_r_, det_category_r_ = \
                sess.run(
                    [img_batch, det_boxes_h, det_scores_h, det_category_h,
                     det_boxes_r, det_scores_r, det_category_r],
                    feed_dict={img_plac: raw_img},
                    options=config_pb2.RunOptions(
        report_tensor_allocations_upon_oom=True)
                )
            end = time.time()
            # print("score : " + str(det_scores_r_.shape))
            # print("score : " + str(det_scores_r_.shape))
            # for i in range(len(det_scores_r_)):
            #     print(det_scores_r_[i])
            # res_r = coordinate_convert.forward_convert(det_boxes_r_, False)
            # res_r = np.array(res_r, np.int32)
            # for r in res_r:
            #     f.write('{},{},{},{},{},{},{},{}\n'.format(r[0], r[1], r[2], r[3],
            #                                                r[4], r[5], r[6], r[7]))
            # f.close()
            # print(det_boxes_r.astype(np.int64))
            #画像をぼかそうかな
            det_detections_h = draw_box_in_img.draw_box_cv(np.squeeze(resized_img, 0),
                                                           boxes=det_boxes_h_,
                                                           labels=det_category_h_,
                                                           scores=det_scores_h_)
            det_detections_r = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
                                                                  boxes=det_boxes_r_,
                                                                  labels=det_category_r_,
                                                                  scores=det_scores_r_)
            det_detections_r_cont = draw_box_in_img.draw_rotate_box_cv_cont(np.squeeze(resized_img, 0),
                                                                  boxes_ori=det_boxes_r_,
                                                                  boxes_axis=det_boxes_h_,
                                                                  labels=det_category_r_,
                                                                  scores=det_scores_r_,
                                                                  img_name=a_img_name)
            elapsed_time = time.time() - start
            print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
            # draw_box_in_img.ane_get_box(det_boxes_r_)
            # draw_box_in_img.ane_get_box(det_boxes_h_)
            save_dir = os.path.join(cfgs.INFERENCE_SAVE_PATH, cfgs.VERSION)
            tools.mkdir(save_dir)
            cv2.imwrite(save_dir + '/' + a_img_name + '_h.jpg',
                        det_detections_h)
            cv2.imwrite(save_dir + '/' + a_img_name + '_r.jpg',
                        det_detections_r)
            cv2.imwrite(save_dir + '/' + a_img_name + '_r_cont.jpg',
                        det_detections_r_cont)
            # cv2.imwrite(save_dir + '/' + a_img_name + '_original.jpg',
            #             np.squeeze(resized_img, 0))

            view_bar('{} cost {}s'.format(a_img_name, (end - start)), i + 1, len(imgs))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='data path',
                        default='./inference_image/', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                   is_training=False)

    inference(det_net, data_dir=args.data_dir)
