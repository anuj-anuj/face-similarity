

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep

import numpy as np
import os
from matplotlib import pyplot as plt
import shutil
import cv2
import random

import scipy
import skimage
from skimage.util.shape import view_as_windows
from scipy.ndimage.filters import gaussian_filter

import keras
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Concatenate, Subtract, Dropout, BatchNormalization
from keras.models import Model
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.models import load_model
import time
from keras import backend as K
K.set_image_data_format('channels_first')

from fr_utils import *
from inception_blocks_v2 import *

### Face crop code borrowed from Multi Task CNN
def get_crop(img_path):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor


    img = misc.imread(img_path)
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]

    det = bounding_boxes[:,0:4]
    det_arr = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
    img_center = img_size / 2
    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
    det_arr.append(det[index,:])

    for i, det in enumerate(det_arr):
        det = np.squeeze(det)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-44/2, 0)
        bb[1] = np.maximum(det[1]-44/2, 0)
        bb[2] = np.minimum(det[2]+44/2, img_size[1])
        bb[3] = np.minimum(det[3]+44/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    scaled = misc.imresize(cropped, (96,96), interp='bilinear')
    return scaled
    
def main(args):
    crop1=get_crop(args.image1)
    crop2=get_crop(args.image1)
    crop1 = np.expand_dims(np.around(np.transpose(crop1, (2,0,1))/255.0, decimals=12),axis=0)
    crop2 = np.expand_dims(np.around(np.transpose(crop2, (2,0,1))/255.0, decimals=12),axis=0)
    print('Loading Model....')
    model = load_model(args.modelpath)
    print('Model Loaded.....\n Predicting')

    similiarity=1-model.predict([crop1,crop2])[0][0]
    is_similar= 1 if similiarity>0.5 else 0
    
    print('similiarity: ', similiarity, '\n Is_Similar:',is_similar)





def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image1', type=str, help='Directory with unaligned images.')
    parser.add_argument('image2', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('modelpath', type=str, help='Directory with aligned face thumbnails.')
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


    
