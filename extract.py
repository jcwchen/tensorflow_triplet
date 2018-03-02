import numpy as np
import os,sys
import tensorflow as tf
import os.path as osp
import multiprocessing as mp

import model
import IPython
import data
from data import transform_img
import cv2, pickle



flags = tf.app.flags

flags.DEFINE_string('feature', 'fc6', 'Extract which layer(pool5, fc6, fc7)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_string('file', None, 'File for feature extraction')


FLAGS = flags.FLAGS

layer_list = [FLAGS.feature]

output_root = "triplet_feature"
 

config = tf.ConfigProto()
config.gpu_options.allow_growth=True


with tf.Graph().as_default(), tf.Session(config=config) as sess:
    
    img_input  = tf.placeholder('float32', shape=(None, 227, 227, 3))

    feature = model.inference(img_input, 1, FLAGS.feature, False)

    norm_cross_pred = model.feature_normalize([feature])
    pred = norm_cross_pred[0]

    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.model_dir)


    img_name = FLAGS.file.replace(".jpg","").replace(".png","")
    print("Load image: {}".format(img_name))
    img_name = osp.basename(img_name)

    img = cv2.imread(FLAGS.file, cv2.IMREAD_COLOR)
    img = transform_img(img, 227,227)

    for layer in layer_list:
        output_layer = osp.join(output_root, layer)
        if not osp.exists(output_layer):
            os.makedirs(output_layer)

        out, input= sess.run([pred, img_input], feed_dict={img_input: [img]})
        out = np.array(out[0])
        IPython.embed()

        with open(os.path.join(output_layer, img_name+'.pkl'), 'wb') as ff:
            pickle.dump(out, ff)

