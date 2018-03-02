import numpy as np
import os,sys
import tensorflow as tf
import time
from datetime import datetime
import os
import os.path as osp
import sklearn.metrics as metrics
import cv2, random

import model
import IPython

from scipy import spatial


from data import Dataset, TripletData, PairData
import data
import shutil

flags = tf.app.flags

flags.DEFINE_string('feature', 'fc6', 'Extract which layer(pool5, fc6, fc7)')
flags.DEFINE_integer('batch_size', 64, 'Value of batch size')
flags.DEFINE_boolean('remove', False, 'Remove invalid triplet or not')
flags.DEFINE_float('lr', 0.00005, 'learing rate')
flags.DEFINE_float('dropout', 0.5, 'Training dropout rate')
flags.DEFINE_boolean('da', False, 'Data augmentation')
flags.DEFINE_string('train_dir', 'IG-City8', 'Directory path of training data')
flags.DEFINE_string('model', '', 'model path')

max_epo = 20
SAVE_INTERVAL = 1
print_iter = 10000 # > batch size


FLAGS = flags.FLAGS
 
parameter_name = osp.join(FLAGS.train_dir,
    "{}/{}/{}/{}".format(FLAGS.feature,
    FLAGS.batch_size, FLAGS.lr, FLAGS.remove))

def modelpath(model_name):
    return "model/{}/{}".format(parameter_name, model_name)

def finish_training(saver, sess, epoch):
    print("finish training at {}".format(epoch))
    saver.save(sess, modelpath("final_{}".format(epoch)))

def train(dataset_train, dataset_val, ckptfile='', caffemodel=''):
    print('Training start...')
    is_finetune = bool(ckptfile)
    batch_size = FLAGS.batch_size


    path = modelpath("")
    if not os.path.exists(path):
        os.makedirs(path)    


    with tf.Graph().as_default():

        startstep = 0 #if not is_finetune else int(ckptfile.split('-')[-1])
        global_step = tf.Variable(startstep, trainable=False)
         
        # placeholders for graph input

        anchor   = tf.placeholder('float32', shape=(None, 227, 227, 3))
        positive = tf.placeholder('float32', shape=(None, 227, 227, 3))
        negative = tf.placeholder('float32', shape=(None, 227, 227, 3))

        keep_prob_ = tf.placeholder('float32')

        # graph outputs
        feature_anchor = model.inference(anchor, keep_prob_, FLAGS.feature, False)
        feature_positive = model.inference(positive, keep_prob_, FLAGS.feature)
        feature_negative = model.inference(negative, keep_prob_, FLAGS.feature)
        
        feature_size = tf.size(feature_anchor)/batch_size


        feature_list = model.feature_normalize(
            [feature_anchor, feature_positive, feature_negative])


        loss, d_pos, d_neg, loss_origin = model.triplet_loss(feature_list[0], feature_list[1], feature_list[2])


        # summary
        summary_op = tf.merge_all_summaries()

        training_loss = tf.placeholder('float32', shape=(), name='training_loss')
        training_summary = tf.scalar_summary('training_loss', training_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.lr).minimize(loss) #batch size 512
        

        #validation
        validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
        validation_summary = tf.scalar_summary('validation_loss', validation_loss)



        init_op = tf.initialize_all_variables()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as sess:


            saver = tf.train.Saver(max_to_keep=max_epo)
            if ckptfile:
                # load checkpoint file
                saver.restore(sess, ckptfile)
                print('restore variables done')
            elif caffemodel:
                # load caffemodel generated with caffe-tensorflow
                sess.run(init_op)
                model.load_alexnet(sess, caffemodel)
                print('loaded pretrained caffemodel:{}'.format(caffemodel))
            else:
                # from scratch
                sess.run(init_op)
                print('init_op done')

            summary_writer = tf.train.SummaryWriter("logs/{}/{}/{}".format(
                FLAGS.train_dir, FLAGS.feature, parameter_name),
                                                graph=sess.graph)

            epoch = 1
            global_step = step = print_iter_sum =0
            min_loss = min_test_loss = sys.maxint
            loss_sum = []

            while True:

                batch_x, batch_y, batch_z, isnextepoch, start, end = dataset_train.sample_path2img(batch_size)
                
                step += len(batch_x)
                global_step += len(batch_x)
                print_iter_sum += len(batch_x)

                feed_dict = {anchor  : batch_x,
                             positive: batch_y,
                             negative: batch_z,
                             keep_prob_: FLAGS.dropout } # dropout rate

                _, loss_value, pos_value, neg_value, origin_value, anchor_value= sess.run(
                        [optimizer, loss, d_pos, d_neg, loss_origin, feature_list[0]],
                        feed_dict=feed_dict)
                loss_value = np.mean(loss_value)
                loss_sum.append(loss_value)

                if print_iter_sum/print_iter >= 1:
                    loss_sum = np.mean(loss_sum)
                    print('epo{}, {}/{}, loss: {}'.format(
                        epoch, step, len(dataset_train.data), loss_sum))
                    print_iter_sum -= print_iter
                    loss_sum = []

                loss_valuee = sess.run(training_summary,
                                        feed_dict={training_loss: loss_value})

                summary_writer.add_summary(loss_valuee, global_step)
                summary_writer.flush()

                action = 0
                if FLAGS.remove and loss_value == 0:
                    action = dataset_train.remove(start, end)
                    if action == 1:
                        finish_training(saver, sess, epoch)
                        break                        


                if isnextepoch or action == -1:

                    val_loss_sum = []
                    isnextepoch = False # set for validation 
                    step = 0
                    print_iter_sum = 0

                    # validation
                    while not isnextepoch:

                        val_x, val_y, val_z, isnextepoch, start, end = dataset_val.sample_path2img(batch_size)
                        val_feed_dict = { 
                                         anchor  : val_x,
                                         positive: val_y,
                                         negative: val_z,
                                         keep_prob_: 1.
                                    }
                        val_loss = sess.run([loss], feed_dict=val_feed_dict)
                        val_loss_sum.append(np.mean(val_loss))

                    dataset_val.reset_sample()
                    val_loss_sum = np.mean(val_loss_sum)
                    print("Validation loss: {}".format(val_loss_sum))

                    summary_val_loss_sum = sess.run(validation_summary,
                                                feed_dict={validation_loss: val_loss_sum})
                    summary_writer.add_summary(summary_val_loss_sum , global_step)



                    # ready to flush
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, global_step)
                    summary_writer.flush()

                    # save by validation
                    if min_loss > val_loss_sum:
                        min_loss = val_loss_sum
                        best_path = modelpath(
                            "val_{}_{}".format(epoch, val_loss_sum))
                        saver.save(sess, best_path)
                        print(best_path)
                    
                    # save by SAVE_INTERVAL
                    elif epoch % SAVE_INTERVAL == 0:
                        path = modelpath(epoch)
                        saver.save(sess, path)
                        print(path)

                    dataset_train.reset_sample()
                    print(epoch)

                    epoch += 1
                    if epoch >= max_epo:
                        finish_training(saver, sess, epoch)
                        break   



def create_triplet(val_ratio = 0):
        
    train_data = []
    landmark_root = FLAGS.train_dir
    landmarks = {}

    for landmark_dir in os.listdir(landmark_root):
        for img_name in os.listdir(osp.join(landmark_root, landmark_dir)):
            if not landmark_dir in landmarks:
                landmarks[landmark_dir] = []
            landmarks[landmark_dir].append(osp.join(landmark_root, landmark_dir, img_name))


    for landmark in landmarks:
        for img in landmarks[landmark]:

            positive = landmarks[landmark][:]
            positive.remove(img)
            negative = dict(landmarks)
            negative.pop(landmark)

            pos_num = len(positive) # triplet with all positve images
            
            for _ in xrange(pos_num):
                img_pos = positive.pop(random.randrange(len(positive)))
                img_neg = random.choice(negative[random.choice(negative.keys())])    
                train_data.append(TripletData(img, img_pos, img_neg))
                sys.stdout.write("\r{:8d}".format(len(train_data)))
                sys.stdout.flush()


    random.shuffle(train_data)
    print("\nFinish loading... size of data: {}".format(len(train_data)))
    

    if val_ratio == 0: # testing
        return Dataset(train_data)
    else: # validation on training data
        split_index = int(len(train_data) *val_ratio)
        return Dataset(train_data[split_index:]), Dataset(train_data[:split_index])


def main(argv):

    print("Loading training data...")
    train_data, val_data = create_triplet(1./5)

    if FLAGS.model: # fine-tune
        train(train_data, val_data, ckptfile=FLAGS.model)
    else:           # from scratch
        train(train_data, val_data, caffemodel='alexnet_imagenet.npy')

if __name__ == '__main__':
    main(sys.argv)

