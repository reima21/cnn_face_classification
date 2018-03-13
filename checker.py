#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2


NUM_CLASSES = 4
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

def inference(images_placeholder, keep_prob):
    """ モデルを作成する関数

    引数:
      images_placeholder: inputs()で作成した画像のplaceholder
      keep_prob: dropout率のplace_holder

    返り値:
      cross_entropy: モデルの計算結果
    """
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(images_placeholder, [-1, 28, 28, 3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

if __name__ == '__main__':
    test_image = []
    for i in range(1, len(sys.argv)):
        img = cv2.imread(sys.argv[i])
        img = cv2.resize(img, (28, 28))
        # 28*28にリサイズされた画像を表示してみる
        
#        cv2.imshow('image', img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()

        test_image.append(img.flatten().astype(np.float32)/255.0)
    test_image = np.asarray(test_image)

    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder("float")

    logits = inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "model1222.ckpt")

    for i in range(len(test_image)):
        pred = np.argmax(logits.eval(feed_dict={
            images_placeholder: [test_image[i]],
            keep_prob: 1.0 })[0])

        probability = logits.eval(feed_dict={images_placeholder: [test_image[i]],keep_prob: 1.0 })[0]

        # 確率を降順でソート
#        probability.sort()
#        print (probability)

        list_salt = ["しお顔", probability[0]]
        list_sauce = ["ソース顔", probability[1]]
        list_soy_sauce = ["しょうゆ顔", probability[2]]
        list_sugar = ["さとう顔", probability[3]]
        
        list_all = []
        list_all.append(list_salt)
        list_all.append(list_sauce)
        list_all.append(list_soy_sauce)
        list_all.append(list_sugar)
        

#        print (sorted(list_all, key=lambda x: x[1], reverse = True))

        list_all_sorted = sorted(list_all, key=lambda x: x[1], reverse = True)

        print ("\n")
        print ("-----------------------------------------------------------------------------")
        print ("  顔の系統分類")
        print ("---------------" + "\n" + "\n")
        print ("*------------------*")
        print ("      分類候補      ")
        print ("*------------------*")
        print ("『しお顔』" + "\n" + "  一重か奥二重のシャープな目元が特徴で中性的な人" + "\n" + "\n" +"『ソース顔』" + "\n" + "  彫りが深く目鼻がくっきりしている洋風の濃い顔立ち" +  "\n" + "\n" + "『しょうゆ顔』" + "\n" + "  切れ長で一重・奥二重をした和風の薄い顔立ち" + "\n" + "\n" + "『さとう顔』" + "\n" + "  いくつになっても少年っぽさが残るベビーフェィスな顔つきの人" + "\n" + "\n" + "\n")
        print ("*------------------*")
        print ("      分類結果      ")
        print ("*------------------*")
        print (list_all_sorted[0][0], ":", list_all_sorted[0][1] *100, "%")
        
        print (list_all_sorted[1][0], ":", list_all_sorted[1][1] *100, "%")
        
        print (list_all_sorted[2][0], ":", list_all_sorted[2][1] *100, "%")
        
        print (list_all_sorted[3][0], ":", list_all_sorted[3][1] *100, "%", "\n", "\n")


        if pred == 0:
            print ("『", sys.argv[1], "』は『しお顔』と予測されます" + "\n")
        elif pred == 1:
            print ("『", sys.argv[1], "』は『ソース顔』と予測されます" + "\n")
        elif pred == 2:
            print ("『", sys.argv[1], "』は『しょうゆ顔』と予測されます" + "\n")
        else:
            print ("『", sys.argv[1], "』は『さとう顔』と予測されます" + "\n")

        print ("-----------------------------------------------------------------------------")
        print ("\n")
