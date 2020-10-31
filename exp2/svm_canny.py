#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author       : BobAnkh
# @Github       : https://github.com/BobAnkh
# @Date         : 2020-10-22 19:29:56
# @LastEditTime : 2020-10-31 09:03:32
# @Description  :
# @Copyright 2020 BobAnkh

import cv2
from sklearn import svm
import os
import random
import numpy as np
# from tqdm import tqdm
from sklearn.externals import joblib
import logging


def SVM_HOG_TRAIN(DATA_TRAIN, model_place='exp2.model', loglevel='DEBUG'):
    '''
    使用SVM+HOG进行训练.

    Args:
        DATA_TRAIN (str): 训练集地址.
        model_place (str, optional): 模型存储的位置. Defaults to 'exp2.model'.
        loglevel (str, optional): log输出的级别,'DEBUG'即全输出,'NOTSET'即无输出. Defaults to 'DEBUG'.
    '''
    if loglevel == 'DEBUG':
        logging.basicConfig(format="[%(levelname)s]%(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="[%(levelname)s]%(message)s", level=logging.NOTSET)
    
    logging.debug('-----------Train start!-----------')
    train_data = []
    categories = os.listdir(DATA_TRAIN)

    # load training data
    for category in categories:
        path = os.path.join(DATA_TRAIN, category)
        load_cat = 'loading category: ' + category
        logging.debug(load_cat)
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            img = cv2.resize(img, (60, 80))  # average size
            blurred = cv2.GaussianBlur(img,(7,7),0)
            fd = cv2.Canny(blurred, 10, 70).reshape(1, -1).squeeze()
            train_data.append((fd, category))
    # 随机调整数据顺序
    random.shuffle(train_data)
    logging.debug('read data success!')

    # divide into train and validation
    n = int(0.8 * len(train_data))
    train_set = train_data[:n]
    val_set = train_data[n:]
    TV_number = 'Train_set: ' + str(len(train_set)) + ' Val_set: ' + str(len(val_set))
    logging.debug(TV_number)

    # unzip dataset
    X_train, Y_train = map(list, zip(*train_set))
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_test, Y_test = map(list, zip(*val_set))
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # SVM
    classifier = svm.SVC()
    classifier.fit(X_train, Y_train)

    predicted = classifier.predict(X_test)

    # Validation
    correct = 0
    for i, correct_result in enumerate(Y_test):
        if predicted[i] == correct_result:
            correct += 1
    acc = 'Accuracy:' + str(correct / len(X_test))
    logging.debug(acc)

    # save model
    joblib.dump(classifier, model_place)
    save_place = 'Model saved as ' + model_place
    logging.debug(save_place)
    logging.debug('-----------Train end!-----------')


def SVM_HOG_TEST(DATA_TEST, model_place='exp2.model', loglevel='DEBUG'):
    '''
    使用训练好的模型进行测试，返回测试类别名称.

    Args:
        DATA_TEST (str): 测试集地址.
        model_place (str, optional): 模型存储的位置. Defaults to 'exp2.model'.
        loglevel (str, optional): log输出的级别,'DEBUG'即全输出,'NOTSET'即无输出. Defaults to 'DEBUG'.

    Returns:
        dict: 字典结构(json)的测试图片及其类别名称.
    '''
    if loglevel == 'DEBUG':
        logging.basicConfig(format="[%(levelname)s]%(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="[%(levelname)s]%(message)s", level=logging.NOTSET)
    logging.debug('-----------Test start!-----------')
    # Load model
    classifier = joblib.load(model_place)

    # Load test data
    test_data = []
    test_imgs = os.listdir(DATA_TEST)
    logging.debug('loading test images')
    for test_img in test_imgs:
        path = os.path.join(DATA_TEST, test_img)
        img = cv2.imread(path)
        img = cv2.resize(img, (60, 80))  # average size
        blurred = cv2.GaussianBlur(img,(7,7),0)
        fd = cv2.Canny(blurred, 10, 70).reshape(1, -1).squeeze()
        test_data.append(fd)
    # test
    logging.debug('Test result:')
    test_predict = classifier.predict(test_data)
    test_result = {}
    for i, test_img in enumerate(test_imgs):
        tmp = test_img + ':' + test_predict[i]
        tmp_dic = {test_img: test_predict[i]}
        test_result.update(tmp_dic)
        logging.debug(tmp)
    logging.debug('-----------Test end!-----------')
    return test_result


if __name__ == '__main__':
    DATA_TRAIN = 'exp2/data/train'
    DATA_TEST = 'exp2/data/test'
    SVM_HOG_TRAIN(DATA_TRAIN)
    test_result = SVM_HOG_TEST(DATA_TEST)
    print(test_result)
