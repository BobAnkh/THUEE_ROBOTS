#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author       : BobAnkh
# @Github       : https://github.com/BobAnkh
# @Date         : 2020-10-22 19:29:56
# @LastEditTime : 2020-10-22 19:55:31
# @Description  :
# @Copyright 2020 BobAnkh

import cv2
from sklearn import svm
from skimage.feature import hog
import os
import random
import numpy as np
import json
from tqdm import tqdm
from sklearn.externals import joblib


def SVM_HOG_TRAIN(DATA_TRAIN, model_place='exp1.model'):
    train_data = []
    categroies = os.listdir(DATA_TRAIN)

    # load training data
    for category in categroies:
        path = os.path.join(DATA_TRAIN, category)
        print('loading category :' + category)
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            img = cv2.resize(img, (60, 60))  # average size
            fd = hog(img,
                     orientations=9,
                     pixels_per_cell=(6, 6),
                     cells_per_block=(2, 2),
                     multichannel=True)
            train_data.append((fd, category))

    random.shuffle(train_data)
    print('read data success!')

    # divide into train and validation
    n = int(0.7 * len(train_data))
    train_set = train_data[:n]
    val_set = train_data[n:]
    print('Train_set:', len(train_set), 'Val_set:', len(val_set))

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
    print('Accuracy:', correct / len(X_test))

    # save model
    joblib.dump(classifier, model_place)


def SVM_HOG_TEST(DATA_TEST, model_place='exp1.model'):
    # Load model
    classifier = joblib.load(model_place)

    # Load test data
    test_data = []
    test_imgs = os.listdir(DATA_TEST)
    print('loading test images')
    for i in tqdm(range(len(test_imgs))):
        path = os.path.join(DATA_TEST, test_imgs[i])
        img = cv2.imread(path)
        img = cv2.resize(img, (60, 60))  # average size
        fd = hog(img,
                 orientations=9,
                 pixels_per_cell=(6, 6),
                 cells_per_block=(2, 2),
                 multichannel=True)
        test_data.append(fd)

    print('testing...')
    test_predict = classifier.predict(test_data)
    test_result = {}
    for i in tqdm(range(len(test_imgs))):
        test_result[test_imgs[i]] = test_predict[i]
    json.dump(test_result, open('exp1_test.json', 'w'))


if __name__ == '__main__':
    DATA_TRAIN = '../../image_exp/image_exp/Classification/Data/Train'
    DATA_TEST = '../../image_exp/image_exp/Classification/Data/Test'
    SVM_HOG_TRAIN(DATA_TRAIN)
    SVM_HOG_TEST(DATA_TEST)
