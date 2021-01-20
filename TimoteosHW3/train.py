from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report

import os
import glob
import math
import cv2
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

def train(kmeans, train_dir):
    X = np.zeros((1, n_clusters))
    y = []

    n_clusters = kmeans.n_clusters

    # classes = {}

    i = 0
    for root, subdirs, files in os.walk(train_dir):
        subdirs.sort()
        for subdir in subdirs:
            class_ = subdir
            subdir = os.path.join(root, subdir)

            # classes[class_] = i
            for file in sorted(os.listdir(subdir)):

                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    image = cv2.imread(os.path.join(subdir, file))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    gray = cv2.GaussianBlur(gray, (5, 5), 0)

                    # detect the keypoints and their descriptor vectors
                    keypoints, descriptors = features(gray, extractor)

                    if descriptors is not None:
                        descriptors = descriptors.astype(np.float)
                        pred = list(kmeans.predict(descriptors))

                        X = np.vstack((X, [pred.count(n) / descriptors.shape[0] for n in range(n_clusters)]))
                        y.append(class_)

            i += 1
        break

    y = np.array(y)
    X = X[1:]
    
    if upsampling:
        values, counts = np.unique(y, return_counts=True)

        n_sample = max(counts)

        from sklearn.utils import resample
        X_train = np.zeros((1, n_clusters))
        y_train = []

        for val in values:
            X_train = np.vstack((X_train, resample(X[y == val], n_samples=n_sample, replace=True, random_state=10)))
            y_train.extend([val] * n_sample)
        X_train = X_train[1:]
        
        clf = AdaBoostClassifier(n_estimators=500, random_state=0, learning_rate=0.05).fit(X_train, y_train)
        
        print('Training Accuracy: {}'.format(clf.score(X_train, y_train)))
    
    else:
        clf = AdaBoostClassifier(n_estimators=500, random_state=0, learning_rate=0.05).fit(X, y)
        
        print('Training Accuracy: {}'.format(clf.score(X, y)))

    return clf
