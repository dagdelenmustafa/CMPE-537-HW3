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

def test(clf, test_dir):
    X_test = np.zeros((1, n_clusters))
    y_test = []

    for root, subdirs, files in os.walk(test_dir):
        subdirs.sort()
        
        for subdir in subdirs:
            class_ = subdir
            subdir = os.path.join(root, subdir)

            for file in sorted(os.listdir(subdir)):

                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    image = cv2.imread(os.path.join(subdir, file))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # detect the keypoints and their descriptor vectors
                    keypoints, descriptors = features(gray, extractor)

                    if descriptors is not None:
                        descriptors = descriptors.astype(np.float)
                        pred = list(kmeans.predict(descriptors))

                        X_test = np.vstack((X_test, [pred.count(n) / descriptors.shape[0] for n in range(n_clusters)]))
                        y_test.append(class_)
        break

    y_test = np.array(y_test)
    X_test = X_test[1:]

    print('Test Accuracy: {}'.format(clf.score(X_test, y_test)))
