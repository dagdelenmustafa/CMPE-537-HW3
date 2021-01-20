from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

import os
import glob
import math
import cv2
import numpy as np
import pickle
import time

def features(image, extractor):
    return extractor.detectAndCompute(image, None)

def clustering(n_clusters, root_dir):
    # defining feature extractor that we want to use
    extractor = cv2.SIFT_create()

    desc = {}
    for root, subdirs, files in os.walk(train_dir):
        subdirs.sort()
        frequency = dict.fromkeys(subdirs, 0)

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
                        if class_ not in desc:
                            desc[class_] = descriptors
                        else:
                            desc[class_] = np.vstack((desc[class_], descriptors))

                        frequency[class_] = frequency[class_] + descriptors.shape[0]

        break

    X = np.zeros((1, 128))

    # balancing issues
    min_ = min(frequency.values())

    for key, val in desc.items():
        n_desc = frequency[key]

        if 10 * min_ < n_desc:
            X = np.vstack((X, val[:10 * min_]))
        else:
            X = np.vstack((X, val))

    X = X[1:]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=1000).fit(X)

    # save the model
    filename = "kmeans_" + str(n_clusters) + ".pkl"
    pickle.dump(kmeans, open(filename, "wb"))
