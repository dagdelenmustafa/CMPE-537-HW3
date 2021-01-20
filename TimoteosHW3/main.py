from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix

import os
import glob
import math
import cv2
import numpy as np
import pickle
import time

def features(image, extractor):
    return extractor.detectAndCompute(image, None)

def main():
    train_dir = './Caltech20/training'
    test_dir = './Caltech20/testing'

    # clustering(n_clusters=100, root_dir=train_dir)

    # defining feature extractor that we want to use
    extractor = cv2.SIFT_create()

    kmeans = pickle.load(open("kmeans_100.pkl", "rb"))

    clf = train(kmeans, train_dir)

    test(clf, test_dir)

if __name__ == '__main__':
    main()
