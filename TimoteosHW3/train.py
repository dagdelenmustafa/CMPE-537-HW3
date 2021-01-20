from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report

import os
import glob
import math
import cv2
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    # https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/19252430
    # Calvin Duy Canh Tran & georg-un

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(16, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.3f}; misclass={:0.3f}'.format(accuracy, misclass))
    
    plt.show()    

def train(kmeans, train_dir, upsampling=False):
    X = np.zeros((1, n_clusters))
    y = []

    n_clusters = kmeans.n_clusters

    classes = {}

    i = 0
    for root, subdirs, files in os.walk(train_dir):
        subdirs.sort()
        for subdir in subdirs:
            class_ = subdir
            subdir = os.path.join(root, subdir)

            classes[class_] = i
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
        
        y_pred = clf.predict(X_train)
        cm = confusion_matrix(y_train, y_pred)
        plot_confusion_matrix(cm, target_names=classes.keys(), filename='test', normalize=False)
        
    else:
        clf = AdaBoostClassifier(n_estimators=500, random_state=0, learning_rate=0.05).fit(X, y)
        print('Training Accuracy: {}'.format(clf.score(X, y)))
        
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        plot_confusion_matrix(cm, target_names=classes.keys(), filename='test', normalize=False)

    return clf
