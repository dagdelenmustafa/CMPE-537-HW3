from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report

import os
import cv2
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    # https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/19252430
    # Calvin Duy Canh Tran & georg-un
    
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
    
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix2(cm, target_names=clf.classes_, normalize=False)
