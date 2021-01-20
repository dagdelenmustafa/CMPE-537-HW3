import os
import pickle
import random
import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from tqdm import trange
from sklearn import metrics

import config
from myML.KMeans import KMeans as myKMeans
from myML.NN import NN


def SURF(image):
    surf = cv2.xfeatures2d.SIFT_create(400)
    kp, des = surf.detectAndCompute(image, None)
    return kp, des


def create_df(path, subsample=0):
    class_arr = [name for name in os.listdir(path) if os.path.isdir(path + "/" + name)]
    class_labels = {v: i for i, v in enumerate(class_arr)}
    descriptors = []
    descriptors_class = {v: [] for v in class_arr}
    df = []
    for d_class in class_labels.keys():
        for img_name in os.listdir(path + "/" + d_class):
            image_path = path + "/" + d_class + "/" + img_name
            img = cv2.imread(image_path, 0)
            kp, des = SURF(img)
            if des is not None:
                descriptors.extend(des)
                descriptors_class[d_class].extend(des)
                df.append((d_class, des))

    if not subsample:
        return descriptors, pd.DataFrame(df, columns=['label', 'description']), class_labels, descriptors_class

    desc_lens = {k: len(v) for k, v in descriptors_class.items()}
    n_of_max_sample = min(desc_lens.values()) * 10

    descriptors_sampled = []
    df_sampled = []
    descriptors_class_sample = {v: [] for v in class_arr}
    for k, v in desc_lens.items():
        if v > n_of_max_sample:
            current_desc = 0
            shuffle_df = df[:]
            random.shuffle(shuffle_df)
            for i in shuffle_df:
                if i[0] == k:
                    if current_desc + len(i[1]) < n_of_max_sample:
                        descriptors_sampled.extend(i[1])
                        df_sampled.append(i)
                        descriptors_class_sample[k].extend(i[1])
                        current_desc = current_desc + len(i[1])
        else:
            for i in df:
                if i[0] == k:
                    df_sampled.append(i)
                    descriptors_class_sample[k].extend(i[1])
                    descriptors_sampled.extend(i[1])

    return np.array(descriptors_sampled, dtype=np.double), pd.DataFrame(df_sampled, columns=['label',
                                                                                             'description']), class_labels, descriptors_class_sample


def get_vlad_features(df, n_clusters, centers, class_labels):
    vlad_ = []
    for i in range(len(df)):
        vlad_vector = np.zeros((n_clusters, 128), dtype=np.double)
        words = cdist(df.iloc[i][1], centers).argmin(axis=1)
        for j, w in enumerate(words):
            vlad_vector[w] = np.add(vlad_vector[w], np.subtract(df.iloc[i][1][j], centers[w]))
        vlad_vector = (vlad_vector / np.sqrt(np.sum(np.abs(vlad_vector) ** 2))).flatten()
        vlad_.append(vlad_vector)
    X_data = np.array(vlad_)
    y_data = np.array([class_labels[i] for i in df['label'].values])

    return X_data, y_data


def iterate_mini_batches(inputs, targets, batchsize, shuffle=False):
    '''
    To add mini batch operation, I borrow the code from https://bit.ly/3q5CwtS
    '''
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def create_clusters(cluster_option, n_clusters):
    if cluster_option == 0:
        c_model = pickle.load(open("models/kmeans_subsampled.pkl", "rb"))
    elif cluster_option == 1:
        c_model = pickle.load(open("models/kmeans.pkl", "rb"))
    elif cluster_option == 2:
        c_model = KMeans(n_clusters=n_clusters)
        c_model.fit(descriptors_train)
        pickle.dump(c_model, open("models/kmeans.pkl", "wb"))
    elif cluster_option == 3:
        c_model = myKMeans(n_clusters=n_clusters)
        c_model.fit(descriptors_train)
    else:
        raise ValueError("Unimplemented clustering method '%s'" % args.cluster_option)

    return c_model


if __name__ == '__main__':
    args = config.get_args()
    print(args)
    training_path = args.train_path
    test_path = args.test_path
    n_clusters = args.cluster_size

    print("Creating training descriptors...")
    descriptors_train, df_train, class_labels, descriptors_class = create_df(training_path, args.subsampling)
    print("Creating test descriptors...")
    descriptors_test, df_test, _, _ = create_df(test_path)

    print("Starting Clustering...")
    c_model = create_clusters(args.cluster_option, n_clusters)
    centers = c_model.cluster_centers_

    print("Getting Vlad Features...")
    X_train, y_train = get_vlad_features(df_train, n_clusters, centers, class_labels)
    X_test, y_test = get_vlad_features(df_test, n_clusters, centers, class_labels)

    print("Model initialization...")
    model = NN.Sequential([
        NN.Linear(X_train.shape[1], 500),
        NN.ReLU(),
        NN.Linear(500, len(class_labels))
    ])
    model.set_lr(args.learning_rate)
    best_model = model
    curr_acc = 0

    tr_acc = []
    tst_acc = []
    print("Starting Training...")
    for epoch in range(args.n_epoch):
        for x_batch, y_batch in iterate_mini_batches(X_train, y_train, batchsize=args.batchsize, shuffle=True):
            model(x_batch)
            model.backward(y_batch)

        tr_acc.append(np.mean(model.predict(X_train) == y_train))
        tst_acc.append(np.mean(model.predict(X_test) == y_test))

        if tst_acc[-1] > curr_acc:
            best_model = copy.deepcopy(model)
            curr_acc = tst_acc[-1]

        print("Epoch: {}, Train accuracy: {}, Test accuracy: {}".format(epoch, tr_acc[-1], tst_acc[-1]))

    model = copy.deepcopy(best_model)
    print("Best model acc: {}".format(np.mean(model.predict(X_test) == y_test)))
    pickle.dump(model, open("models/mlp_model.pkl", "wb"))

    plt.plot(tr_acc, label='train accuracy')
    plt.plot(tst_acc, label='test accuracy')
    plt.legend()
    plt.grid()
    plt.savefig("plots/final.png")

    y_pred = model.predict(X_test)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(conf_matrix, index=list(class_labels.keys()), columns=list(class_labels.keys()))
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap="Blues_r")
    plt.savefig("plots/conf_matrix.png")

    print(metrics.classification_report(y_test, y_pred, digits=3, target_names=class_labels.keys()))
