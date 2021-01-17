import os
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix,\
    f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def generate_initial_clusters():

    print("Generating initial cluster labels")

    all_image_paths = []

    for root, dirs, files in os.walk("data/Caltech20/testing", topdown=False):
        for name in files:
            full_path = os.path.join(root, name)
            all_image_paths.append(full_path)

    for root, dirs, files in os.walk("data/Caltech20/training", topdown=False):
        for name in files:
            full_path = os.path.join(root, name)
            all_image_paths.append(full_path)

    orb = cv2.ORB_create()
    # orb.setMaxFeatures(50)
    # orb.setScaleFactor(2)

    all_descriptors = []
    grouped_descriptors = []

    used_paths = []
    max_keypoints = 150

    for image_path in all_image_paths:
        read_image = cv2.imread(image_path, 0)
        image_keypoints = orb.detect(read_image, None)

        if image_keypoints:
            image_keypoints.sort(key=lambda x: x.response, reverse=True)
            image_keypoints = image_keypoints[:max_keypoints]
            if len(image_keypoints) < max_keypoints:
                pass
            else:
                _, image_descriptors = orb.compute(read_image, image_keypoints)
                all_descriptors.extend(image_descriptors)
                grouped_descriptors.append(image_descriptors)
                # grouped_extended_descriptors.append(np.concatenate(image_descriptors))
                used_paths.append(image_path)
        else:
            print("No keypoints found for: " + image_path)

    number_of_testing_images = 0
    for used_path in used_paths:
        if "testing" in used_path:
            number_of_testing_images += 1
    # print([len(des) for des in descriptors])
    # print(sum([len(des) != 2560 for des in descriptors]))
    number_of_training_images = len(used_paths) - number_of_testing_images
    # grouped_extended_descriptors.append(np.concatenate(image_descriptors))

    all_descriptors = np.asarray(all_descriptors)
    all_descriptors = np.float32(all_descriptors)

    # k-means clustering of descriptors

    max_iter = 20
    epsilon = 0.3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

    cluster_count = 100
    best_labels = None
    attempt_count = 10
    compactness, cluster_labels, centers = cv2.kmeans(all_descriptors, cluster_count, best_labels, criteria,
                                                      attempt_count,
                                                      cv2.KMEANS_RANDOM_CENTERS)

    print("Saving")
    # listified_all_descriptors = np.ndarray.tolist(all_descriptors)
    listified_grouped_descriptors = [np.ndarray.tolist(des) for des in grouped_descriptors]
    listified_cluster_labels = np.ndarray.tolist(cluster_labels)

    testing_clusters = (used_paths[:number_of_testing_images],
                        listified_grouped_descriptors[:number_of_testing_images],
                        listified_cluster_labels[:(number_of_testing_images*max_keypoints)])

    training_clusters = (used_paths[number_of_testing_images:],
                         listified_grouped_descriptors[number_of_testing_images:],
                         listified_cluster_labels[(number_of_testing_images*max_keypoints):])

    with open("initial_testing_cluster.json", 'w') as f_write:
        json.dump(testing_clusters, f_write)

    with open("initial_training_cluster.json", 'w') as f_write:
        json.dump(training_clusters, f_write)

    """
       plt.title("ORB test")
       plt.imshow(img_1)
       plt.show()
       plt.imshow(img_2)
       plt.show()
       """

    return


def generate_feature_vectors(mode):
    print("Generating quantized feature vectors")

    all_labels = os.listdir("data/Caltech20/training")

    image_labels = []
    image_quantized_descriptors = []
    if mode == 'testing':
        with open("initial_testing_cluster.json", 'r') as f_read:
            misc_data = json.load(f_read)
    else:
        with open("initial_training_cluster.json", 'r') as f_read:
            misc_data = json.load(f_read)

    used_paths, grouped_descriptors, cluster_labels = misc_data
    cluster_labels = [label[0] for label in cluster_labels]
    cluster_label_index = 0

    for image_index, image_descriptor_group in enumerate(grouped_descriptors):
        # get image label
        image_path = used_paths[image_index]
        image_label = image_path.split("\\")[1]

        current_label_num = all_labels.index(image_label)

        image_labels.append(current_label_num)

        # calculate quantized image vector
        temp_cluster_labels = []
        for descriptor in image_descriptor_group:
            descriptor_cluster_label = cluster_labels[cluster_label_index]
            cluster_label_index += 1
            temp_cluster_labels.append(descriptor_cluster_label)

        quantized_vector = np.zeros(max(cluster_labels) + 1, int)

        for general_label_index, general_label in enumerate(quantized_vector):
            quantized_vector[general_label_index] = temp_cluster_labels.count(general_label_index)
        image_quantized_descriptors.append(np.ndarray.tolist(quantized_vector))

    for group_index, descriptor_group in enumerate(image_quantized_descriptors):
        group_total = sum(descriptor_group)
        image_quantized_descriptors[group_index] = [num/group_total for num in descriptor_group]

    classification_data = (image_quantized_descriptors, image_labels)

    if mode == 'testing':
        with open("classifier_testing_input.json", 'w') as f_write:
            json.dump(classification_data, f_write)
    else:
        with open("classifier_training_input.json", 'w') as f_write:
            json.dump(classification_data, f_write)

    return


def classify_images():
    print("Classifying")

    # get training data
    with open("classifier_training_input.json", 'r') as f_read:
        image_quantized_descriptors, image_labels = json.load(f_read)

    image_quantized_descriptors = np.asarray(image_quantized_descriptors)
    image_labels = np.asarray(image_labels)

    x_train = image_quantized_descriptors
    y_train = image_labels
    x_train = np.float32(x_train)
    y_train = np.float32(y_train)

    # get test data

    with open("classifier_testing_input.json", 'r') as f_read:
        image_quantized_descriptors_test, image_labels_test = json.load(f_read)

    image_quantized_descriptors_test = np.asarray(image_quantized_descriptors_test)
    image_labels_test = np.asarray(image_labels_test)

    x_test = image_quantized_descriptors_test
    y_test = image_labels_test

    x_test = np.float32(x_test)
    y_test = np.float32(y_test)

    # create classifier

    pipelineRFC = make_pipeline(StandardScaler(), RandomForestClassifier(criterion='gini', random_state=1))

    # Create the parameter grid

    param_grid_rfc = [{
        'randomforestclassifier__max_depth': [15, 20, 25],
        'randomforestclassifier__max_features': [20, 25, 30],
    }]

    # create an instance of the GridSearch Cross-validation estimator

    gsRFC = GridSearchCV(estimator=pipelineRFC,
                         param_grid=param_grid_rfc,
                         scoring='accuracy',
                         cv=5,
                         refit=True,
                         n_jobs=1)

    # train the RandomForestClassifier
    gsRFC = gsRFC.fit(x_train, y_train)

    # the training score of the best model
    # print(gsRFC.best_score_)

    # model parameters of the best model
    print(gsRFC.best_params_)
    #
    # Print the test score of the best model
    #
    clfRFC = gsRFC.best_estimator_

    print('Train accuracy: %.3f' % clfRFC.score(x_train, y_train))
    print('Test accuracy: %.3f' % clfRFC.score(x_test, y_test))

    y_pred = clfRFC.predict(x_test)
    y_true = y_test

    macro_f1_score = f1_score(y_true, y_pred, average='macro')
    per_class_f1_score = f1_score(y_true, y_pred, average=None)
    per_class_precision_score = precision_score(y_true, y_pred, average=None)
    per_class_recall_score = recall_score(y_true, y_pred, average=None)
    total_confusion_matrix = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(clfRFC, x_train, y_train)
    plt.show()

    plot_confusion_matrix(clfRFC, x_test, y_test)
    plt.show()
    """
    for class_index in range(21):
        print(str(class_index + 1) + " & "
              + str(per_class_f1_score[class_index]) + " & "
              + str(per_class_precision_score[class_index]) + " & "
              + str(per_class_recall_score[class_index]) + " \\\\")

    for class_index_row in range(20):
        print(str(class_index_row + 1) + " & ", end='')
        for class_index_col in range(20):
            print(str(total_confusion_matrix[class_index_row][class_index_col]), end='')
            if class_index_col != 19:
                print(" & ", end=''),
        print(" \\\\")
        print("\\hline")
    """

    return


def main():
    generate_initial_clusters()
    generate_feature_vectors('training')
    generate_feature_vectors('testing')
    classify_images()


main()
