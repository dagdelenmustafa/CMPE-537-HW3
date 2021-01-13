import argparse


parser = argparse.ArgumentParser("Application Setup")

parser.add_argument("-t", "--train_path", default="./Caltech20/training", type=str, help="Training path")
parser.add_argument("-tt", "--test_path", default="./Caltech20/testing", type=str, help="Test path")
parser.add_argument("-ss", "--subsampling", default=1, type=int, help="Set if you want to subsample train data")
parser.add_argument("-o", "--cluster_option", default=0, type=int, help="0 -> Pre-trained Kmeans model (Trained with "
                                                                        "Scikit and subsampled data)\n "
                                                                        "1 -> Pre-trained Kmeans model (Trained with "
                                                                        "Scikit and normal data)\n "
                                                                        "2 -> Scikit implemented Kmeans model\n"
                                                                        "3 -> Personally implemented Kmeans model\n")
parser.add_argument("-s", "--cluster_size", default=100, type=int, help="Cluster size")
parser.add_argument("-n", "--n_epoch", default=25, type=int, help="Number of epoch for MLP")
parser.add_argument("-l", "--learning_rate", default=0.4, type=float, help="Learning rate for MLP")
parser.add_argument("-b", "--batchsize", default=32, type=int, help="MLP training batchsize")


def get_args():
    args = parser.parse_args()

    return args
