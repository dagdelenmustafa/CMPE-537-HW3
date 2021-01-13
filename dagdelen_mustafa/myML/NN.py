import numpy as np


class NN:
    class Sequential:
        def __init__(self, arr):
            self.lr = 0.1
            self.seq = arr

        def __call__(self, X):
            return self.forward(X)

        def set_lr(self, lr=0.1):
            self.lr = lr

        def forward(self, X):
            self.input = X
            self.outs = []
            for lay in self.seq:
                self.input = lay(self.input)
                self.outs.append(self.input)
            self.outs = [X] + self.outs
            return self.input

        def loss(self, y, grad=False):
            if not grad:
                corr_prob = self.input[np.arange(len(self.input)), y]
                self.entropy = -corr_prob + np.log(np.sum(np.exp(self.input), axis=-1))
                return self.entropy
            else:
                corr_prob = np.zeros_like(self.input)
                corr_prob[np.arange(len(self.input)), y] = 1
                softmax = np.exp(self.input) / np.exp(self.input).sum(axis=-1, keepdims=True)
                return (- corr_prob + softmax) / self.input.shape[0]

        def backward(self, y):
            loss_grad = self.loss(y, grad=True)
            for i in reversed(range(len(self.seq))):
                layer = self.seq[i]
                loss_grad = layer.backward(self.outs[i], loss_grad, self.lr)

        def predict(self, X):
            return self.forward(X).argmax(axis=-1)

    class Linear:
        def __init__(self, in_features, out_features):
            self.weights = np.random.normal(scale=np.sqrt(2 / (in_features + out_features)),
                                            size=(in_features, out_features))
            self.biases = np.zeros(out_features)

        def __call__(self, X):
            return self.forward(X)

        def forward(self, X):
            return (X @ self.weights) + self.biases

        def backward(self, input, grad_output, lr):
            grad_input = np.dot(grad_output, self.weights.T)

            self.weights = self.weights - lr * np.dot(input.T, grad_output)
            self.biases = self.biases - lr * grad_output.mean(axis=0) * input.shape[0]
            return grad_input

    class ReLU:
        def __init__(self): pass

        def __call__(self, X):
            return self.forward(X)

        def forward(self, X):
            return np.maximum(0, X)

        def backward(self, input, grad_output, lr):
            return grad_output * (input > 0)

    class Softmax:
        def __init__(self): pass

        def __call__(self, X):
            return np.exp(X - np.max(X)) / np.sum(np.exp(X - np.max(X)), axis=0)
