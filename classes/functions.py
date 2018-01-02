import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    return x > 0


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - (x ** 2)


def softmax_loss(y_prime, y):
    probs = np.exp(y_prime - np.max(y_prime, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = y_prime.shape[0]
    loss = -np.sum(np.log(probs[range(N), y])) / N
    dx = probs.copy()
    dx[range(N), y] -= 1
    return loss, dx


def logistic_loss(y_prime, y):
    N = y_prime.shape[0]
    loss = np.sum(np.square(y - y_prime) / 2) / N
    dx = -(y - y_prime)
    return loss, dx


def adam_update(w, d, m, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    m['t'] += 1

    m['m'] = beta1 * m['m'] + (1 - beta1) * d
    m['v'] = beta2 * m['v'] + (1 - beta2) * (d ** 2)

    m_ = m['m'] / (1 - beta1 ** m['t'])
    v_ = m['v'] / (1 - beta2 ** m['t'])

    return w - lr * m_ / (np.sqrt(v_) + eps)


def adagrad_update(w, d, m, lr, eps=1e-8):
    m['m'] += d * d
    return w - lr * d / np.sqrt(m['m'] + eps)
