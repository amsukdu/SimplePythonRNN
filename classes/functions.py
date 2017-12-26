import numpy as np


def softmax_loss(y_prime, y):
    probs = np.exp(y_prime - np.max(y_prime, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = y_prime.shape[0]
    loss = -np.sum(np.log(probs[range(N), y]))
    dx = probs
    dx[range(N), y] -= 1
    return loss, dx


def adam_update(w, d, m, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    m['m'] = beta1 * m['m'] + (1 - beta1) * d
    m['v'] = beta2 * m['v'] + (1 - beta2) * (d ** 2)

    m_ = m['m'] / (1 - beta1 ** t)
    v_ = m['v'] / (1 - beta2 ** t)

    return w - lr * m_ / (np.sqrt(v_) + eps)


def adagrad_update(w, d, m, lr):
    m['m'] += d * d
    return w - lr * d / np.sqrt(m['m'] + 1e-8)
