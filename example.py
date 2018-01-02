from classes.rnn import RNN
import numpy as np

data = open('anna.txt', 'r').read().lower().replace('\n', ' ')
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

seq_length = 128
learning_rate = 1e-3
rnn = RNN(
    [
        {'type': 'lstm', 'hidden_size': 256},
        {'type': 'lstm', 'hidden_size': 256},
        {'type': 'lstm', 'hidden_size': 256},
        # {'type': 'lstm', 'hidden_size': 256, 'dropout': 0.5, 'u_type': 'adagrad'},
    ],
    vocab_size, learning_rate, bi=False)

print('bi: {}'.format(rnn.bi))
print(rnn.archi)
print('with seq_length {}'.format(seq_length))

n, p = 0, 0


def sample(seed_ix, n):
    x = np.zeros((1, vocab_size, 1))
    x[0][seed_ix][0] = 1
    ixes = []
    rnn.reset_h_predict_to_h()
    for t in range(n):
        h, y = rnn.predict(x)
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((1, vocab_size, 1))
        x[0][ix][0] = 1
        ixes.append(ix)
    return ixes


smooth_loss = -np.log(1.0 / vocab_size)  # loss at iteration 0
while True:
    if p + seq_length + 1 >= len(data) or n == 0:
        p = 0
        rnn.reset_h()

    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    x, y = np.zeros((seq_length, vocab_size, 1), np.float32), np.zeros((seq_length, vocab_size, 1), np.float32)
    x[range(len(x)), inputs] = 1
    y[:, targets] = 1

    if n % 1000 == 0:
        sample_ix = sample(inputs[0], 500)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt,))

    loss = rnn.epoch(x, targets)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))

    p += seq_length
    n += 1
