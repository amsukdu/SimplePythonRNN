from classes.rnn import RNN
import numpy as np

data = open('anna.txt', 'r').read().lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

seq_length = 25
learning_rate = 1e-3
rnn = RNN(
    [
        {'type': 'vanilla', 'hidden_size': 64},
        {'type': 'vanilla', 'hidden_size': 64},
        # {'type': 'vanilla', 'hidden_size': 150}
     ],
    vocab_size, learning_rate)

n, p = 0, 0

def sample(seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
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


smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:
        # hprev = np.zeros((hidden_size, 1), np.float32)  # reset RNN memory
        p = 0  # go from start of data
        rnn.reset_h()
        print('reset!')

    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    x, y = np.zeros((seq_length, vocab_size, 1), np.float32), np.zeros((seq_length, vocab_size, 1), np.float32)
    x[range(len(x)), inputs] = 1
    y[:, targets] = 1

    # sample from the model now and then
    if n % 10000 == 0:
        sample_ix = sample(inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt,))

    # forward seq_length characters through the net and fetch gradient
    loss = rnn.epoch(x, targets)
    # print(loss)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss / seq_length))  # print progress

    p += seq_length  # move data pointer
    n += 1  # iteration counter
