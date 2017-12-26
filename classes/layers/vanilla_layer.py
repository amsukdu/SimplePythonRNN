from .layer import Layer
import classes.functions as u
import numpy as np
import scipy.stats


class VanillaLayer(Layer):
    def __init__(self, h_size, i_size, o_size,
                 is_input, is_hidden, is_ouput,
                 clip=(-3, 3), u_type='adam', **kwargs):

        self.h_size = h_size
        self.i_size = i_size
        self.o_size = o_size
        self.is_input = is_input
        self.is_hidden = is_hidden
        self.is_output = is_ouput
        self.clip = clip
        self.u_type = u_type

        self.first_h_prev = np.zeros((1, self.h_size, 1), np.float32)
        self.h = self.first_h_prev
        self.h_predict = self.first_h_prev
        self.x = None

        self._w_keys = ['w_xh', 'w_hh', 'w_hy']
        self._b_keys = ['b_h', 'b_y']

        if not self.is_output:
            self._w_keys.remove('w_hy')
            self._b_keys.remove('b_y')

        self.w, self.d, self.m = {}, {}, {}

        lower = -2
        upper = 2
        mu = 0
        sigma = 1

        for k in self._w_keys + self._b_keys:
            if k == 'w_xh':
                self.w[k] = scipy.stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma,
                                                      loc=mu, scale=sigma, size=(self.h_size, self.i_size)).astype(np.float32) * 0.01
            elif k == 'w_hh':
                self.w[k] = scipy.stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma,
                                                      loc=mu, scale=sigma, size=(self.h_size, self.h_size)).astype(np.float32) * 0.01
            elif k == 'b_h':
                self.w[k] = np.zeros((self.h_size, 1), np.float32)
            elif k == 'w_hy':
                self.w[k] = scipy.stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma,
                                                      loc=mu, scale=sigma, size=(self.o_size, self.h_size)).astype(np.float32) * 0.01
            elif k == 'b_y':
                self.w[k] = np.zeros((self.o_size, 1), np.float32)
            else:
                raise ValueError('{} key has to be initiated'.format(k))

            self.m[k] = {'m': 0, 'v': 0}
            self.d[k] = np.zeros_like(self.w[k], np.float32)

        self.t = 0

    def forward(self, x):
        self.x = x

        h, y_prime = [], []
        h_prev = self.h[-1]

        for t in range(len(x)):
            h.append(np.tanh(self.w['w_xh'].dot(x[t]) + self.w['w_hh'].dot(h_prev) + self.w['b_h']))
            if self.is_output:
                y_prime.append(np.dot(self.w['w_hy'], h[t]) + self.w['b_y'])
            h_prev = h[t]

        h = np.array(h)
        y_prime = np.array(y_prime)

        self.h = h
        return h, y_prime

    def backward(self, dy):
        assert self.x is not None

        dhnext = np.zeros_like(self.h[0], np.float32)
        output_d = []
        for t in reversed(range(len(self.x))):
            if self.is_output:
                # dL/dy * dy/dWxy
                self.d['w_hy'] += np.dot(dy[t], self.h[t].T)
                # dL/dy * dy/dby
                self.d['b_y'] += dy[t]

                # dL/dy * dL/dh
                dh = np.dot(self.w['w_hy'].T, dy[t]) + dhnext
            else:
                # dL/dy * dL/dh
                dh = dy[t] + dhnext

            # dL/dh * dh/dtanh
            dtanh = (1 - self.h[t] * self.h[t]) * dh

            output_d.insert(0, np.dot(self.w['w_xh'].T, dtanh))

            # dL/dtanh * dtanh/dbh
            self.d['b_h'] += dtanh
            # dL/dtanh * dtanh/dWxh
            self.d['w_xh'] += np.dot(dtanh, self.x[t].T)
            # dL/dtanh * dtanh/dWhh
            self.d['w_hh'] += np.dot(dtanh, self.h[t - 1].T if t - 1 >= 0 else self.first_h_prev[-1].T)
            # dL/dtanh * dtanh/dht-1
            dhnext = np.dot(self.w['w_hh'].T, dtanh)

        for k in self._w_keys + self._b_keys:
            np.clip(self.d[k], self.clip[0], self.clip[1], out=self.d[k])  # clip to mitigate exploding gradients

        return np.array(output_d)

    def update(self, lr):
        assert self.x is not None

        self.t += 1
        for k in self._w_keys + self._b_keys:
            if self.u_type == 'adam':
                self.w[k] = u.adam_update(self.w[k], self.d[k], self.m[k], self.t, lr)
            elif self.u_type == 'adagrad':
                self.w[k] = u.adagrad_update(self.w[k], self.d[k], self.m[k], lr)

        for k in self._w_keys + self._b_keys:
            self.d[k] = np.zeros_like(self.w[k], np.float32)

    def output_size(self):
        return (self.o_size, 1)

    def predict(self, x):
        h, y_prime = [], []
        h_prev = self.h_predict[-1]

        for t in range(len(x)):
            h.append(np.tanh(self.w['w_xh'].dot(x[t]) + self.w['w_hh'].dot(h_prev) + self.w['b_h']))
            if self.is_output:
                y_prime.append(np.dot(self.w['w_hy'], h[t]) + self.w['b_y'])
            h_prev = h[t]

        h = np.array(h)
        y_prime = np.array(y_prime)

        self.h_predict = h
        return h, y_prime
