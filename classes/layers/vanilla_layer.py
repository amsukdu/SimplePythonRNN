from .layer import Layer
import classes.functions as funcs
import numpy as np


class VanillaLayer(Layer):
    def __init__(self, h_size, i_size, o_size,
                 is_input, is_output, is_hidden,
                 u_type='adam', dropout=1.0, **kwargs):

        self._w, self._d, self._m = {}, {}, {}

        self.h_size = h_size
        self.i_size = i_size
        self.o_size = o_size
        self.is_ouput = is_output
        self.is_input = is_input
        self.is_hidden = is_hidden
        self.u_type = u_type
        self.dropout = dropout

        self.clip = (-3, 3)

        self.first_h = [np.zeros((self.h_size, 1), np.float32)]
        self.h = self.first_h.copy()
        self.h_predict_prev = self.first_h[0].copy()

        # for backward
        self._h_prev = self.h[-1]

        self.x = []

        self._w_keys = ['w_xh', 'w_hh', 'w_hy']
        self._b_keys = ['b_h', 'b_y']

        if not self.is_ouput:
            self._w_keys.remove('w_hy')
            self._b_keys.remove('b_y')

        for k in self._w_keys + self._b_keys:
            if k == 'w_xh':
                self._w[k] = np.random.randn(self.h_size, self.i_size).astype(np.float32) * np.sqrt(2.0 / (self.i_size))
            elif k == 'w_hh':
                self._w[k] = np.random.randn(self.h_size, self.h_size).astype(np.float32) * np.sqrt(2.0 / (self.h_size))
            elif k == 'b_h':
                self._w[k] = np.zeros((self.h_size, 1), np.float32)
            elif k == 'w_hy':
                self._w[k] = np.random.randn(self.o_size, self.h_size).astype(np.float32) * np.sqrt(2.0 / (self.h_size))
            elif k == 'b_y':
                self._w[k] = np.zeros((self.o_size, 1), np.float32)
            else:
                raise ValueError('{} key has to be initiated'.format(k))

        for k in self._w_keys + self._b_keys:
            self._m[k] = {'m': 0, 'v': 0, 't': 0}
            self._d[k] = np.zeros_like(self._w[k], np.float32)

    def forward(self, x):
        self.x = x
        h, y_prime = [], []

        h_prev = self.h[-1]
        self._h_prev = h_prev

        for t in range(len(x)):
            h.append(funcs.tanh(self._w['w_xh'].dot(x[t]) + self._w['w_hh'].dot(h_prev) + self._w['b_h']))
            if self.is_ouput:
                y_prime.append(np.dot(self._w['w_hy'], h[t]) + self._w['b_y'])
            h_prev = h[t]

        h = np.array(h)
        y_prime = np.array(y_prime)

        self.h = h
        return h, y_prime

    def backward(self, dy):
        dh_next = np.zeros_like(self.h[0], np.float32)
        output_d = []
        for t in reversed(range(len(self.x))):
            if self.is_ouput:
                # dL/dy * dy/dWxy
                self._d['w_hy'] += np.dot(dy[t], self.h[t].T)
                # dL/dy * dy/dby
                self._d['b_y'] += dy[t]

                # dL/dy * dL/dh
                # h was split so add dhnext
                dh = np.dot(self._w['w_hy'].T, dy[t]) + dh_next
            else:
                # dL/dy * dL/dh
                dh = dy[t] + dh_next

            # dL/dh * dh/dtanh
            dtanh = funcs.d_tanh(self.h[t]) * dh

            output_d.insert(0, np.dot(self._w['w_xh'].T, dtanh))

            # dL/dtanh * dtanh/dbh
            self._d['b_h'] += dtanh
            # dL/dtanh * dtanh/dWxh
            self._d['w_xh'] += np.dot(dtanh, self.x[t].T)
            # dL/dtanh * dtanh/dWhh
            self._d['w_hh'] += np.dot(dtanh, self.h[t - 1].T if t - 1 >= 0 else self._h_prev.T)
            # dL/dtanh * dtanh/dht-1
            dh_next = np.dot(self._w['w_hh'].T, dtanh)

        for k in self._w_keys + self._b_keys:
            np.clip(self._d[k], self.clip[0], self.clip[1], out=self._d[k])

        return np.array(output_d)

    def update(self, lr):
        for k in self._w_keys + self._b_keys:
            if self.u_type == 'adam':
                self._w[k] = funcs.adam_update(self._w[k], self._d[k], self._m[k], lr)
            elif self.u_type == 'adagrad':
                self._w[k] = funcs.adagrad_update(self._w[k], self._d[k], self._m[k], lr)

        for k in self._w_keys + self._b_keys:
            self._d[k] = np.zeros_like(self._w[k], np.float32)

    def predict(self, x):
        h, y_prime = [], []

        h_prev = self.h_predict_prev

        for t in range(len(x)):
            h.append(funcs.tanh(self._w['w_xh'].dot(x[t]) + self._w['w_hh'].dot(h_prev) + self._w['b_h']))
            if self.is_ouput:
                y_prime.append(np.dot(self._w['w_hy'], h[t]) + self._w['b_y'])
            h_prev = h[t]

        self.h_predict_prev = h_prev
        h = np.array(h)
        y_prime = np.array(y_prime)

        return h, y_prime
