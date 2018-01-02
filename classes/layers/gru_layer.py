from classes.layers.layer import Layer
from classes import functions as funcs

import numpy as np


class GruLayer(Layer):
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

        self.clip = (-2, 2)

        self.first_h = [np.zeros((self.h_size, 1), np.float32)]
        self.h = self.first_h.copy()
        self.h_predict_prev = self.first_h[0].copy()

        # for backward
        self._h_prev = self.h[-1]

        # for cache
        self.stacked_x, self.stacked_x_r, self.z, self.r, self.h_hat = [], [], [], [], []

        self._w_keys = ['w_r', 'w_z', 'w_h', 'w_y']
        self._b_keys = ['b_r', 'b_z', 'b_h', 'b_y']

        if not self.is_ouput:
            self._w_keys.remove('w_y')
            self._b_keys.remove('b_y')

        self._w, self._d, self._m = {}, {}, {}

        total_input = h_size + i_size
        for k in self._w_keys:
            if k == 'w_y':
                # dev = np.sqrt(6.0 / (o_size + h_size))
                # self._w[k] = np.random.uniform(-dev, dev, (o_size, h_size)).astype(np.float32)
                self._w[k] = np.random.randn(o_size, h_size).astype(np.float32) * np.sqrt(2.0 / (h_size))
                # self._w[k] = np.random.randn(o_size, h_size).astype(np.float32) * 0.01
                # self._w[k] = np.random.randn(o_size, h_size).astype(np.float32) * np.sqrt(2.0 / (h_size + o_size))
            else:
                # dev = np.sqrt(6.0 / (total_input + h_size))
                # self._w[k] = np.random.uniform(-dev, dev, (h_size, total_input)).astype(np.float32)
                self._w[k] = np.random.randn(h_size, total_input).astype(np.float32) * np.sqrt(2.0 / (total_input))
                # self._w[k] = np.random.randn(h_size, total_input).astype(np.float32) * 0.01
                # self._w[k] = np.random.randn(h_size, total_input).astype(np.float32) * np.sqrt(2.0 / (total_input + h_size))

        for k in self._b_keys:
            if k == 'b_y':
                self._w[k] = np.zeros((self.o_size, 1), np.float32)
            else:
                self._w[k] = np.zeros((self.h_size, 1), np.float32)

        for k in self._w_keys + self._b_keys:
            self._m[k] = {'m': 0, 'v': 0, 't': 0}
            self._d[k] = np.zeros_like(self._w[k], np.float32)

    def forward(self, x):
        self.stacked_x, self.stacked_x_r, self.z, self.r, self.h_hat = [], [], [], [], []

        h_prev = self.h[-1]
        h_t, y_prime = [], []

        # for backward.
        self._h_prev = h_prev

        for t in range(len(x)):
            s_x = np.row_stack((h_prev, x[t]))
            self.stacked_x.append(s_x)

            r = funcs.sigmoid(self._w['w_r'].dot(s_x) + self._w['b_r'])
            z = funcs.sigmoid(self._w['w_z'].dot(s_x) + self._w['b_z'])
            s_x_r = np.row_stack((h_prev * r, x[t]))
            h_hat = funcs.tanh(self._w['w_h'].dot(s_x_r) + self._w['b_h'])
            h = z * h_prev + (1 - z) * h_hat

            h_prev = h

            self.stacked_x_r.append(s_x_r)
            self.r.append(r)
            self.z.append(z)
            self.h_hat.append(h_hat)
            h_t.append(h)

            if self.is_ouput:
                y_prime.append(self._w['w_y'].dot(h) + self._w['b_y'])

        self.h = np.array(h_t)
        y_prime = np.array(y_prime)

        return self.h, y_prime

    def backward(self, dy):
        dh_next = np.zeros_like(self.h[0])
        output_d = []

        for t in reversed(range(len(self.stacked_x))):
            if self.is_ouput:
                # dL/dy * dy/dWy
                self._d['w_y'] += np.dot(dy[t], self.h[t].T)
                # dL/dy * dy/dby
                self._d['b_y'] += dy[t]
                # dL/dy * dy/dh
                dh = self._w['w_y'].T.dot(dy[t]) + dh_next
            else:
                dh = dy[t] + dh_next

            # dL/dh * dh/dh_prev
            dh_prev = self.z[t] * dh
            # dL/dh * dh/dh_hat * dh_hat/dtanh
            dh_hat = (1.0 - self.z[t]) * dh * funcs.d_tanh(self.h_hat[t])

            # dL/dh * dh/dz
            dz = ((self.h[t - 1] if t - 1 >= 0 else self._h_prev) - self.h_hat[t]) * dh * funcs.d_sigmoid(self.z[t])

            # dL/dh_hat * dh_hat/dstacked_x_r
            self._d['w_h'] += dh_hat.dot(self.stacked_x_r[t].T)
            self._d['b_h'] += dh_hat
            dstacked_x_r = self._w['w_h'].T.dot(dh_hat)
            dx = dstacked_x_r

            # dL/dh_hat * dh_hat/dstacked_x_r & p to h
            dstacked_x_r_h = dstacked_x_r[:self.h_size, :]
            # dL/dstacked_x * dh_hat.shape dh_hat/dstacked_x
            dr = dstacked_x_r_h * (self.h[t - 1] if t - 1 >= 0 else self._h_prev) * funcs.d_sigmoid(self.r[t])
            dh_prev += dstacked_x_r_h * self.r[t]

            self._d['w_z'] += dz.dot(self.stacked_x[t].T)
            self._d['b_z'] += dz
            dstacked_x = self._w['w_z'].T.dot(dz)
            dh_prev += dstacked_x[:self.h_size, :]
            dx += dstacked_x

            self._d['w_r'] += dr.dot(self.stacked_x[t].T)
            self._d['b_r'] += dr
            dstacked_x = self._w['w_r'].T.dot(dz)
            dh_prev += dstacked_x[:self.h_size, :]
            dx += dstacked_x

            dh_next = dh_prev

            output_d.insert(0, dx[self.h_size:, :])

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
        h_prev = self.h_predict_prev
        h_t, y_prime = [], []

        for t in range(len(x)):
            s_x = np.row_stack((h_prev, x[t]))

            r = funcs.sigmoid(self._w['w_r'].dot(s_x) + self._w['b_r'])
            z = funcs.sigmoid(self._w['w_z'].dot(s_x) + self._w['b_z'])
            s_x_r = np.row_stack((h_prev * r, x[t]))
            h_hat = np.tanh(self._w['w_h'].dot(s_x_r) + self._w['b_h'])
            h = z * h_prev + (1 - z) * h_hat

            h_prev = h
            h_t.append(h)

            if self.is_ouput:
                y_prime.append(self._w['w_y'].dot(h) + self._w['b_y'])

        self.h_predict_prev = h_prev

        return h_t, np.array(y_prime)
