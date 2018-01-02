from classes.layers.layer import Layer
from classes import functions as funcs

import numpy as np


class FcLayer(Layer):
    def __init__(self, h_size, i_size, o_size,
                 is_input, is_output, is_hidden,
                 u_type='adam', dropout=1.0, a_type="relu", need_activation=True, **kwargs):

        self._w, self._d, self._m = {}, {}, {}

        assert h_size == o_size

        self.h_size = h_size
        self.i_size = i_size
        self.o_size = o_size
        self.is_ouput = is_output
        self.is_input = is_input
        self.is_hidden = is_hidden
        self.u_type = u_type
        self.dropout = dropout
        self.a_type = a_type
        self.need_activation = need_activation

        # for cache
        self.x = []

        self._w_keys = ['w']
        self._b_keys = ['b']

        self._w, self._d, self._m = {}, {}, {}

        for k in self._w_keys:
            self._w[k] = np.random.randn(o_size, i_size).astype(np.float32) * np.sqrt(2.0 / i_size)

        for k in self._b_keys:
            self._w[k] = np.zeros((self.o_size, 1), np.float32)

        for k in self._w_keys + self._b_keys:
            self._m[k] = {'m': 0, 'v': 0, 't': 0}
            self._d[k] = np.zeros_like(self._w[k], np.float32)

    def forward(self, x):
        self.x = x
        y_prime = []

        for t in range(len(x)):
            y = self._w['w'].dot(x[t]) + self._w['b']

            if self.need_activation:
                if self.a_type == 'relu':
                    y = funcs.relu(y)
                elif self.a_type == 'sigmoid':
                    y = funcs.sigmoid(y)
                elif self.a_type == 'tanh':
                    y = funcs.sigmoid(y)

            y_prime.append(y)

        y_prime = np.array(y_prime)

        return y_prime, y_prime

    def backward(self, dy):
        output_d = []

        for t in range(len(self.x)):

            if self.need_activation:
                if self.a_type == 'relu':
                    dy = funcs.d_relu(dy)
                elif self.a_type == 'sigmoid':
                    dy = funcs.d_sigmoid(dy)
                elif self.a_type == 'tanh':
                    dy = funcs.d_tanh(dy)

            self._d['w'] += dy[t].dot(self.x[t].T)
            self._d['b'] += dy[t]
            dx = self._w['w'].T.dot(dy[t])

            output_d.append(dx)

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
        y_prime = []

        for t in range(len(x)):
            y = self._w['w'].dot(x[t]) + self._w['b']

            if self.need_activation:
                if self.a_type == 'relu':
                    y = funcs.relu(y)
                elif self.a_type == 'sigmoid':
                    y = funcs.sigmoid(y)
                elif self.a_type == 'tanh':
                    y = funcs.sigmoid(y)

            y_prime.append(y)

        y_prime = np.array(y_prime)

        return y_prime, y_prime
