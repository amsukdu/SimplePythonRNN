from classes.layers.layer import Layer
from classes import functions as funcs

import numpy as np


class LstmLayer(Layer):
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

        self.first_c = [np.zeros((self.h_size, 1), np.float32)]
        self.c = self.first_c.copy()
        self.c_predict_prev = self.first_c[0].copy()

        # for backward
        self._c_prev = self.c[0]

        # for cache
        self.stacked_x, self.tanh_c, self.hf, self.hi, self.ho, self.hc = [], [], [], [], [], []

        self._w_keys = ['w_f', 'w_i', 'w_c', 'w_o', 'w_y']
        self._b_keys = ['b_f', 'b_i', 'b_c', 'b_o', 'b_y']

        if not self.is_ouput:
            self._w_keys.remove('w_y')
            self._b_keys.remove('b_y')

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
        self.stacked_x, self.tanh_c, self.hf, self.hi, self.ho, self.hc = [], [], [], [], [], []

        h_t, c_t, y_prime = [], [], []
        h_prev = self.h[-1]
        c_prev = self.c[-1]

        # for backward. h is on stacked_x
        self._c_prev = c_prev

        for t in range(len(x)):
            s_x = np.row_stack((h_prev, x[t]))
            self.stacked_x.append(s_x)

            output_f = funcs.sigmoid(self._w['w_f'].dot(s_x) + self._w['b_f'])
            output_i = funcs.sigmoid(self._w['w_i'].dot(s_x) + self._w['b_i'])
            output_o = funcs.sigmoid(self._w['w_o'].dot(s_x) + self._w['b_o'])
            output_c = np.tanh(self._w['w_c'].dot(s_x) + self._w['b_c'])

            c = (output_f * c_prev) + (output_i * output_c)
            c_t.append(c)
            tanh_c = np.tanh(c)

            h = output_o * tanh_c
            h_t.append(h)

            h_prev = h
            c_prev = c

            self.hf.append(output_f)
            self.hi.append(output_i)
            self.ho.append(output_o)
            self.hc.append(output_c)

            self.tanh_c.append(tanh_c)

            if self.is_ouput:
                y_prime.append(self._w['w_y'].dot(h) + self._w['b_y'])

        self.h = h_t
        self.c = c_t

        return np.array(self.h), np.array(y_prime)

    def backward(self, dy):
        dh_next = np.zeros_like(self.h[0])
        dc_next = np.zeros_like(self.c[0])

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

            # dL/dh * dh/dho * dho/dsigmoid
            dho = self.tanh_c[t] * dh * funcs.d_sigmoid(self.ho[t])

            # dL/dh * dh/dc
            dc = self.ho[t] * dh * funcs.d_tanh(self.tanh_c[t]) + dc_next

            # dL/dc * dc/dhf * dhf/dsigmoid
            dhf = (self.c[t - 1] if t - 1 >= 0 else self._c_prev) * dc * funcs.d_sigmoid(self.hf[t])

            # dL/dc * dc/dhi * dhi/dsigmoid
            dhi = self.hc[t] * dc * funcs.d_sigmoid(self.hi[t])

            # dL/dc * dc/dhc * dhc/dtanh
            dhc = self.hi[t] * dc * funcs.d_tanh(self.hc[t])

            # dL/dsigmoid * dsigmoid/dwf
            self._d['w_f'] += dhf.dot(self.stacked_x[t].T)
            self._d['b_f'] += dhf

            # dL/dsigmoid * dsigmoid/dwi
            self._d['w_i'] += dhi.dot(self.stacked_x[t].T)
            self._d['b_i'] += dhi

            # dL/dsigmoid * dsigmoid/dwo
            self._d['w_o'] += dho.dot(self.stacked_x[t].T)
            self._d['b_o'] += dho

            # dL/dtanh * dtanh/dwc
            self._d['w_c'] += dhc.dot(self.stacked_x[t].T)
            self._d['b_c'] += dhc

            dx = self._w['w_f'].T.dot(dhf) + self._w['w_i'].T.dot(dhi) + self._w['w_c'].T.dot(dhc) + self._w['w_o'].T.dot(
                dho)

            dh_next = dx[:self.h_size, :]
            dc_next = self.hf[t] * dc

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
        h_t, y_prime = [], []
        h_prev = self.h_predict_prev
        c_prev = self.c_predict_prev

        for t in range(len(x)):
            s_x = np.row_stack((h_prev, x[t]))

            output_f = funcs.sigmoid(self._w['w_f'].dot(s_x) + self._w['b_f'])
            output_i = funcs.sigmoid(self._w['w_i'].dot(s_x) + self._w['b_i'])
            output_o = funcs.sigmoid(self._w['w_o'].dot(s_x) + self._w['b_o'])
            output_c = funcs.tanh(self._w['w_c'].dot(s_x) + self._w['b_c'])

            c = output_f * c_prev + output_i * output_c
            h = output_o * funcs.tanh(c)

            if self.is_ouput:
                y_prime.append(self._w['w_y'].dot(h) + self._w['b_y'])

            h_t.append(h)

            h_prev = h
            c_prev = c

        self.h_predict_prev = h_prev
        self.c_predict_prev = c_prev

        return h_t, np.array(y_prime)
