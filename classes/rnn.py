from classes.layers.vanilla_layer import VanillaLayer
from classes.layers.lstm_layer import LstmLayer
from classes.layers.gru_layer import GruLayer
from classes.layers.fc_layer import FcLayer
from classes.layers.layer import Layer
from classes import functions as funcs

import numpy as np


class RNN(object):
    key_u_type = ('adam', 'adagrad')
    key_layer_type = ('vanilla', 'lstm', 'gru', 'fc')

    def __init__(self, archi, d_size, lr=1e-3, bi=False):
        layers = []

        self.archi = archi
        self.dropout_masks = []
        self.bi = bi
        input_size = d_size

        for layer in archi:
            assert 'type' in layer and 'hidden_size' in layer, 'type and hidden_size has to be defined'
            assert layer['type'] in self.key_layer_type, 'wrong layer type'

            is_output = True if layer is archi[-1] else False
            is_input = True if layer is archi[0] else False
            is_hidden = not is_output and not is_input

            layer['is_output'], layer['is_input'], layer['is_hidden'] = is_output, is_input, is_hidden

            if layer['type'] == 'vanilla':
                hidden_size = layer['hidden_size']
                output_size = d_size if is_output else hidden_size
                lay = VanillaLayer(hidden_size, input_size, output_size, **layer)
                if self.bi:
                    lay = lay, VanillaLayer(hidden_size, input_size, output_size, **layer)

            elif layer['type'] == 'lstm':
                hidden_size = layer['hidden_size']
                output_size = d_size if is_output else hidden_size
                lay = LstmLayer(hidden_size, input_size, output_size, **layer)
                if self.bi:
                    lay = lay, LstmLayer(hidden_size, input_size, output_size, **layer)

            elif layer['type'] == 'gru':
                hidden_size = layer['hidden_size']
                output_size = d_size if is_output else hidden_size
                lay = GruLayer(hidden_size, input_size, output_size, **layer)
                if self.bi:
                    lay = lay, GruLayer(hidden_size, input_size, output_size, **layer)

            elif layer['type'] == 'fc':
                if self.bi:
                    input_size *= 2
                hidden_size = layer['hidden_size']
                output_size = hidden_size
                need_activation = not is_output
                lay = FcLayer(hidden_size, input_size, output_size, need_activation=need_activation, **layer)

            layers.append(lay)
            input_size = output_size

        if self.bi and isinstance(layers[-1], tuple):
            output_size *= len(layers[-1])

        self.layers = layers
        self.lr = lr

    def epoch(self, x, y):
        assert self.layers, 'layers must be made'

        if self.bi:
            next_input = x, np.flip(x, 0)
        else:
            next_input = x

        for layer in self.layers:
            # next_input is tuple
            if self.bi:
                if isinstance(layer, tuple):
                    l1: Layer = layer[0]
                    l2: Layer = layer[1]

                    output_1, _ = l1.forward(next_input[0])
                    output_2, _ = l2.forward(next_input[1])

                    next_input, y_prime = (output_1, output_2), None
                    layer = l1
                    if layer.is_output:
                        y_prime = np.concatenate((next_input[0], next_input[1]), 1)
                else:
                    layer: Layer

                    next_input = np.concatenate((next_input[0], next_input[1]), 1)
                    next_input, y_prime = layer.forward(next_input)
            else:
                layer: Layer

                next_input, y_prime = layer.forward(next_input)

            if layer.dropout < 1 and not layer.is_output:
                shape = next_input[0].shape if self.bi else next_input.shape

                dropout_mask = np.random.rand(*shape[1:]) < layer.dropout
                dropout_mask = np.tile(dropout_mask, shape[0]).T.reshape(shape)
                self.dropout_masks.append(dropout_mask)

                if self.bi:
                    next_input = tuple(i * dropout_mask / layer.dropout for i in next_input)
                else:
                    next_input *= dropout_mask / layer.dropout

        loss, next_d = funcs.softmax_loss(y_prime, y)

        for layer in reversed(self.layers):
            l = layer[0] if isinstance(layer, tuple) else layer
            if l.dropout < 1 and not l.is_output and self.dropout_masks:
                dropout_mask = self.dropout_masks.pop()
                if self.bi and isinstance(next_d, tuple):
                    next_d = tuple(i * dropout_mask for i in next_d)
                else:
                    next_d *= dropout_mask

            # next_d is tuple
            if self.bi:
                if isinstance(layer, tuple):
                    l1: Layer = layer[0]
                    l2: Layer = layer[1]

                    next_d = l1.backward(next_d[0]), l2.backward(next_d[1])
                else:
                    layer: Layer
                    if isinstance(next_d, tuple):
                        next_d = np.concatenate((next_d[0], next_d[1]), 1)
                    next_d = layer.backward(next_d)
                    half = layer.i_size // 2
                    next_d = next_d[:, :half, :], next_d[:, half:, :]
            else:
                next_d = layer.backward(next_d)

        for layer in reversed(self.layers):
            if self.bi and isinstance(layer, tuple):
                l1: Layer = layer[0]
                l2: Layer = layer[1]

                l1.update(self.lr)
                l2.update(self.lr)
            else:
                layer.update(self.lr)

        return loss

    def predict(self, x):
        if self.bi:
            next_input = x, np.flip(x, 0)
        else:
            next_input = x

        y_prime = None

        for layer in self.layers:
            if self.bi:
                if isinstance(layer, tuple):
                    l1: Layer = layer[0]
                    l2: Layer = layer[1]
                    output_1, _ = l1.forward(next_input[0])
                    output_2, _ = l2.forward(next_input[1])

                    next_input = output_1, output_2
                else:
                    layer: Layer
                    next_input = np.concatenate((next_input[0], next_input[1]), 1)
                    next_input, y_prime = layer.forward(next_input)
            else:
                layer: Layer
                next_input, y_prime = layer.forward(next_input)

        return next_input, y_prime

    def reset_h(self):
        for layer in self.layers:
            if isinstance(layer, VanillaLayer):
                layer.h = layer.first_h.copy()
            elif isinstance(layer, LstmLayer):
                layer.h = layer.first_h.copy()
                layer.c = layer.first_c.copy()
            elif isinstance(layer, GruLayer):
                layer.h = layer.first_h.copy()

    def reset_h_predict(self):
        for layer in self.layers:
            if isinstance(layer, VanillaLayer):
                layer.h_predict = layer.first_h[0].copy()
            elif isinstance(layer, LstmLayer):
                layer.h_predict = layer.first_h[0].copy()
                layer.c_predict = layer.first_c[0].copy()
            elif isinstance(layer, GruLayer):
                layer.h_predict = layer.first_h[0].copy()

    def reset_h_predict_to_h(self):
        for layer in self.layers:
            if isinstance(layer, VanillaLayer):
                layer.h_predict_prev = layer.h[-1]
            elif isinstance(layer, LstmLayer):
                layer.h_predict_prev = layer.h[-1]
                layer.c_predict_prev = layer.c[-1]
            elif isinstance(layer, GruLayer):
                layer.h_predict_prev = layer.h[-1]
