from classes.layers.vanilla_layer import VanillaLayer
from classes.layers.layer import Layer
from classes import functions as funcs

import typing as ty


class RNN(object):

    key_u_type = ('adam', 'adagrad')
    key_layer_type = ('vanilla',)

    def __init__(self, archi, d_size, lr=1e-3):
        layers: ty.List[Layer] = []

        self.archi = archi
        input_size = d_size

        for layer in archi:
            assert 'type' in layer and 'hidden_size' in layer, 'type and hidden_size has to be defined'
            assert layer['type'] in self.key_layer_type, 'wrong layer type'

            is_output = True if layer is archi[-1] else False
            is_input = True if layer is archi[0] else False
            is_hidden = not is_output and not is_input

            if layer['type'] == 'vanilla':
                hidden_size = layer['hidden_size']
                output_size = d_size if is_output else hidden_size
                l = VanillaLayer(hidden_size, input_size, output_size, is_input, is_hidden, is_output, **layer)
                layers.append(l)
                input_size = l.output_size()[0]

        self.layers = layers
        self.lr = lr

    def epoch(self, x, y):
        assert self.layers, 'layers mast be made'

        input_ = x
        for layer in self.layers:
            input_, y_prime = layer.forward(input_)

        loss, d = funcs.softmax_loss(y_prime, y)

        for layer in reversed(self.layers):
            d = layer.backward(d)

        for layer in reversed(self.layers):
            layer.update(self.lr)

        return loss

    def predict(self, x):
        input_ = x
        y_prime = None
        for layer in self.layers:
            input_, y_prime = layer.predict(input_)

        return input_, y_prime

    def reset_h(self):
        for layer in self.layers:
            if isinstance(layer, VanillaLayer):
                layer.h = layer.first_h_prev

    def reset_h_predict_to_h(self):
        for layer in self.layers:
            if isinstance(layer, VanillaLayer):
                layer.h_predict = layer.h
