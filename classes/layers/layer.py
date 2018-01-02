from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    dropout = 1.0
    is_input = False
    is_output = False
    is_hidden = False

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dy):
        pass

    @abstractmethod
    def update(self, lr):
        pass

    @abstractmethod
    def predict(self, x):
        pass
