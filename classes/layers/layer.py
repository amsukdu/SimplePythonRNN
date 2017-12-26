from abc import ABC, abstractmethod


class Layer(ABC):

    # for update temporary memory
    m = {}
    u_type = 'adam'
    is_output = False
    is_input = False
    is_hidden = False

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dy):
        pass

    @abstractmethod
    def output_size(self):
        pass

    @abstractmethod
    def update(self, lr):
        pass

    @abstractmethod
    def predict(self, x):
        pass
