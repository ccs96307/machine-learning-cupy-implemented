import numpy as np

from custom_types import Backend, DataArray
from utils import ReLU, softmax


class CustomModel:
    def __init__(
        self, 
        input_dim: int = 784,
        hidden_dim: int = 784,
        num_classes: int = 10,
        lr: float = 2e-3,
        backend: Backend = np,
    ):
        self.backend = backend
        self._w1 = backend.random.uniform(
            low=-1.0,
            high=1.0,
            size=(input_dim, hidden_dim),
        )
        self._w2 = backend.random.uniform(
            low=-1.0,
            high=1.0,
            size=(hidden_dim, num_classes),
        )
        self._b1 = backend.zeros((1, hidden_dim))
        self._b2 = backend.zeros((1, num_classes))
        self.lr = lr

    def forward(self, x: DataArray) -> DataArray:
        self.x = x
        self.out_layer_1 = x.dot(self._w1) + self._b1
        self.out_activate_1 = ReLU(self.out_layer_1)
        self.out_layer_2 = self.out_activate_1.dot(self._w2) + self._b2
        self.out_activate_2 = softmax(self.out_layer_2)
        
        return self.out_activate_2

    def backward(self, y_true: DataArray) -> None:
        # Compute cross-entropy gradient
        init_gradient = self.out_activate_2 - y_true

        # Compute the second layer gradient
        dL_dw2 = self.out_activate_1.T.dot(init_gradient)
        dL_db2 = self.backend.sum(init_gradient, axis=0)

        # Compute the first layer gradient
        gradient_2_to_1 = init_gradient.dot(self._w2.T)
        gradient_2_to_1 = gradient_2_to_1 * (self.out_layer_1 > 0) # ReLU
        dL_dw1 = self.x.T.dot(gradient_2_to_1)
        dL_db1 = self.backend.sum(gradient_2_to_1, axis=0)

        # Update weights and biases
        self._w1 -= self.lr * dL_dw1
        self._b1 -= self.lr * dL_db1
        self._w2 -= self.lr * dL_dw2
        self._b2 -= self.lr * dL_db2

    def save_checkpoint(self, path: str = "./checkpoint.npz") -> None:
        self.backend.savez(
            path,
            w1=self._w1,
            w2=self._w2,
            b1=self._b1,
            b2=self._b2,
        )

    def load_checkpoint(self, path: str = "./checkpoint.npz") -> None:
        with self.backend.load(path) as data:
            self._w1 = self.backend.asarray(data["w1"])
            self._w2 = self.backend.asarray(data["w2"])
            self._b1 = self.backend.asarray(data["b1"])
            self._b2 = self.backend.asarray(data["b2"])
