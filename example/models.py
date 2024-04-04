from typing import Union
import numpy as np
import cupy as cp
from utils import ReLU, softmax


# Settings
SEED = 2999
INPUT_DIM = 28*28
HIDDEN_DIM = 28*28
OUTPUT_DIM = 10

DataArray = Union[np.ndarray, cp.ndarray]


class CustomModel:
    def __init__(self, lr: float = 2e-3, backend = np):
        self.backend = backend
        self.w1 = backend.random.uniform(
            low=-1.0,
            high=1.0,
            size=(INPUT_DIM, HIDDEN_DIM),
        )
        self.w2 = backend.random.uniform(
            low=-1.0,
            high=1.0,
            size=(HIDDEN_DIM, OUTPUT_DIM),
        )
        self.b1 = backend.zeros((1, HIDDEN_DIM))
        self.b2 = backend.zeros((1, OUTPUT_DIM))
        self.lr = lr

    def forward(self, x: DataArray) -> DataArray:
        self.x = x
        self.out_layer_1 = x.dot(self.w1) + self.b1
        self.out_activate_1 = ReLU(self.out_layer_1)
        self.out_layer_2 = self.out_activate_1.dot(self.w2) + self.b2
        self.out_activate_2 = softmax(self.out_layer_2)
        
        return self.out_activate_2

    def backward(self, y_true: DataArray) -> None:
        # Compute cross-entropy gradient
        init_gradient = self.out_activate_2 - y_true

        # Compute the second layer gradient
        dL_dw2 = self.out_activate_1.T.dot(init_gradient)
        dL_db2 = self.backend.sum(init_gradient, axis=0)

        # Compute the first layer gradient
        gradient_2_to_1 = init_gradient.dot(self.w2.T) * (self.out_layer_1 > 0)
        dL_dw1 = self.x.T.dot(gradient_2_to_1)
        dL_db1 = self.backend.sum(gradient_2_to_1, axis=0)

        # Update weights and biases
        self.w1 -= self.lr * dL_dw1
        self.b1 -= self.lr * dL_db1
        self.w2 -= self.lr * dL_dw2
        self.b2 -= self.lr * dL_db2        

    def save_checkpoint(self, path: str = "./checkpoint.npz") -> None:
        self.backend.savez(
            path,
            w1=self.w1,
            w2=self.w2,
            b1=self.b1,
            b2=self.b2,
        )

    def load_checkpoint(self, path: str = "./checkpoint.npz") -> None:
        with self.backend.load(path) as data:
            self.w1 = self.backend.asarray(data["w1"])
            self.w2 = self.backend.asarray(data["w2"])
            self.b1 = self.backend.asarray(data["b1"])
            self.b2 = self.backend.asarray(data["b2"])
