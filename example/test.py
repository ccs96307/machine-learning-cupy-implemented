import cupy as cp
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

from dataloader import get_mnist_dataloader
from models import CustomModel


def main() -> None:
    backend = cp
    model = CustomModel(backend=backend)
    model.load_checkpoint("./checkpoint.npz")

    # Get dataloader
    train_x, test_x, train_y, test_y = get_mnist_dataloader()

    batch_size = 4

    all_preds = []
    all_labels = []

    steps = len(test_x) // batch_size

    for times in tqdm(range(steps)):
        inputs = test_x[times*batch_size:(times+1)*batch_size]
        labels = test_y[times*batch_size:(times+1)*batch_size]

        inputs = backend.asarray(inputs)
        labels = backend.asarray(labels)

        outputs = model.forward(inputs)

        # Get predictions
        preds = np.argmax(outputs, axis=1).tolist()
        labels = np.argmax(labels, axis=1).tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    main()
