import cupy as cp
import numpy as np
from tqdm import tqdm

from dataloader import get_mnist_dataloader
from loss_function import cross_entropy_loss
from models import CustomModel
from utils import AverageMeter


def main() -> None:
    backend = cp
    model = CustomModel(lr=0.02, backend=backend)

    # Get dataloader
    train_x, test_x, train_y, test_y = get_mnist_dataloader(backend=backend)

    # Settings
    batch_size = 16
    epochs = 30

    # Logger
    loss_logger = AverageMeter()

    # Train
    for epoch in range(1, epochs + 1):
        steps = len(train_x) // batch_size
        train_pbar = tqdm(total=steps, desc=f"[Epoch {epoch}/{epochs}]")

        for times in range(steps):
            inputs = train_x[times*batch_size:(times+1)*batch_size]
            labels = train_y[times*batch_size:(times+1)*batch_size]

            # Forward
            outputs = model.forward(inputs)
            
            # Compute loss
            loss = cross_entropy_loss(labels, outputs)
            loss_logger.update(loss)
            
            # Backward
            model.backward(labels)

            train_pbar.set_description(f"[Epoch {epoch}/{epochs}], Loss: {loss_logger.avg:.4f}")
            train_pbar.update(1)

        train_pbar.close()

        # Save checkpoint
        model.save_checkpoint()


if __name__ == "__main__":
    main()
