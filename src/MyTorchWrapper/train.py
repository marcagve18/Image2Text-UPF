import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from .evaluation import BasicEvaluation
from typing import Dict, List, Optional


class Trainer:
    """Class to train a Neural Network model."""

    def __init__(
        self,
        evaluation: BasicEvaluation,
        epochs: int,
        data_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """
        Args:
            evaluation (BasicEvaluation): evaluation instance with the desired
            methods of evaluation, including the loss. See the BasicEvaluation
            class for more details.
            epochs (int): number of training epochs
            data_loader (DataLoader): Data with which to train the torch model
            device (torch.device): device in which to perform the computations
        """
        self.evaluation = evaluation
        self.epochs = epochs
        self.data_loader = data_loader
        self.device = device


    def train(self, model: nn.Module, optimizer: torch.optim.Optimizer, seed_value: Optional[int] = 10, verbose: bool = True) -> Dict[str, List[float]]:
        """Train the torch model with the training data provided.

        Args:
            model (nn.Module): the model to train
            optimizer (torch.optim.Optimizer): optimization algorithm to use
            seed_value (int | None, optional): Set a manual random seed to get consistent results.
            If it is None, then no manual seed is set. Defaults to 10.
            verbose (bool, optional): Whether to print training progress or not. Defaults to True.

        Returns:
            Dict[str, List[float]]: Performance evaluation of the training
            process at each step.
        """
        if seed_value is not None: torch.manual_seed(seed_value) # Ensure repeatable results
        model.train() # Set the model in training mode
        model.to(self.device)

        total_steps = len(self.data_loader)
        feedback_step = round(total_steps / 3) + 1
        results = self.evaluation.create_results()

        for epoch in range(self.epochs):
            # Iterate over all batches of the dataset
            for i, (features, labels) in enumerate(self.data_loader):
                # Move the data to the torch device
                features = features.to(self.device)
                labels = labels.to(self.device) # FIXME: Perhaps we need to use .to(self.device, dtype=torch.long)

                outputs = model(features)  # Forward pass
                loss = self.evaluation(outputs, labels, results)  # Evaluation

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and ((i + 1) % feedback_step == 0 or i + 1 == total_steps):
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                            epoch + 1, self.epochs, i + 1, total_steps, loss.item()
                        )
                    )

        return results.as_dict(averaged=False)
