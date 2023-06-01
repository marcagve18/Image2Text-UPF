import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from .evaluation import BasicEvaluation
from typing import Dict

class Tester:
    """Class to evaluate the performance of a trained model."""

    def __init__(
        self,
        evaluation: BasicEvaluation,
        data_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """
        Args:
            evaluation (BasicEvaluation): evaluation instance with the desired
            methods of evaluation, including the loss. See the BasicEvaluation
            class for more details.
            data_loader: the dataset to test the model with.
            device (torch.device): device in which to perform the computations
        """
        self.evaluation = evaluation
        self.data_loader = data_loader
        self.device = device


    def test(self, model: nn.Module) -> Dict[str, float]:
        """Test the performance of a model with a given dataset.

        Args:
            model (nn.Module): the model to test

        Returns:
            Dict[str, float]: Aggregated performance results.
        """
        model.eval()
        model.to(self.device)
        results = self.evaluation.create_results()

        with torch.no_grad():
            for features, labels in self.data_loader:
                # Move the data to the torch device
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass (network predictions)
                outputs = model(features)

                # Evaluate performance of the model
                self.evaluation(outputs, labels, results)

        return results.as_dict(averaged=True)
