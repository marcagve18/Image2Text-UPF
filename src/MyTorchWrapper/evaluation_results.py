import numpy as np
from typing import List, Dict, Union


class BasicResults:
    """Class used to store and retrieve the results of a training or testing process.
    """    

    def __init__(self) -> None:
        self.loss = []
        self.batch_sizes = []


    def _log_loss(self, loss: float) -> None:
        """Adds a loss result to the results history.

        Args:
            loss (float): The loss value to store.
            batch_size (int): Size of the batch from which the result has been
            obtained. Used to accurately average the results.
        """        
        self.loss.append(loss)


    def _log_batch_size(self, batch_size: int) -> None:
        """Adds a batch size to the history.

        Args:
            batch_size (int): Size of the batch fto log.
            Used to accurately average the results.
        """        
        self.batch_sizes.append(batch_size)

    @property
    def loss_avg(self) -> float:
        """Returns an average of the loss history. Uses the batch size of each
        result to ponderate each contribution appropriately.
        """        
        a = np.sum(np.multiply(self.loss, self.batch_sizes))
        b = np.sum(self.batch_sizes)
        return a / b
    

    def as_dict(self, averaged=True) -> Dict[str, Union[float, List[float]]]:
        """Create a dictionary representation of all the results.

        Args:
            averaged (bool, optional): Whether to average the results or not. Defaults to True.

        Returns:
            Dict[str, Union[float, List[float]]]: Dictionary representation of the results.
        """        
        return {'loss': self.loss_avg if averaged else self.loss}



class AccuracyResults(BasicResults):
    def __init__(self) -> None:
        super().__init__()
        self.accuracy = []


    def _log_accuracy(self, accuracy: float) -> None:
        """Adds an accuracy result to the results history.

        Args:
            accuracy (float): The accuracy value to store.
        """        
        self.accuracy.append(accuracy)


    @property
    def accuracy_avg(self) -> float:
        a = np.sum(np.multiply(self.accuracy, self.batch_sizes))
        b = np.sum(self.batch_sizes)
        return a / b
    

    def as_dict(self, averaged=True) -> Dict[str, Union[float, List[float]]]:
        dict = super().as_dict(averaged)
        dict['accuracy'] = self.accuracy_avg if averaged else self.accuracy
        return dict
