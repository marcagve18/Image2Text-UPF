import torch
from torch import nn
import torchinfo
from .train import Trainer
from typing import Dict, List, Union


def training_summary(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer: Trainer,
    test_results: Dict[str, Union[float, List[str]]],
) -> str:  
    """Build a performance summary report for future reference.

    Args:
        model (nn.Module): the model from which we want to get an architecture summary.
        optimizer (torch.optim.Optimizer): the optimizer used during training
        trainer (Trainer): trainer instance used to train the model.
        test_results (EvaluationResults): EvaluationResults obtained during
        testing. Used to record the performance of the model.

    Returns:
        str: Summary content. Usually used for printing it to screen or
        writing it to a file.
    """
    batch, _ = next(iter(trainer.data_loader))
    model_stats = torchinfo.summary(model, input_size=batch.shape, device=trainer.device, verbose=0)

    summary = (
        f"Test results: {test_results}\n"
        + f"Loss function used: {trainer.evaluation.loss_criterion}\n"
        + f"Epochs: {trainer.epochs}\n"
        + f"Optimizer: {optimizer}\n"
        + str(model_stats)
    )

    return summary


def get_torch_device(use_gpu: bool = True, debug: bool = False) -> torch.device:
    """Obtains a torch device in which to perform computations

    Args:
        use_gpu (bool, optional): Use GPU if available. Defaults to True.
        debug (bool, optional): Whether to print debug information or not. Defaults to False.

    Returns:
        torch.device: Device in which to perform computations
    """    
    device = torch.device(
        'cuda:0' if use_gpu and torch.cuda.is_available() else
        'mps' if use_gpu and torch.backends.mps.is_available() else
        'cpu'
    )
    if debug: print("Device selected:", device)
    return device


def get_model_params(model: nn.Module) -> int:
    """Computes the number of trainable parameters of a model.

    Args:
        model (nn.Module): Model to evaluate.

    Returns:
        int: Number of parameters of the model
    """  
    params = 0
    for p in model.parameters():
        params += p.numel()
    return params
