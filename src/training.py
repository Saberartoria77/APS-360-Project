"""Deterministic CNN-LSTM training and evaluation."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import DatasetBundle
from src.models import CNNLSTM


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 0.001
    patience: int = 3
    seed: int = 42
    device: str | None = None


@dataclass
class TrainResult:
    history: dict[str, list[float]]
    test_probabilities: np.ndarray
    test_predictions: np.ndarray
    parameter_count: int
    best_epoch: int
    best_state: dict[str, torch.Tensor]
    device: str


def _choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = len(labels)
            total_loss += float(loss.detach().cpu()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            count += batch_size
    return total_loss / count, correct / count


def train_model(bundle: DatasetBundle, config: TrainConfig) -> TrainResult:
    """Train with validation checkpointing and return held-out probabilities."""
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _choose_device(config.device)

    model = CNNLSTM(n_features=bundle.x_train.shape[-1]).to(device)
    class_counts = np.bincount(bundle.y_train, minlength=3).astype(np.float32)
    class_weights = len(bundle.y_train) / (3.0 * np.maximum(class_counts, 1.0))
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = _loader(bundle.x_train, bundle.y_train, config.batch_size, True, config.seed)
    val_loader = _loader(bundle.x_val, bundle.y_val, config.batch_size, False, config.seed)

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        val_loss, val_accuracy = _run_epoch(model, val_loader, loss_function, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    probabilities = []
    test_loader = _loader(bundle.x_test, bundle.y_test, config.batch_size, False, config.seed)
    with torch.inference_mode():
        for features, _ in test_loader:
            logits = model(features.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    test_probabilities = np.concatenate(probabilities, axis=0)
    return TrainResult(
        history=history,
        test_probabilities=test_probabilities,
        test_predictions=test_probabilities.argmax(axis=1).astype(np.int64),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        best_epoch=best_epoch,
        best_state={key: value.detach().cpu() for key, value in best_state.items()},
        device=str(device),
    )
