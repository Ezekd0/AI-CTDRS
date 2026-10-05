"""PyTorch LSTM classifier for tabular intrusion features (see ``sequences.py`` for how features become a sequence).

Imported lazily by ``models.build_model`` so PyTorch is only required when the LSTM is trained.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.training.sequences import sequence_shape, to_sequences


class _LSTMNet(nn.Module):
    def __init__(self, step_size: int, hidden_size: int, num_layers: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(step_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


def _seed_torch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


class LSTMModel:
    name = "lstm"
    model_file = "model.pt"

    def __init__(self, n_classes: int, seed: int, params: dict[str, Any], n_features: int) -> None:
        self.n_classes, self.seed, self.params, self.n_features = n_classes, seed, params, n_features
        self.timesteps, self.step_size, self.padding = sequence_shape(n_features, int(params["features_per_step"]))
        self.net: _LSTMNet | None = None
        self.device = _resolve_device(str(params["device"]))

    def _build(self) -> _LSTMNet:
        p = self.params
        return _LSTMNet(self.step_size, int(p["hidden_size"]), int(p["num_layers"]), self.n_classes, float(p["dropout"]))

    def _batches(self, X: np.ndarray, batch_size: int):
        for start in range(0, len(X), batch_size):
            yield torch.from_numpy(to_sequences(X[start:start + batch_size], self.step_size)).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, class_weights=None) -> dict[str, Any]:
        p = self.params
        _seed_torch(self.seed)
        self.net = self._build().to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=float(p["learning_rate"]), weight_decay=float(p["weight_decay"]))
        weight = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        dataset = TensorDataset(torch.from_numpy(to_sequences(X, self.step_size)), torch.from_numpy(np.asarray(y, dtype=np.int64)))
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=int(p["batch_size"]), shuffle=True, generator=generator)
        has_val = X_val is not None and len(X_val) > 0

        history: list[dict[str, float]] = []
        best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, None, 0
        for epoch in range(1, int(p["epochs"]) + 1):
            self.net.train()
            total, seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), float(p["grad_clip"]))
                optimizer.step()
                total += loss.item() * len(yb)
                seen += len(yb)
            record = {"epoch": epoch, "train_loss": total / max(seen, 1)}
            if has_val:
                record["val_loss"] = self._loss(X_val, y_val, loss_fn)
                if record["val_loss"] < best_loss - 1e-6:
                    best_loss, best_epoch, bad_epochs = record["val_loss"], epoch, 0
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.net.state_dict().items()})
                else:
                    bad_epochs += 1
            history.append(record)
            if has_val and bad_epochs >= int(p["patience"]):
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return {"history": history, "best_epoch": best_epoch, "epochs_run": len(history), "device": str(self.device)}

    @torch.no_grad()
    def _loss(self, X: np.ndarray, y: np.ndarray, loss_fn: nn.Module) -> float:
        self.net.eval()
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        total, start = 0.0, 0
        for xb in self._batches(X, 4096):
            yb = yt[start:start + len(xb)].to(self.device)
            total += loss_fn(self.net(xb), yb).item() * len(yb)
            start += len(xb)
        return total / max(len(yt), 1)

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.net.eval()
        out = [torch.softmax(self.net(xb), dim=1).cpu().numpy() for xb in self._batches(np.asarray(X), 4096)]
        return np.concatenate(out).astype(np.float64)

    def save(self, directory: Path) -> Path:
        path = Path(directory) / self.model_file
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                "arch": {"step_size": self.step_size, "hidden_size": int(self.params["hidden_size"]),
                         "num_layers": int(self.params["num_layers"]), "dropout": float(self.params["dropout"])},
                "n_classes": self.n_classes, "n_features": self.n_features,
                "features_per_step": int(self.params["features_per_step"]),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "LSTMModel":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        params = {"features_per_step": blob["features_per_step"], "device": "cpu", **blob["arch"]}
        obj = cls(blob["n_classes"], 0, params, blob["n_features"])
        obj.net = obj._build()
        obj.net.load_state_dict(blob["state_dict"])
        obj.net.eval()
        return obj
