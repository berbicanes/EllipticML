"""Graph neural networks in plain PyTorch (no graph library needed).

Three models, all trained full-batch on the whole transaction graph:

* ``gcn``      two-layer GCN (Kipf & Welling 2017), as in Weber et al.
               H = ReLU(Â X W0),  logits = Â H W1
* ``skipgcn``  GCN plus a linear skip connection from the raw features to the
               output, Weber et al.'s best graph model
               logits = Â H W1 + X Ws
* ``dirsage``  our directed variant in the spirit of GraphSAGE: the
               transaction's own features, the mean of its *inputs* and the mean
               of its *outputs* get separate weights. The EDA shows direction
               matters (illicit transactions have at most three outputs but up to
               ~180 inputs), which the symmetric GCN cannot see.

Why training on the full graph is not leakage: no edge crosses time steps, so
the 15 test-step graphs are disconnected from the 34 training-step graphs.
Messages never flow between them. The loss only uses labelled training rows.
Feature scaling statistics come from training steps only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn

from elliptic import evaluate, graph


def default_device() -> torch.device:
    """Apple-silicon GPU when available (about 6x faster here), else CPU.

    Note: results on MPS are not bit-for-bit reproducible across runs; the
    five-seed averages are.
    """
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def to_torch_sparse(m: sp.spmatrix) -> torch.Tensor:
    m = m.tocoo()
    idx = torch.as_tensor(np.vstack([m.row, m.col]), dtype=torch.long)
    vals = torch.as_tensor(m.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, m.shape, check_invariants=True).coalesce()


def mean_operator(a: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalised adjacency: (M x)_i = mean of x over i's neighbours (0 if none)."""
    deg = np.asarray(a.sum(1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=np.float64), where=deg > 0)
    return (sp.diags(inv) @ a).tocsr().astype(np.float32)


@dataclass
class GraphTensors:
    x: torch.Tensor  # standardised features, all nodes
    a_hat: torch.Tensor  # GCN propagation matrix
    m_in: torch.Tensor  # mean over in-neighbours (senders)
    m_out: torch.Tensor  # mean over out-neighbours (receivers)

    @property
    def device(self) -> torch.device:
        return self.x.device

    def to(self, device: torch.device) -> GraphTensors:
        return GraphTensors(*(t.to(device) for t in (self.x, self.a_hat, self.m_in, self.m_out)))


def build_tensors(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    cols: list[str],
    fit_mask: np.ndarray,
    device: torch.device | None = None,
) -> GraphTensors:
    """Standardise features with statistics from ``fit_mask`` rows only."""
    X = nodes[cols].to_numpy(np.float32)
    mu = X[fit_mask].mean(0)
    sd = X[fit_mask].std(0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    n = len(nodes)
    a = graph.adjacency(edges, n)
    return GraphTensors(
        x=torch.as_tensor(X),
        a_hat=to_torch_sparse(graph.normalized_adjacency(edges, n)),
        m_in=to_torch_sparse(mean_operator(a.T.tocsr())),
        m_out=to_torch_sparse(mean_operator(a)),
    ).to(device or torch.device("cpu"))


class GCN(nn.Module):
    def __init__(self, n_in: int, hidden: int = 100, dropout: float = 0.0, skip: bool = False):
        super().__init__()
        self.w0 = nn.Linear(n_in, hidden)
        self.w1 = nn.Linear(hidden, 2)
        self.skip = nn.Linear(n_in, 2, bias=False) if skip else None
        self.drop = nn.Dropout(dropout)

    def embed(self, g: GraphTensors) -> torch.Tensor:
        return torch.relu(torch.sparse.mm(g.a_hat, self.w0(self.drop(g.x))))

    def forward(self, g: GraphTensors) -> torch.Tensor:
        h = self.embed(g)
        out = torch.sparse.mm(g.a_hat, self.w1(self.drop(h)))
        if self.skip is not None:
            out = out + self.skip(g.x)
        return out


class DirSAGE(nn.Module):
    """Two layers of  h' = ReLU(W_self h + W_in mean_in(h) + W_out mean_out(h))."""

    def __init__(self, n_in: int, hidden: int = 100, dropout: float = 0.0):
        super().__init__()
        self.l1 = nn.ModuleList(
            [
                nn.Linear(n_in, hidden),
                nn.Linear(n_in, hidden, bias=False),
                nn.Linear(n_in, hidden, bias=False),
            ]
        )
        self.l2 = nn.ModuleList(
            [
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, hidden, bias=False),
                nn.Linear(hidden, hidden, bias=False),
            ]
        )
        self.head = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _layer(layer: nn.ModuleList, h: torch.Tensor, g: GraphTensors) -> torch.Tensor:
        self_w, in_w, out_w = layer
        return torch.relu(
            self_w(h) + in_w(torch.sparse.mm(g.m_in, h)) + out_w(torch.sparse.mm(g.m_out, h))
        )

    def embed(self, g: GraphTensors) -> torch.Tensor:
        h = self._layer(self.l1, self.drop(g.x), g)
        return self._layer(self.l2, self.drop(h), g)

    def forward(self, g: GraphTensors) -> torch.Tensor:
        return self.head(self.drop(self.embed(g)))


def make_gnn(kind: str, n_in: int, hidden: int = 100, dropout: float = 0.0) -> nn.Module:
    if kind == "gcn":
        return GCN(n_in, hidden, dropout)
    if kind == "skipgcn":
        return GCN(n_in, hidden, dropout, skip=True)
    if kind == "dirsage":
        return DirSAGE(n_in, hidden, dropout)
    raise ValueError(f"Unknown GNN {kind!r}")


@dataclass
class TrainConfig:
    kind: str = "gcn"
    hidden: int = 100
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 1000
    illicit_weight: float = 0.7  # Weber et al.: 0.3 licit / 0.7 illicit
    eval_every: int = 25


def train(
    g: GraphTensors,
    y: np.ndarray,
    train_mask: np.ndarray,
    cfg: TrainConfig,
    seed: int = 0,
    val_mask: np.ndarray | None = None,
) -> tuple[nn.Module, list[dict]]:
    """Full-batch training. If ``val_mask`` is given, log validation PR-AUC.

    Returns the model after ``cfg.epochs`` epochs and the validation history,
    from which the caller picks the epoch count for the final refit.
    """
    torch.manual_seed(seed)
    dev = g.device
    model = make_gnn(cfg.kind, g.x.shape[1], cfg.hidden, cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    w = torch.tensor([1 - cfg.illicit_weight, cfg.illicit_weight], dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    y_t = torch.as_tensor(y, dtype=torch.long, device=dev)
    tr = torch.as_tensor(np.flatnonzero(train_mask), device=dev)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(g)[tr], y_t[tr])
        loss.backward()
        opt.step()
        if val_mask is not None and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            p = predict(model, g)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss),
                    "val_pr_auc": evaluate.pr_auc(y[val_mask], p[val_mask]),
                }
            )
    return model, history


@torch.no_grad()
def predict(model: nn.Module, g: GraphTensors) -> np.ndarray:
    model.eval()
    return torch.softmax(model(g), dim=1)[:, 1].cpu().numpy()


@torch.no_grad()
def embeddings(model: nn.Module, g: GraphTensors) -> np.ndarray:
    model.eval()
    return model.embed(g).cpu().numpy()
