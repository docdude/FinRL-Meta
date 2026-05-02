from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch as th


EPS = 1e-8


def resolve_device(gpu_id: int = -1, device: Optional[str] = None) -> th.device:
    if device is not None:
        return th.device(device)
    if th.cuda.is_available() and gpu_id >= 0:
        return th.device(f"cuda:{gpu_id}")
    return th.device("cpu")


@dataclass
class TensorNormalizerState:
    mean: th.Tensor
    var: th.Tensor
    count: float


class TorchRunningMeanStd:
    def __init__(self, shape: tuple[int, ...], device: th.device, epsilon: float = 1e-4):
        self.device = device
        self.mean = th.zeros(shape, dtype=th.float32, device=device)
        self.var = th.ones(shape, dtype=th.float32, device=device)
        self.count = float(epsilon)

    def update(self, batch: th.Tensor) -> None:
        if batch.ndim == len(self.mean.shape):
            batch = batch.unsqueeze(0)
        batch = batch.to(device=self.device, dtype=th.float32)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = float(batch.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: th.Tensor,
        batch_var: th.Tensor,
        batch_count: float,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta.square() * (self.count * batch_count / total_count)
        new_var = (m_a + m_b + correction) / total_count

        self.mean = new_mean
        self.var = new_var.clamp_min(EPS)
        self.count = total_count

    def get_state(self) -> TensorNormalizerState:
        return TensorNormalizerState(
            mean=self.mean.clone(),
            var=self.var.clone(),
            count=self.count,
        )

    def set_state(self, state: TensorNormalizerState | dict) -> None:
        self.mean = state["mean"].clone() if isinstance(state, dict) else state.mean.clone()
        self.var = state["var"].clone() if isinstance(state, dict) else state.var.clone()
        self.count = float(state["count"] if isinstance(state, dict) else state.count)