from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
RANGES_JSON = REPO / "RQ6_intervention/results/action_ranges.json"

NULL_POLICY: dict[str, tuple[str, dict[str, str], dict[str, int]]] = {
    "greenhouse":          ("phys_zero", {},                              {}),
    "vitaldb":             ("phys_zero", {},                              {}),
    "mimic_cardio":        ("norm_zero", {},                              {"action_combo": 0}),
    "wastewater_nutrient": ("phys_zero", {},                              {}),
    "predist":             ("phys_zero", {"supply_setpoint": "unit_median"}, {}),
    "pleiadata":           ("phys_zero", {"setpoint_norm": "unit_median"},   {"onoff": 0}),
    "cgmacros":            ("phys_zero", {},                              {"action_combo": 0}),
    "shanghai_diabetes":   ("phys_zero", {},                              {"action_combo": 0}),
}

_NORM_LO = {"minmax01": 0.0, "log1p01": 0.0, "minmax11": -1.0}
_NORM_HI = {"minmax01": 1.0, "log1p01": 1.0, "minmax11": 1.0}


@dataclass
class ActionSpace:

    dataset: str
    kind: str
    channels: tuple[str, ...]
    lo: torch.Tensor
    hi: torch.Tensor
    null_norm: torch.Tensor
    cat_null: dict[str, int]

    @classmethod
    def from_cache(cls, name: str, blob: dict, device="cpu") -> "ActionSpace":
        spec = json.loads(RANGES_JSON.read_text())[name]
        kind = spec["kind"]
        chans = tuple(spec["channels"])
        n = blob["n"]
        cont = blob["tensors"]["continuous_future"]
        C = cont.shape[-1]
        assert C == len(chans), (name, C, chans)

        if C == 0:
            empty = torch.zeros((n, 0), dtype=torch.float32, device=device)
            return cls(name, kind, chans, empty, empty, empty.clone(),
                       dict(NULL_POLICY[name][2]))

        if spec["scope"] == "global":
            r = torch.tensor(spec["ranges"], dtype=torch.float64, device=device)
            lo = r[:, 0].unsqueeze(0).expand(n, C).contiguous()
            hi = r[:, 1].unsqueeze(0).expand(n, C).contiguous()
        else:
            table = spec["ranges"]
            lo = torch.empty((n, C), dtype=torch.float64, device=device)
            hi = torch.empty((n, C), dtype=torch.float64, device=device)
            for i, u in enumerate(blob["unit_ids"]):
                r = table[str(u)]
                lo[i] = torch.tensor([x[0] for x in r], dtype=torch.float64, device=device)
                hi[i] = torch.tensor([x[1] for x in r], dtype=torch.float64, device=device)

        self = cls(name, kind, chans, lo, hi,
                   torch.zeros((n, C), dtype=torch.float32, device=device),
                   dict(NULL_POLICY[name][2]))
        self.null_norm = self._build_null(blob, device)
        return self

    def _build_null(self, blob: dict, device) -> torch.Tensor:
        default, override, _ = NULL_POLICY[self.dataset]
        n, C = self.lo.shape
        cont = blob["tensors"]["continuous_future"].to(device)
        out = torch.zeros((n, C), dtype=torch.float32, device=device)

        hist = blob["tensors"]["continuous_history"].to(device)
        unit_med = None

        for c, name in enumerate(self.channels):
            policy = override.get(name, default)
            if policy == "norm_zero":
                out[:, c] = 0.0
            elif policy == "phys_zero":
                out[:, c] = self._to_norm(
                    torch.zeros((n, 1), dtype=torch.float64, device=device),
                    slice(None), c)[0].squeeze(1)
            elif policy == "unit_median":
                if unit_med is None:
                    unit_med = _per_unit_median(hist, blob["unit_ids"])
                out[:, c] = unit_med[:, c]
            else:
                raise ValueError(f"{self.dataset}.{name}: unknown baseline policy {policy!r}")
        return out

    def _to_phys(self, norm: torch.Tensor, rows, c: int) -> torch.Tensor:
        lo, hi = self.lo[rows, c], self.hi[rows, c]
        x = norm.double()
        while lo.dim() < x.dim():
            lo, hi = lo.unsqueeze(-1), hi.unsqueeze(-1)
        if self.kind == "minmax01":
            return x * (hi - lo) + lo
        if self.kind == "log1p01":
            return torch.expm1(x * torch.log1p(hi))
        if self.kind == "minmax11":
            return (x + 1.0) * 0.5 * (hi - lo) + lo
        raise ValueError(self.kind)

    def _to_norm(self, phys: torch.Tensor, rows, c: int) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = self.lo[rows, c], self.hi[rows, c]
        x = phys.double()
        while lo.dim() < x.dim():
            lo, hi = lo.unsqueeze(-1), hi.unsqueeze(-1)
        if self.kind == "minmax01":
            y = (x - lo) / (hi - lo)
        elif self.kind == "log1p01":
            y = torch.log1p(x.clamp(min=0.0)) / torch.log1p(hi)
        elif self.kind == "minmax11":
            y = 2.0 * (x - lo) / (hi - lo) - 1.0
        else:
            raise ValueError(self.kind)
        nlo, nhi = _NORM_LO[self.kind], _NORM_HI[self.kind]
        clipped = (y < nlo) | (y > nhi)
        return y.clamp(nlo, nhi).float(), clipped

    def null(self, cont_fut: torch.Tensor, rows) -> torch.Tensor:
        if cont_fut.shape[-1] == 0:
            return cont_fut
        return self.null_norm[rows].unsqueeze(1).expand_as(cont_fut).contiguous()

    def scale(self, cont_fut: torch.Tensor, rows, c: float,
              only: int | None = None) -> tuple[torch.Tensor, float]:
        if cont_fut.shape[-1] == 0:
            return cont_fut, 0.0
        out = cont_fut.clone()
        n_clip = n_tot = 0
        cols = range(cont_fut.shape[-1]) if only is None else (only,)
        for col in cols:
            phys = self._to_phys(cont_fut[..., col], rows, col) * c
            y, clipped = self._to_norm(phys, rows, col)
            out[..., col] = y
            n_clip += int(clipped.sum())
            n_tot += clipped.numel()
        return out, (n_clip / n_tot if n_tot else 0.0)

    def set_dose(self, cont_fut: torch.Tensor, rows, col: int,
                 dose_norm: torch.Tensor | float) -> torch.Tensor:
        out = cont_fut.clone()
        out[..., col] = dose_norm if not torch.is_tensor(dose_norm) else dose_norm
        return out

    def build_marginal(self, blob: dict, col: int) -> torch.Tensor:
        x = blob["tensors"]["continuous_future"][..., col].reshape(-1).float()
        return torch.sort(x).values.to(self.null_norm.device)

    def quantile_shift(self, cont_fut: torch.Tensor, col: int,
                       marginal: torch.Tensor, delta: float) -> torch.Tensor:
        v = marginal
        N = v.numel()
        x = cont_fut[..., col].float().contiguous()
        if delta >= 0:
            rank = torch.searchsorted(v, x, right=True)
            tr = torch.clamp(rank + int(delta * N), max=N - 1)
        else:
            rank = torch.searchsorted(v, x, right=False)
            tr = torch.clamp(rank + int(delta * N), min=0)
        out = cont_fut.clone()
        out[..., col] = v[tr].to(cont_fut.dtype)
        return out

    def dose_quantiles(self, blob: dict, col: int,
                       qs=(0.25, 0.5, 0.75)) -> dict[float, float]:
        x = blob["tensors"]["continuous_future"][..., col].reshape(-1)
        base = self.null_norm[:, col].median().item()
        active = x[(x - base).abs() > 1e-6]
        if active.numel() == 0:
            return {q: base for q in qs}
        return {q: float(torch.quantile(active.float(), q)) for q in qs}

    def active_mask(self, cont_fut: torch.Tensor, rows, col: int) -> torch.Tensor:
        if cont_fut.shape[-1] == 0:
            return torch.ones(cont_fut.shape[0], dtype=torch.bool, device=cont_fut.device)
        base = self.null_norm[rows, col].unsqueeze(1)
        return ((cont_fut[..., col] - base).abs() > 1e-6).any(dim=1)


def _per_unit_median(hist: torch.Tensor, unit_ids: list) -> torch.Tensor:
    n, _L, C = hist.shape
    out = torch.empty((n, C), dtype=torch.float32, device=hist.device)
    by_unit: dict = {}
    for i, u in enumerate(unit_ids):
        by_unit.setdefault(u, []).append(i)
    for u, rows in by_unit.items():
        med = hist[rows].reshape(-1, C).median(dim=0).values
        out[rows] = med
    return out
