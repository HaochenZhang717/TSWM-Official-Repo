from __future__ import annotations

import torch


class Slot:

    def __init__(self, name: str, kind: str, col: int, hi: int | None = None):
        self.name, self.kind, self.col, self.hi = name, kind, col, hi

    def key(self, cont: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        if self.kind == "continuous":
            return cont[..., self.col].float()
        return (cat[..., self.col] == self.hi).float()

    def __repr__(self) -> str:
        return f"Slot({self.name},{self.kind},col={self.col})"


class Constraint:

    def __init__(self, slot_idx: int, tgt: int, sign: int,
                 cond_col: int | None = None, cond_cls: int | None = None,
                 key: str = ""):
        self.slot_idx, self.tgt, self.sign = slot_idx, tgt, sign
        self.cond_col, self.cond_cls, self.key = cond_col, cond_cls, key

    @property
    def conditional(self) -> bool:
        return self.cond_col is not None

    def mask(self, cat: torch.Tensor) -> torch.Tensor | None:
        if not self.conditional:
            return None
        return (cat[..., self.cond_col] == self.cond_cls).float()

    def __repr__(self) -> str:
        c = "" if not self.conditional else f"@col{self.cond_col}=={self.cond_cls}"
        return f"Constraint({self.key or f'slot{self.slot_idx}->tgt{self.tgt}'}{c},s={self.sign:+d})"


def build_spec(dataset: str, schema, train_priors) -> tuple[list[Slot], list[Constraint]]:
    from RQ6_intervention.priors import channels, resolve

    ch = channels(dataset)
    slots = [Slot(n, "continuous", i) for i, n in enumerate(ch.continuous)]
    by_name = {s.name: i for i, s in enumerate(slots)}

    cons: list[Constraint] = []
    for p in train_priors:
        tgt = resolve(dataset, p.target, "target")
        if p.kind == "continuous":
            si = by_name[p.action]
        elif p.kind == "categorical":
            if p.action not in by_name:
                col = ch.categorical.index(p.action)
                slots.append(Slot(p.action, "categorical", col, hi=p.levels[1]))
                by_name[p.action] = len(slots) - 1
            si = by_name[p.action]
        else:
            raise ValueError(f"{p.key}: event priors are not supported")
        cc, cl = None, None
        if p.condition is not None:
            cc, cl = ch.categorical.index(p.condition[0]), p.condition[1]
        cons.append(Constraint(si, tgt, p.sign, cc, cl, p.key))
    return slots, cons
