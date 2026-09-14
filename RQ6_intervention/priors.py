from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

FROZEN_AT = "2026-08-25"


DATASETS = (
    "greenhouse", "pleiadata", "predist", "wastewater_nutrient",
    "vitaldb", "mimic_cardio", "cgmacros", "shanghai_diabetes",
)
MECHANISTIC_DATASETS = DATASETS[:4]
CLINICAL_DATASETS = DATASETS[4:]

UP, DOWN = +1, -1


@dataclass(frozen=True)
class Prior:

    dataset: str
    action: str
    kind: str
    target: str
    sign: int
    tier: str
    provenance: str
    levels: tuple[int, int] | None = None
    condition: tuple[str, int] | None = None
    event_classes: tuple[int, ...] | None = None

    @property
    def key(self) -> str:
        cond = "" if self.condition is None else f"@{self.condition[0]}={self.condition[1]}"
        return f"{self.dataset}:{self.action}{cond}->{self.target}"


_GH = "greenhouse/dataset.py ACTION_CHANNELS (mechanism stated directly)"
_PL = "pleiadata/dataset.py _MODE_NAMES (mechanism stated directly)"
_PD = "predist: valve opening -> flow -> power; the setpoint is tracked by definition"
_WW = "wastewater: chemical phosphorus removal by metal-salt precipitation"
_VD = "vitaldb pharmacology: propofol `BIS down, MBP down`; remifentanil `HR down, MBP down`"
_MC = "mimic_cardio/build.py DRUG_ITEMS drug groups (vasopressor / chronotropic / oxygenation)"
_CG = "cgmacros: postprandial glucose rise, lag 15-60 min"
_SH = "shanghai: meals raise glucose; insulin/GLP-1 injections lower it"

_M, _C = "mechanistic", "clinical_pending"

PRIORS: tuple[Prior, ...] = (
    Prior("greenhouse", "Tpipe", "continuous", "Tair", UP, _M, _GH),
    Prior("greenhouse", "AssimLight", "continuous", "PARin", UP, _M, _GH),
    Prior("greenhouse", "CO2dosing", "continuous", "CO2air", UP, _M, _GH),
    Prior("greenhouse", "VentLee", "continuous", "Tair", DOWN, _M, _GH),
    Prior("greenhouse", "VentWind", "continuous", "Tair", DOWN, _M, _GH),

    Prior("pleiadata", "setpoint_norm", "continuous", "indoor_temp", UP, _M, _PL,
          condition=("mode", 1)),
    Prior("pleiadata", "setpoint_norm", "continuous", "indoor_temp", DOWN, _M, _PL,
          condition=("mode", 2)),
    Prior("pleiadata", "onoff", "categorical", "step_energy", UP, _M, _PL,
          levels=(0, 1)),

    Prior("predist", "valve_pos", "continuous", "flow", UP, _M, _PD),
    Prior("predist", "valve_pos", "continuous", "heat_power", UP, _M, _PD),
    Prior("predist", "supply_setpoint", "continuous", "s_hc1_supply_temp", UP, _M, _PD),

    Prior("wastewater_nutrient", "metal_dosing", "continuous", "po4", DOWN, _M, _WW),

    Prior("vitaldb", "propofol", "continuous", "BIS", DOWN, _C, _VD),
    Prior("vitaldb", "propofol", "continuous", "ART_MBP", DOWN, _C, _VD),
    Prior("vitaldb", "remifentanil", "continuous", "HR", DOWN, _C, _VD),
    Prior("vitaldb", "remifentanil", "continuous", "ART_MBP", DOWN, _C, _VD),

    Prior("mimic_cardio", "inf_norepinephrine", "continuous", "NBPs", UP, _C, _MC),
    Prior("mimic_cardio", "inf_norepinephrine", "continuous", "NBPm", UP, _C, _MC),
    Prior("mimic_cardio", "inf_phenylephrine", "continuous", "NBPs", UP, _C, _MC),
    Prior("mimic_cardio", "inf_phenylephrine", "continuous", "NBPm", UP, _C, _MC),
    Prior("mimic_cardio", "inf_vasopressin", "continuous", "NBPm", UP, _C, _MC),
    Prior("mimic_cardio", "inf_dobutamine", "continuous", "HR", UP, _C, _MC),
    Prior("mimic_cardio", "inf_dopamine", "continuous", "HR", UP, _C, _MC),
    Prior("mimic_cardio", "FiO2", "continuous", "SpO2", UP, _C, _MC),

    Prior("cgmacros", "meal", "event", "cgm", UP, _C, _CG,
          event_classes=(0, 1, 2, 3)),
    Prior("shanghai_diabetes", "meal", "event", "cgm", UP, _C, _SH,
          event_classes=(0,)),
    Prior("shanghai_diabetes", "injection", "event", "cgm", DOWN, _C, _SH,
          event_classes=tuple(range(1, 17))),
)


@dataclass(frozen=True)
class LagSpec:
    grid: str
    max_lag: int
    rationale: str


LAGS: dict[str, LagSpec] = {
    "vitaldb":             LagSpec("2 s",    90,  "propofol effect-site ke0 half-life ~1.5-3 min -> 0-3 min"),
    "mimic_cardio":        LagSpec("30 min",  4,  "vasopressors act within seconds, but vitals are native q1h -> 0-2 h"),
    "greenhouse":          LagSpec("5 min",  36,  "greenhouse air thermal inertia 30-60 min -> 0-3 h"),
    "predist":             LagSpec("10 min", 12,  "substation thermal inertia 20-40 min -> 0-2 h"),
    "pleiadata":           LagSpec("10 min", 36,  "room thermal inertia 1-2 h -> 0-6 h"),
    "wastewater_nutrient": LagSpec("2 min",  90,  "chemical precipitation + tank residence 30-60 min -> 0-3 h"),
    "cgmacros":            LagSpec("1 min", 180,  "postprandial glucose peak 30-90 min -> 0-3 h"),
    "shanghai_diabetes":   LagSpec("15 min", 12,  "same as above -> 0-3 h"),
}


@dataclass(frozen=True)
class Channels:
    target: tuple[str, ...]
    continuous: tuple[str, ...]
    categorical: tuple[str, ...]
    categorical_n_classes: tuple[int, ...]
    exog: tuple[str, ...]


@lru_cache(maxsize=None)
def channels(dataset: str) -> Channels:
    if dataset == "greenhouse":
        from dataset_utils.greenhouse import dataset as m
        return Channels(tuple(c[0] for c in m.TARGET_CHANNELS),
                        tuple(c[0] for c in m.ACTION_CHANNELS), (), (),
                        tuple(c[0] for c in m.EXOG_CHANNELS))
    if dataset == "pleiadata":
        from dataset_utils.pleiadata import dataset as m
        return Channels(m.TARGET_NAMES, m.CONT_ACTION_NAMES,
                        m.CATEG_ACTION_NAMES, m.CATEG_N_CLASSES, m.EXO_CHANNELS)
    if dataset == "predist":
        from dataset_utils.predist import dataset as m
        return Channels(m.TARGET_NAMES, m.CONT_ACTION_NAMES,
                        m.CATEG_ACTION_NAMES, m.CATEG_N_CLASSES, m.EXO_NAMES)
    if dataset == "wastewater_nutrient":
        from dataset_utils.wastewater_nutrient import dataset as m
        return Channels(m.TARGET_NAMES, m.CONT_ACTION_NAMES,
                        m.CATEG_ACTION_NAMES, m.CATEG_N_CLASSES, m.EXO_NAMES)
    if dataset == "vitaldb":
        from dataset_utils.vital_db import dataset as m
        tgt = tuple(c[0].split("/")[-1] for c in m.TARGET_CHANNELS)
        return Channels(tgt, tuple(c[2] for c in m.ACTION_TS_CHANNELS), (), (), ())
    if dataset == "mimic_cardio":
        from dataset_utils.mimic_cardio import dataset as m
        return Channels(m.TARGET_NAMES, m.CONT_ACTION_NAMES, ("action_combo",), (), ())
    if dataset in ("cgmacros", "shanghai_diabetes"):
        return Channels(("cgm",), (), ("action_combo",), (), ())
    raise KeyError(dataset)


def resolve(dataset: str, name: str, role: str) -> int:
    names = getattr(channels(dataset), role)
    if name not in names:
        raise KeyError(f"{dataset}: {role} has no channel {name!r}; choices {names}")
    return names.index(name)


def check_against_schema(dataset: str, schema) -> None:
    ch = channels(dataset)
    assert len(ch.target) == schema.n_target, (dataset, "target", len(ch.target), schema.n_target)
    assert len(ch.continuous) == schema.n_continuous, (
        dataset, "continuous", len(ch.continuous), schema.n_continuous)
    n_cat = len(schema.cardinalities)
    assert len(ch.categorical) == n_cat, (dataset, "categorical", len(ch.categorical), n_cat)
    assert len(ch.exog) == schema.n_exog, (dataset, "exog", len(ch.exog), schema.n_exog)


def priors_for(dataset: str, tier: str | None = None) -> tuple[Prior, ...]:
    return tuple(p for p in PRIORS
                 if p.dataset == dataset and (tier is None or p.tier == tier))


if __name__ == "__main__":
    import collections
    per_ds = collections.Counter(p.dataset for p in PRIORS)
    print(f"FROZEN_AT = {FROZEN_AT};  {len(PRIORS)} priors, covering "
          f"{len({p.dataset for p in PRIORS})}/8 datasets")
    for ds in list(DATASETS):
        ch = channels(ds)
        for p in priors_for(ds):
            assert p.target in ch.target, p.key
            if p.kind == "continuous":
                assert p.action in ch.continuous, p.key
            elif p.kind == "categorical":
                assert p.action in ch.categorical, p.key
                assert p.levels is not None, p.key
            else:
                assert p.event_classes, p.key
            if p.condition is not None:
                assert p.condition[0] in ch.categorical, p.key
        tier = "mechanistic" if ds in MECHANISTIC_DATASETS else "clinical_pending"
        print(f"  {ds:22s} {per_ds[ds]:2d} priors  [{tier}]  lag<= {LAGS[ds].max_lag:3d} steps ({LAGS[ds].grid})")
    print("channel-name self-check passed.")
