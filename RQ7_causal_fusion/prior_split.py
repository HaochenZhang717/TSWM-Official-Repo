from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RQ6_intervention.priors import PRIORS, MECHANISTIC_DATASETS, priors_for

FROZEN_AT = "2026-08-25"


TRAIN_PRIORS: tuple[str, ...] = (
    "greenhouse:Tpipe->Tair",
    "greenhouse:VentLee->Tair",
    "greenhouse:VentWind->Tair",
    "pleiadata:setpoint_norm@mode=1->indoor_temp",
    "pleiadata:onoff->step_energy",
    "predist:valve_pos->flow",
    "predist:valve_pos->heat_power",
)

HELDOUT_CLEAN: tuple[str, ...] = (
    "greenhouse:AssimLight->PARin",
    "greenhouse:CO2dosing->CO2air",
    "predist:supply_setpoint->s_hc1_supply_temp",
    "wastewater_nutrient:metal_dosing->po4",
)

HELDOUT_CONDITIONAL: tuple[str, ...] = (
    "pleiadata:setpoint_norm@mode=2->indoor_temp",
)

CLINICAL_TRAIN_PRIORS: tuple[str, ...] = (
    "vitaldb:propofol->BIS",
    "vitaldb:propofol->ART_MBP",
    "mimic_cardio:inf_norepinephrine->NBPs",
    "mimic_cardio:inf_norepinephrine->NBPm",
    "mimic_cardio:inf_dobutamine->HR",
    "mimic_cardio:FiO2->SpO2",
)

CLINICAL_HELDOUT_CLEAN: tuple[str, ...] = (
    "vitaldb:remifentanil->HR",
    "vitaldb:remifentanil->ART_MBP",
    "mimic_cardio:inf_phenylephrine->NBPs",
    "mimic_cardio:inf_phenylephrine->NBPm",
    "mimic_cardio:inf_vasopressin->NBPm",
    "mimic_cardio:inf_dopamine->HR",
    "cgmacros:meal->cgm",
    "shanghai_diabetes:meal->cgm",
    "shanghai_diabetes:injection->cgm",
)

SPLITS = {
    "default": (TRAIN_PRIORS, HELDOUT_CLEAN, HELDOUT_CONDITIONAL),
    "clinical": (CLINICAL_TRAIN_PRIORS, CLINICAL_HELDOUT_CLEAN, ()),
}

SPLIT_TIER = {"default": "mechanistic", "clinical": "clinical_pending"}


def _by_key() -> dict:
    return {p.key: p for p in PRIORS}


def train_priors(dataset: str, split: str = "default") -> list:
    keys = set(SPLITS[split][0])
    return [p for p in priors_for(dataset, tier=SPLIT_TIER[split]) if p.key in keys]


def heldout_priors(dataset: str, split: str = "default",
                   include_conditional: bool = False) -> list:
    _, clean, cond = SPLITS[split]
    keys = set(clean) | (set(cond) if include_conditional else set())
    return [p for p in priors_for(dataset, tier=SPLIT_TIER[split]) if p.key in keys]


def role(key: str, split: str = "default") -> str:
    tr, cl, co = SPLITS[split]
    return ("train" if key in tr else "heldout_clean" if key in cl
            else "heldout_conditional" if key in co else "unused")


def _check() -> None:
    known = _by_key()

    for split, (tr, cl, co) in SPLITS.items():
        tier = SPLIT_TIER[split]
        mech = {p.key for p in PRIORS if p.tier == tier}
        allk = list(tr) + list(cl) + list(co)
        for k in allk:
            assert k in known, f"[{split}] unknown prior key: {k}"
            assert known[k].tier == tier, f"[{split}] {k} is not {tier}"
        assert len(allk) == len(set(allk)), f"[{split}] a prior is assigned twice"
        assert set(allk) == mech, (
            f"[{split}] does not cover every {tier} prior; missing {sorted(mech - set(allk))}, "
            f"extra {sorted(set(allk) - mech)}")
        by_ch: dict[tuple, set] = {}
        for k in allk:
            p = known[k]
            by_ch.setdefault((p.dataset, p.action), set()).add(
                "train" if k in tr else "heldout")
        leaky = {ch: v for ch, v in by_ch.items() if len(v) > 1}
        leaky = {ch: v for ch, v in leaky.items() if ch != ("pleiadata", "setpoint_norm")}
        assert not leaky, f"[{split}] action channel appears on both sides: {leaky}"
    print(f"FROZEN_AT = {FROZEN_AT}  self-check passed")


if __name__ == "__main__":
    _check()
    known = _by_key()
    for split in SPLITS:
        tr, cl, co = SPLITS[split]
        print(f"\n=== split = {split} ===")
        print(f"TRAIN (supplied to the loss, not evidence)   {len(tr)} priors")
        for k in tr:
            p = known[k]
            print(f"    {k:44s} {p.sign:+d}  {p.kind}")
        print(f"HELD-OUT clean (primary criterion)          {len(cl)} priors")
        for k in cl:
            print(f"    {k:44s} {known[k].sign:+d}")
        print(f"HELD-OUT conditional (reported separately)  {len(co)} priors")
        for k in co:
            print(f"    {k:44s} {known[k].sign:+d}")
        for ds in MECHANISTIC_DATASETS:
            n_tr = len(train_priors(ds, split))
            n_ho = len(heldout_priors(ds, split))
            print(f"    {ds:22s} train={n_tr}  heldout_clean={n_ho}")
