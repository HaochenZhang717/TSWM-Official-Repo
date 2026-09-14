from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Any

from dataset_utils.greenhouse.dataset import GreenhouseDataset
from dataset_utils.cgmacros.dataset import CGMacrosDataset
from dataset_utils.shanghai_diabetes.dataset import ShanghaiDiabetesDataset
from dataset_utils.vital_db.dataset import (
    TARGET_CHANNELS as VITAL_TARGET_CHANNELS,
    VitalDBDataset,
    load_listings,
    select_caseids,
)
from dataset_utils.pleiadata.dataset import PLEIADataHVACDataset
from dataset_utils.predist.dataset import PreDistSubstationDataset
from dataset_utils.wastewater_nutrient.dataset import WastewaterNutrientDataset
from dataset_utils.mimic_cardio.dataset import MimicCardioDataset

from dataset_utils.common.windowing import build_action_combo_mapping


_THIS = Path(__file__).resolve()
CODE_ROOT = _THIS.parents[2]
REPO_ROOT = CODE_ROOT.parent
_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

DEFAULT_GREENHOUSE_ROOT = _DATA_ROOT / "greenhouse3/TimeSeries"
DEFAULT_VITALDB_ROOT = _DATA_ROOT / "vital_db"
DEFAULT_CGMACROS_ROOT = _DATA_ROOT / "diabetes_datasets/cgmacros"
DEFAULT_SHANGHAI_DIABETES_ROOT = _DATA_ROOT / "diabetes_datasets/Shanghai_T1DM_T2DM"
DEFAULT_PLEIADATA_ROOT = _DATA_ROOT / "PLEIAData"
DEFAULT_PREDIST_ROOT = _DATA_ROOT / "PreDist/predist_dataset/manufacturer_2"
DEFAULT_WASTEWATER_ROOT = _DATA_ROOT / "Wastewater Treatment Plant Data for Nutrient Removal System/IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv"
DEFAULT_MIMIC_CARDIO_ROOT = _DATA_ROOT / "mimic_cardio"

DEFAULT_ROOTS: dict[str, Path] = {
    "greenhouse": DEFAULT_GREENHOUSE_ROOT,
    "vitaldb": DEFAULT_VITALDB_ROOT,
    "cgmacros": DEFAULT_CGMACROS_ROOT,
    "shanghai_diabetes": DEFAULT_SHANGHAI_DIABETES_ROOT,
    "pleiadata": DEFAULT_PLEIADATA_ROOT,
    "predist": DEFAULT_PREDIST_ROOT,
    "wastewater_nutrient": DEFAULT_WASTEWATER_ROOT,
    "mimic_cardio": DEFAULT_MIMIC_CARDIO_ROOT,
}

ALL_DATASETS = [
    "greenhouse", "vitaldb", "cgmacros", "shanghai_diabetes",
    "pleiadata", "predist", "wastewater_nutrient", "mimic_cardio",
]

SPARSE_ACTION_DATASETS = {"cgmacros", "shanghai_diabetes", "mimic_cardio"}


def dataset_names(dataset: str) -> list[str]:
    if dataset == "all":
        return list(ALL_DATASETS)
    return [dataset]


def split_greenhouse_validation(data_root: Path, val_ratio: float, seed: int, **dataset_kwargs: Any) -> tuple[GreenhouseDataset, dict[str, Any]]:
    teams = sorted(
        path.name for path in data_root.iterdir()
        if path.is_dir() and (path / "GreenhouseClimate.xlsx").exists()
    )
    if not teams:
        raise FileNotFoundError(f"No greenhouse teams found under {data_root}")

    shuffled = teams[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    val_teams = shuffled[:n_val]
    train_teams = shuffled[n_val:]

    dataset = GreenhouseDataset(data_root, teams=val_teams, **dataset_kwargs)
    split_info = {
        "split_unit": "team",
        "train_units": train_teams,
        "validation_units": val_teams,
        "n_train_units": len(train_teams),
        "n_validation_units": len(val_teams),
    }
    return dataset, split_info


def split_vitaldb_validation(
    data_root: Path,
    val_ratio: float,
    seed: int,
    split: str,
    download: bool,
    **dataset_kwargs: Any,
) -> tuple[VitalDBDataset, dict[str, Any]]:
    trks, cases = load_listings(data_root, download=download)
    cases_idx = cases.set_index("caseid")
    target_tracks = [channel[0] for channel in dataset_kwargs.get("target_channels", VITAL_TARGET_CHANNELS)]
    caseids = select_caseids(trks, target_tracks, require_all_targets=True)

    if split != "all":
        keep_dept = {
            "general": "General surgery",
            "thoracic": "Thoracic surgery",
            "urology": "Urology",
            "gynecology": "Gynecology",
        }[split]
        caseids = [caseid for caseid in caseids if cases_idx["department"].get(caseid) == keep_dept]

    if not caseids:
        raise ValueError("No VitalDB caseids available after filtering.")

    subject_of = {caseid: int(cases_idx.loc[caseid, "subjectid"]) for caseid in caseids}
    subjects = sorted(set(subject_of.values()))
    shuffled_subjects = subjects[:]
    random.Random(seed).shuffle(shuffled_subjects)
    n_val_subjects = max(1, round(len(shuffled_subjects) * val_ratio))
    val_subjects = set(shuffled_subjects[:n_val_subjects])

    val_caseids = [caseid for caseid in caseids if subject_of[caseid] in val_subjects]
    train_caseids = [caseid for caseid in caseids if subject_of[caseid] not in val_subjects]

    dataset = VitalDBDataset(
        data_root,
        split=split,
        caseids=val_caseids,
        download=download,
        **dataset_kwargs,
    )
    split_info = {
        "split_unit": "subjectid",
        "n_train_subjects": len(set(subject_of[caseid] for caseid in train_caseids)),
        "n_validation_subjects": len(val_subjects),
        "n_train_caseids": len(train_caseids),
        "n_validation_caseids": len(val_caseids),
    }
    return dataset, split_info


def split_cgmacros_validation(
    data_root: Path,
    val_ratio: float,
    seed: int,
    split: str,
    **dataset_kwargs: Any,
) -> tuple[CGMacrosDataset, dict[str, Any]]:
    train_dataset, val_dataset = CGMacrosDataset.subject_split(
        data_root,
        test_ratio=val_ratio,
        seed=seed,
        split=split,
        **dataset_kwargs,
    )
    split_info = {
        "split_unit": "subject",
        "n_train_windows": len(train_dataset),
        "n_validation_windows": len(val_dataset),
        "split_filter": split,
    }
    return val_dataset, split_info


def split_shanghai_diabetes_validation(
    data_root: Path,
    val_ratio: float,
    seed: int,
    split: str,
    **dataset_kwargs: Any,
) -> tuple[ShanghaiDiabetesDataset, dict[str, Any]]:
    train_dataset, val_dataset = ShanghaiDiabetesDataset.subject_split(
        data_root,
        test_ratio=val_ratio,
        seed=seed,
        split=split,
        **dataset_kwargs,
    )
    split_info = {
        "split_unit": "subject",
        "n_train_windows": len(train_dataset),
        "n_validation_windows": len(val_dataset),
        "split_filter": split,
    }
    return val_dataset, split_info


def split_pleiadata_validation(
    data_root: Path,
    val_ratio: float,
    seed: int,
    **dataset_kwargs: Any,
) -> tuple[PLEIADataHVACDataset, dict[str, Any]]:
    dataset = PLEIADataHVACDataset(
        str(data_root),
        split="val",
        test_ratio=val_ratio,
        seed=seed,
        **dataset_kwargs,
    )
    split_info = {
        "split_unit": "subject",
        "n_validation_subjects": len(dataset.subjects),
        "n_validation_windows": len(dataset),
    }
    return dataset, split_info


def split_predist_validation(
    data_root: Path,
    val_ratio: float,
    seed: int,
    **dataset_kwargs: Any,
) -> tuple[PreDistSubstationDataset, dict[str, Any]]:
    train_dataset, val_dataset = PreDistSubstationDataset.subject_split(
        data_root,
        test_ratio=val_ratio,
        seed=seed,
        **dataset_kwargs,
    )
    split_info = {
        "split_unit": "substation",
        "n_train_windows": len(train_dataset),
        "n_validation_windows": len(val_dataset),
    }
    return val_dataset, split_info


def split_mimic_cardio_validation(
    data_root: Path,
    val_ratio: float,
    seed: int,
    grid_minutes: int = 30,
    max_stays: int | None = None,
    **dataset_kwargs: Any,
) -> tuple[MimicCardioDataset, dict[str, Any]]:
    train_dataset, val_dataset = MimicCardioDataset.subject_split(
        data_root,
        test_ratio=val_ratio,
        seed=seed,
        grid_minutes=grid_minutes,
        max_stays=max_stays,
        **dataset_kwargs,
    )
    split_info = {
        "split_unit": "subject_id (patient)",
        "n_train_windows": len(train_dataset),
        "n_validation_windows": len(val_dataset),
    }
    return val_dataset, split_info


def split_wastewater_validation(
    data_path: Path,
    val_ratio: float,
    **dataset_kwargs: Any,
) -> tuple[WastewaterNutrientDataset, dict[str, Any]]:
    train_ratio = 1.0 - val_ratio
    train_dataset, val_dataset = WastewaterNutrientDataset.subject_split(
        str(data_path),
        train_ratio=train_ratio,
        **dataset_kwargs,
    )
    split_info = {
        "split_unit": "time",
        "train_ratio": train_ratio,
        "n_train_windows": len(train_dataset),
        "n_validation_windows": len(val_dataset),
    }
    return val_dataset, split_info


def build_validation_dataset(
    name: str,
    *,
    val_ratio: float,
    seed: int,
    seq_len: int,
    stride: int,
    root: Path | str | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], dict[tuple[int, ...], int] | None]:
    options = options or {}
    data_root = Path(root) if root is not None else DEFAULT_ROOTS[name]
    dataset_kwargs = {"seq_len": seq_len, "stride": stride}

    if name == "greenhouse":
        dataset, split_info = split_greenhouse_validation(
            data_root, val_ratio=val_ratio, seed=seed, **dataset_kwargs,
        )
    elif name == "vitaldb":
        dataset, split_info = split_vitaldb_validation(
            data_root,
            val_ratio=val_ratio,
            seed=seed,
            split=options.get("split", "all"),
            download=options.get("download", False),
            **dataset_kwargs,
        )
    elif name == "cgmacros":
        dataset, split_info = split_cgmacros_validation(
            data_root,
            val_ratio=val_ratio,
            seed=seed,
            split=options.get("split", "all"),
            resample_minutes=options.get("resample_minutes", None),
            **dataset_kwargs,
        )
    elif name == "shanghai_diabetes":
        dataset, split_info = split_shanghai_diabetes_validation(
            data_root,
            val_ratio=val_ratio,
            seed=seed,
            split=options.get("split", "all"),
            **dataset_kwargs,
        )
    elif name == "pleiadata":
        dataset, split_info = split_pleiadata_validation(
            data_root, val_ratio=val_ratio, seed=seed, **dataset_kwargs,
        )
    elif name == "predist":
        dataset, split_info = split_predist_validation(
            data_root, val_ratio=val_ratio, seed=seed, **dataset_kwargs,
        )
    elif name == "wastewater_nutrient":
        dataset, split_info = split_wastewater_validation(
            data_root, val_ratio=val_ratio, **dataset_kwargs,
        )
    elif name == "mimic_cardio":
        dataset, split_info = split_mimic_cardio_validation(
            data_root,
            val_ratio=val_ratio,
            seed=seed,
            grid_minutes=options.get("grid_minutes", 30),
            max_stays=options.get("max_stays", None),
            **dataset_kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    combo_to_id = None
    if name in SPARSE_ACTION_DATASETS:
        combo_to_id = build_action_combo_mapping(dataset)
        split_info["action_combo_vocab_size"] = len(combo_to_id)

    return dataset, split_info, combo_to_id
