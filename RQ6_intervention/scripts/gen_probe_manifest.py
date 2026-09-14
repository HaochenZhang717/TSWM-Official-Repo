from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "RQ6_intervention/scripts/probe_manifest.txt"

DATASETS = ["greenhouse", "pleiadata", "predist", "wastewater_nutrient",
            "vitaldb", "mimic_cardio", "cgmacros", "shanghai_diabetes"]
ARMS = ["none", "res", "res_zero", "film_zero", "xattn", "gate", "concat"]

CONCAT_MAX_WINDOWS = {"vitaldb": 8000}
BATCH = {"vitaldb": 128, "mimic_cardio": 128, "pleiadata": 128, "predist": 128}


SEC4_DATASETS = ["greenhouse", "pleiadata", "predist", "wastewater_nutrient"]
SEC4_PSPACE = ["obs", "vae", "jepa"]
SEC4_ENCODING = [f"{e}__{f}" for e in ("decay", "conv", "attn")
                 for f in ("gate", "film_zero")]


def sec4_lines() -> list[str]:
    out = []
    for ds in SEC4_DATASETS:
        for arm in SEC4_PSPACE + SEC4_ENCODING:
            extra = ["--ladders", "dose"] if arm in SEC4_PSPACE else []
            if ds in BATCH:
                extra += ["--batch-size", str(BATCH[ds])]
            out.append(" ".join([ds, arm, *extra]))
    return out


def lines() -> list[str]:
    out = []
    for ds in DATASETS:
        for arm in ARMS:
            extra = []
            if arm == "gate":
                extra += ["--lambdas", "0", "0.1", "0.3"]
            if arm == "concat":
                extra += ["--ladders", "dose"]
                if ds in CONCAT_MAX_WINDOWS:
                    extra += ["--max-windows", str(CONCAT_MAX_WINDOWS[ds])]
            if ds in BATCH:
                extra += ["--batch-size", str(BATCH[ds])]
            out.append(" ".join([ds, arm, *extra]))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sec4", action="store_true",
                    help="manifest for the prediction-space and plan-encoding axes (4 datasets x 9 arms = 36 lines)")
    a = ap.parse_args()
    ls = sec4_lines() if a.sec4 else lines()
    Path(a.out).write_text("\n".join(ls) + "\n")
    n_cells = (len(SEC4_DATASETS) * (len(SEC4_PSPACE) + len(SEC4_ENCODING)) * 35 if a.sec4
               else len(DATASETS) * (6 * 35 + 35 + 70))
    print(f"wrote {a.out}: {len(ls)} tasks  ->  --array=0-{len(ls) - 1}%16")
    print(f"expected number of cells {n_cells}")
