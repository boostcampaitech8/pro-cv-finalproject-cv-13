from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict


def _load_run_nerve_estimation():
    try:
        from nerve_estimation import run_nerve_estimation  # type: ignore

        return run_nerve_estimation
    except Exception:
        base = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "nerve_estimation",
            base / "__init__.py",
            submodule_search_locations=[str(base)],
        )
        if spec is None or spec.loader is None:
            raise ImportError("Failed to load nerve_estimation package")
        module = importlib.util.module_from_spec(spec)
        sys.modules["nerve_estimation"] = module
        spec.loader.exec_module(module)
        return module.run_nerve_estimation

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segmentation_dir", type=Path, help="Input segmentation root directory.")
    parser.add_argument("output_dir", type=Path, help="Output directory.")
    args = parser.parse_args()

    run_nerve_estimation = _load_run_nerve_estimation()
    results: Dict[str, Any] = run_nerve_estimation(
        segmentation_dir=str(args.segmentation_dir),
        output_dir=str(args.output_dir),
    )

    for nerve in results.get("nerves", []):
        print(f"{nerve['nerve']} ({nerve['side']}): {nerve['type']}")
    for risk in results.get("risks", []):
        print(f"{risk['nerve']} ({risk['side']}): {risk['risk_level']} ({risk['min_distance_mm']:.1f}mm)")


if __name__ == "__main__":
    main()
