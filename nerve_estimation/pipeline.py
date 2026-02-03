"""신경 추정 및 위험도 계산 메인 파이프라인."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np

from .mask_loader import MaskLoader
from .estimators import (
    VagusEstimator,
    EBSLNEstimator,
    RLNEstimator,
    PhrenicEstimator,
    EstimationResult,
)
from .risk import RiskCalculator, RiskResult
from .utils import get_spacing
from .config import SIDES


class NerveEstimationPipeline:
    """신경 추정 및 위험도 계산 파이프라인."""

    def __init__(
        self,
        segmentation_dir: Optional[str] = None,
        normal_structure_dir: Optional[str] = None,
        tumor_path: Optional[str] = None,
    ):
        tumor_dir = None
        if tumor_path is not None:
            tumor_path = Path(tumor_path)
            tumor_dir = str(tumor_path.parent) if tumor_path.is_file() else str(tumor_path)

        self.mask_loader = MaskLoader(
            segmentation_dir=segmentation_dir,
            normal_structure_dir=normal_structure_dir,
            tumor_dir=tumor_dir,
        )

        self.estimators = {
            "vagus": VagusEstimator(self.mask_loader),
            "ebsln": EBSLNEstimator(self.mask_loader),
            "rln": RLNEstimator(self.mask_loader),
            "phrenic": PhrenicEstimator(self.mask_loader),
        }

        self._nerve_results: List[EstimationResult] = []
        self._risk_results: List[RiskResult] = []
        self._failed_nerves: List[Dict[str, str]] = []

    def estimate_nerves(
        self,
        nerves: Optional[List[str]] = None,
        sides: Optional[List[str]] = None,
    ) -> List[EstimationResult]:
        if nerves is None:
            nerves = list(self.estimators.keys())
        if sides is None:
            sides = SIDES

        self._nerve_results = []
        self._failed_nerves = []

        for nerve_name in nerves:
            if nerve_name not in self.estimators:
                print(f"Warning: Unknown nerve '{nerve_name}', skipping")
                continue

            estimator = self.estimators[nerve_name]

            for side in sides:
                result = estimator.estimate(side)

                if result.success:
                    self._nerve_results.append(result)
                else:
                    self._failed_nerves.append({
                        "nerve": nerve_name,
                        "side": side,
                        "error": result.error or "Unknown error",
                    })
                    if result.error:
                        print(f"Warning: {nerve_name} ({side}): {result.error}")

        return self._nerve_results

    def calculate_risk(
        self,
        estimation_results: Optional[List[EstimationResult]] = None,
    ) -> List[RiskResult]:
        if estimation_results is None:
            estimation_results = self._nerve_results

        if not estimation_results:
            print("Warning: No nerve estimation results available for risk calculation")
            return []

        tumor_mask = self.mask_loader.load_mask("tumor")
        if tumor_mask is None:
            print("Warning: No tumor mask found, skipping risk calculation")
            return []

        if np.sum(tumor_mask) == 0:
            print("Warning: Tumor mask is empty, skipping risk calculation")
            return []

        risk_calculator = RiskCalculator(self.mask_loader, tumor_mask)
        self._risk_results = risk_calculator.calculate_all_risks(estimation_results)

        return self._risk_results

    def get_results(self) -> Dict[str, Any]:
        affine = self.mask_loader.get_affine()
        spacing = get_spacing(affine) if affine is not None else None

        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "spacing_mm": spacing.tolist() if spacing is not None else None,
            },
            "nerves": [r.to_dict() for r in self._nerve_results],
            "risks": [r.to_dict() for r in self._risk_results],
            "failed_nerves": self._failed_nerves,
        }

    def save_results(
        self,
        output_dir: Union[str, Path],
        save_report: bool = True,
    ) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}
        results = self.get_results()

        nerve_path = output_dir / "nerve_results.json"
        nerve_data = {
            "metadata": results["metadata"],
            "nerves": results["nerves"],
            "failed_nerves": results["failed_nerves"],
        }
        with open(nerve_path, "w") as f:
            json.dump(nerve_data, f, indent=2)
        saved_files["nerve_results"] = nerve_path

        if results["risks"]:
            risk_path = output_dir / "risk_assessment.json"
            risk_data = {
                "metadata": results["metadata"],
                "risks": results["risks"],
            }
            with open(risk_path, "w") as f:
                json.dump(risk_data, f, indent=2)
            saved_files["risk_assessment"] = risk_path

        if save_report:
            report_path = output_dir / "risk_report.txt"
            report = self._generate_report(results)
            with open(report_path, "w") as f:
                f.write(report)
            saved_files["risk_report"] = report_path

        return saved_files

    def _generate_report(self, results: Dict[str, Any]) -> str:
        lines = [
            "=" * 60,
            "NERVE ESTIMATION AND RISK ASSESSMENT REPORT",
            "=" * 60,
            "",
            f"Generated: {results['metadata']['timestamp']}",
            "",
        ]

        if results["metadata"]["spacing_mm"]:
            spacing = results["metadata"]["spacing_mm"]
            lines.append(f"Voxel spacing (mm): {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f}")
            lines.append("")

        lines.extend([
            "-" * 60,
            "NERVE ESTIMATION RESULTS",
            "-" * 60,
            "",
        ])

        if results["nerves"]:
            for nerve in results["nerves"]:
                lines.append(f"  {nerve['nerve'].upper()} ({nerve['side']})")
                lines.append(f"    Type: {nerve['type']}")
                lines.append(f"    Method: {nerve['method']}")
                lines.append(f"    Reference: {nerve['reference']}")
                lines.append(f"    Uncertainty: {nerve['uncertainty_mm']} mm")
                if nerve.get("warnings"):
                    lines.append(f"    Warnings: {', '.join(nerve['warnings'])}")
                lines.append("")
        else:
            lines.append("  No successful nerve estimations")
            lines.append("")

        if results["failed_nerves"]:
            lines.append("  Failed estimations:")
            for failed in results["failed_nerves"]:
                lines.append(f"    - {failed['nerve']} ({failed['side']}): {failed['error']}")
            lines.append("")

        if results["risks"]:
            lines.extend([
                "-" * 60,
                "RISK ASSESSMENT RESULTS",
                "-" * 60,
                "",
            ])

            risk_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
            sorted_risks = sorted(results["risks"], key=lambda x: risk_order.get(x["risk_level"], 3))

            for risk in sorted_risks:
                level_marker = "!!!" if risk["risk_level"] == "HIGH" else "   "
                lines.append(f"{level_marker} {risk['nerve'].upper()} ({risk['side']}): {risk['risk_level']}")
                lines.append(f"       Min distance: {risk['min_distance_mm']:.1f} mm")
                lines.append(f"       Uncertainty: {risk['uncertainty_mm']:.1f} mm")
                lines.append(f"       Effective distance: {risk['effective_distance_mm']:.1f} mm")
                if risk["overlap"]:
                    lines.append(f"       OVERLAP: {risk['overlap_ratio']*100:.1f}% of nerve points")
                if risk.get("warning"):
                    lines.append(f"       Warning: {risk['warning']}")
                lines.append("")
        else:
            lines.extend([
                "-" * 60,
                "RISK ASSESSMENT",
                "-" * 60,
                "",
                "  No tumor mask available or no successful nerve estimations",
                "  Risk assessment was not performed",
                "",
            ])

        lines.extend([
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])

        return "\n".join(lines)

    def run(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        self.estimate_nerves()
        self.calculate_risk()
        results = self.get_results()

        if output_dir is not None:
            self.save_results(output_dir)

        return results
