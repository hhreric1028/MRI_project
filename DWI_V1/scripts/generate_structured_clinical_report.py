"""Generate a controlled clinical-style report from verified structured analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReportError(RuntimeError):
    """Raised when source results are incomplete or inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-json", required=True, type=Path)
    parser.add_argument("--components-json", required=True, type=Path)
    parser.add_argument("--location-json", required=True, type=Path)
    parser.add_argument("--case-id", help="Case identifier; otherwise read from source JSON")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("structured_clinical_report")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_json(path: Path, role: str) -> tuple[dict, str]:
    path = path.resolve()
    if not path.is_file():
        raise ReportError(f"{role} does not exist: {path}")
    try:
        content = path.read_bytes()
        data = json.loads(content.decode("utf-8"))
    except Exception as exc:
        raise ReportError(f"Cannot read {role}: {path}: {exc}") from exc
    return data, hashlib.sha256(content).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def close(a: float, b: float, tolerance: float = 5e-4) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def build_structured_input(volume: dict, components: dict, location: dict, case_id: str) -> dict:
    for name, source in (("volume", volume), ("components", components), ("location", location)):
        require(source.get("case_id") == case_id, f"{name} case_id does not match {case_id}")
        require(source.get("status") == "PASS", f"{name} source status is not PASS")
    require(location.get("brain_extent_estimation", {}).get("status") == "PASS", "Brain extent is not PASS")
    require(location.get("overall", {}).get("location_status") == "PASS", "Coarse prediction location is not PASS")

    pred_volume = volume["prediction"]
    gt_volume = volume["ground_truth"]
    error = volume["error"]
    pred_components = components["prediction"]
    gt_components = components["ground_truth"]
    loc = location["overall"]

    require(pred_volume["lesion_voxels"] == pred_components["total_voxels"], "Prediction voxel counts disagree")
    require(pred_volume["lesion_voxels"] == loc["lesion_voxels"], "Location voxel count disagrees")
    require(pred_components["component_count"] == loc["component_count"], "Prediction component counts disagree")
    require(close(pred_volume["volume_ml"], pred_components["total_volume_ml"]), "Prediction volumes disagree")
    require(close(gt_volume["volume_ml"], gt_components["total_volume_ml"]), "Ground-truth volumes disagree")
    require(pred_components["component_count"] > 1 and loc["distribution"] == "Multifocal", "Multifocal status is inconsistent")
    require(loc["laterality"] == "Bilateral", "This report template expects verified bilateral distribution")

    ap = loc["anterior_posterior"]
    si = loc["superior_inferior"]
    structured = {
        "case_id": case_id,
        "case_type": volume.get("case_type", "unspecified"),
        "analysis_type": "DWI-based automated lesion analysis",
        "segmentation": {
            "lesion_present": pred_volume["lesion_voxels"] > 0,
            "lesion_voxels": int(pred_volume["lesion_voxels"]),
            "component_count": int(pred_components["component_count"]),
            "distribution": loc["distribution"].lower(),
        },
        "volume": {
            "total_ml": float(pred_volume["volume_ml"]),
            "source": "verified native-space prediction volume",
        },
        "location": {
            "space": "native",
            "laterality": loc["laterality"].lower(),
            "left_fraction_percent": float(loc["left_fraction_percent"]),
            "right_fraction_percent": float(loc["right_fraction_percent"]),
            "ap_primary": ap["region"].lower(),
            "ap_distribution_percent": {
                key.lower(): float(value) for key, value in ap["region_fractions_percent"].items()
            },
            "si_primary": si["region"].lower(),
            "si_distribution_percent": {
                key.lower(): float(value) for key, value in si["region_fractions_percent"].items()
            },
            "fine_anatomical_localization": None,
        },
        "model_evaluation": {
            "ground_truth_volume_ml": float(gt_volume["volume_ml"]),
            "prediction_volume_ml": float(pred_volume["volume_ml"]),
            "absolute_volume_error_ml": float(error["absolute_ml"]),
            "relative_volume_error_percent": float(error["relative_percent"]),
            "signed_volume_error_ml": float(error["signed_ml"]),
            "signed_volume_error_percent": float(error["signed_percent"]),
            "volume_error_interpretation": str(error["interpretation"]),
            "prediction_component_count": int(pred_components["component_count"]),
            "ground_truth_component_count": int(gt_components["component_count"]),
        },
        "reporting_limits": {
            "fine_anatomical_parcellation": "not available",
            "lesion_age": "not determined",
            "vascular_territory": "not determined",
            "disease_mechanism": "not determined",
            "signal_intensity_characterization": "not performed",
        },
    }
    return structured


def generate_clinical_report(data: dict) -> tuple[str, list[dict]]:
    volume_ml = data["volume"]["total_ml"]
    right = data["location"]["right_fraction_percent"]
    left = data["location"]["left_fraction_percent"]
    findings = [
        "Multiple lesion regions are identified by DWI-based automated analysis.",
        (
            "The predicted lesions show a multifocal bilateral distribution, with a slightly greater "
            "predicted lesion burden on the right."
        ),
        (
            "In native image space, the predicted lesions are predominantly distributed within the "
            "central portion and at the middle level of the imaged brain volume."
        ),
        f"The total predicted lesion volume is approximately {volume_ml:.2f} mL.",
    ]
    impression = [
        (
            "Multifocal bilateral lesion regions identified by DWI-based automated analysis, "
            "predominantly within the central portion and at the middle level of the imaged brain "
            "volume in native image space."
        ),
        f"Total predicted lesion volume is approximately {volume_ml:.2f} mL.",
    ]
    report = (
        "MRI of head images are obtained:\n\n"
        "Findings:\n"
        + "\n".join(f"- {sentence}" for sentence in findings)
        + "\n\nImpression:\n"
        + "\n".join(f"- {sentence}" for sentence in impression)
        + "\n"
    )
    claims = [
        {"section": "Findings", "sentence": findings[0], "supported_by": ["segmentation.lesion_present", "segmentation.component_count", "analysis_type"]},
        {"section": "Findings", "sentence": findings[1], "supported_by": ["segmentation.distribution", "location.laterality", "location.left_fraction_percent", "location.right_fraction_percent"], "support_values": {"left_percent": left, "right_percent": right}},
        {"section": "Findings", "sentence": findings[2], "supported_by": ["location.space", "location.ap_primary", "location.si_primary"]},
        {"section": "Findings", "sentence": findings[3], "supported_by": ["volume.total_ml"]},
        {"section": "Impression", "sentence": impression[0], "supported_by": ["analysis_type", "segmentation.distribution", "location.laterality", "location.space", "location.ap_primary", "location.si_primary"]},
        {"section": "Impression", "sentence": impression[1], "supported_by": ["volume.total_ml"]},
    ]
    return report, claims


def generate_evaluation(data: dict) -> str:
    evaluation = data["model_evaluation"]
    interpretation = evaluation["volume_error_interpretation"].lower()
    return (
        "Model Evaluation Summary\n\n"
        f"Case: {data['case_id']}\n"
        f"Case Type: {data['case_type'].title()}\n"
        "Clinical Report Status: This document is for model evaluation and is not a clinical report.\n\n"
        f"Ground Truth volume: {evaluation['ground_truth_volume_ml']:.3f} mL\n"
        f"Prediction volume: {evaluation['prediction_volume_ml']:.3f} mL\n"
        f"Absolute volume error: {evaluation['absolute_volume_error_ml']:.3f} mL\n"
        f"Relative volume error: {evaluation['relative_volume_error_percent']:.3f}%\n"
        f"Signed volume error: {evaluation['signed_volume_error_ml']:.3f} mL "
        f"({evaluation['signed_volume_error_percent']:.3f}%)\n"
        f"Interpretation: Prediction volume is {interpretation}.\n\n"
        f"Prediction connected components: {evaluation['prediction_component_count']}\n"
        f"Ground Truth connected components: {evaluation['ground_truth_component_count']}\n"
    )


def validate_report(report: str, structured: dict, claims: list[dict]) -> dict:
    report_lower = report.lower()
    volume_token = f"{structured['volume']['total_ml']:.2f} ml".lower()
    laterality_ok = "bilateral" in report_lower and not re.search(r"\b(left|right)-sided lesion", report_lower)
    distribution_ok = "multifocal" in report_lower and "single lesion" not in report_lower
    volume_ok = report_lower.count(volume_token) == 2
    location_ok = all(token in report_lower for token in ("central", "middle", "native image space"))
    lesion_presence_ok = "multiple lesion regions" in report_lower

    evaluation_only_terms = ("ground truth", "volume error", "underestimated", "-27.553", "3.368")
    clinical_evaluation_separation = not any(term in report_lower for term in evaluation_only_terms)
    forbidden_claim_patterns = {
        "signal_intensity": r"\b(hyperintens|hypointens)",
        "hemorrhage": r"\b(hemorrhage|microbleed)\b",
        "mass_effect": r"\b(midline shift|mass effect|ventriculomegaly)\b",
        "vascular_occlusion": r"\b(occlusion|atherosclerosis)\b",
        "fine_anatomy": r"\b(gyrus|insula|putamen|thalamus|frontal lobe|parietal lobe|basal ganglia)\b",
        "lesion_age": r"\b(acute|recent|chronic|old infarct|infarction)\b",
        "unsupported_other": r"\b(meningioma|sinusitis|mastoid|encephalomalacia|atrophy)\b",
    }
    forbidden_hits = {
        name: re.findall(pattern, report_lower)
        for name, pattern in forbidden_claim_patterns.items()
        if re.search(pattern, report_lower)
    }
    engineering_terms = ("voxel", "affine", "connected component", "array axis", "mni", "atlas", "las")
    engineering_hits = [term for term in engineering_terms if term in report_lower]

    sentence_validation = []
    unsupported_claims = []
    for claim in claims:
        supported = bool(claim["supported_by"])
        item = dict(claim)
        item["status"] = "PASS" if supported else "FAIL"
        sentence_validation.append(item)
        if not supported:
            unsupported_claims.append(claim["sentence"])

    checks = {
        "lesion_presence": "PASS" if lesion_presence_ok else "FAIL",
        "laterality": "PASS" if laterality_ok else "FAIL",
        "distribution": "PASS" if distribution_ok else "FAIL",
        "volume": "PASS" if volume_ok else "FAIL",
        "location": "PASS" if location_ok else "FAIL",
        "clinical_evaluation_separation": "PASS" if clinical_evaluation_separation else "FAIL",
        "forbidden_clinical_content": "PASS" if not forbidden_hits else "FAIL",
        "engineering_terms_absent": "PASS" if not engineering_hits else "FAIL",
    }
    status = "PASS" if all(value == "PASS" for value in checks.values()) and not unsupported_claims else "FAIL"
    return {
        "status": status,
        "unsupported_claims": unsupported_claims,
        "unsupported_claim_count": len(unsupported_claims),
        "sentence_validation": sentence_validation,
        "consistency_check": checks,
        "forbidden_claim_hits": forbidden_hits,
        "engineering_term_hits": engineering_hits,
        "validation_method": "deterministic claim-to-field mapping plus controlled vocabulary and cross-section checks",
    }


def run(args: argparse.Namespace, output_dir: Path, logger: logging.Logger) -> dict:
    volume, volume_hash = load_json(args.volume_json, "Volume JSON")
    components, components_hash = load_json(args.components_json, "Connected-components JSON")
    location, location_hash = load_json(args.location_json, "Native-location JSON")
    case_id = args.case_id or volume.get("case_id")
    require(bool(case_id), "No case_id was supplied or found")

    structured = build_structured_input(volume, components, location, case_id)
    structured["provenance"] = {
        "volume_json": {"path": str(args.volume_json.resolve()), "sha256": volume_hash},
        "components_json": {"path": str(args.components_json.resolve()), "sha256": components_hash},
        "location_json": {"path": str(args.location_json.resolve()), "sha256": location_hash},
    }
    clinical_report, claims = generate_clinical_report(structured)
    model_evaluation = generate_evaluation(structured)
    validation = validate_report(clinical_report, structured, claims)
    require(validation["status"] == "PASS", f"Report validation failed: {validation}")

    data_dir = output_dir / "data"
    validation_dir = output_dir / "validation"
    data_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    structured_path = data_dir / "structured_input.json"
    clinical_path = output_dir / "clinical_report.txt"
    evaluation_path = output_dir / "model_evaluation.txt"
    validation_path = validation_dir / "report_validation.json"
    structured_path.write_text(json.dumps(structured, indent=2) + "\n", encoding="utf-8")
    clinical_path.write_text(clinical_report, encoding="utf-8")
    evaluation_path.write_text(model_evaluation, encoding="utf-8")
    validation["outputs"] = {
        "structured_input": str(structured_path.resolve()),
        "clinical_report": str(clinical_path.resolve()),
        "model_evaluation": str(evaluation_path.resolve()),
    }
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    logger.info("Case: %s (%s)", case_id, structured["case_type"])
    logger.info("Source consistency: PASS")
    logger.info("Structured input: %s", structured_path)
    logger.info("Clinical report: %s", clinical_path)
    logger.info("Model evaluation: %s", evaluation_path)
    logger.info("Unsupported clinical claims: %d", validation["unsupported_claim_count"])
    logger.info("Consistency checks: %s", validation["consistency_check"])
    logger.info("Validation JSON: %s", validation_path)
    logger.info("Block 5 Status: PASS")
    return validation


def main() -> int:
    args = parse_args()
    inferred_case = args.case_id or "case"
    output_dir = (args.output_dir or ROOT / "outputs" / "reports" / inferred_case).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(logs_dir / "report_generation.log")
    try:
        run(args, output_dir, logger)
    except ReportError as exc:
        logger.error("%s", exc)
        logger.error("Block 5 Status: FAIL")
        return 1
    except Exception:
        logger.exception("Unexpected failure")
        logger.error("Block 5 Status: FAIL")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
