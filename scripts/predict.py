"""Predict points from prepared pregame features with the frozen model artifact."""

import argparse
from pathlib import Path
import tempfile

from scripts.model_artifact import load_artifact, predict_frame, read_inference_csv


def predict_csv(model_dir, input_path, output_path):
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Prediction output must not overwrite its input")
    artifact_files = [Path(model_dir) / name for name in
                      ["model.joblib", "model.json", "baseline.json", "model_manifest.json", "verification.json"]]
    if output_path.resolve() in [p.resolve() for p in artifact_files]:
        raise ValueError("Prediction output must not overwrite a frozen artifact file")
    model, manifest = load_artifact(model_dir)
    frame = read_inference_csv(input_path)
    result = predict_frame(frame, model, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=".predict-", suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        result.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = predict_csv(args.model_dir, args.input, args.output)
    print(f"Saved {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
