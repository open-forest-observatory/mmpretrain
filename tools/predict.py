import json
import argparse
from pathlib import Path

from mmpretrain.apis import ImageClassificationInferencer


def parse_args():
    parser = argparse.ArgumentParser(help="Generate one classification prediction per image in a folder")
    parser.add_argument("input_folder", type=Path, help="Path to chips to classify")
    parser.add_argument(
        "config_path", type=Path, help="Path to .py config file to use for prediction"
    )
    parser.add_argument(
        "model_path", type=Path, help="Path to .pth model file to use for prediction"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="File path with .json extension to write out predicted class per file",
    )

    args = parser.parse_args()
    return args


def main(input_folder: Path, config_path: Path, model_path: Path, output_path: Path):
    # Listing input files
    input_files = [
        str(f)
        for f in Path(input_folder).rglob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    print(f"Running on {len(input_files)} files")

    # Setting up the model
    inferencer = ImageClassificationInferencer(
        model=config_path, pretrained=model_path, device="cuda"
    )

    # Run inference. This is the slow step.
    results = inferencer(input_files, batch_size=128)

    # Extract predicted classes
    pred_labels = [r["pred_class"] for r in results]

    # Build a dict from filename to predicted class
    results_per_file = {str(k): v for k, v in zip(input_files, pred_labels)}

    # Write out results
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as file_h:
        json.dump(results_per_file, file_h)


if __name__ == "__main__":
    args = parse_args()

    main(**args.__dict__)
