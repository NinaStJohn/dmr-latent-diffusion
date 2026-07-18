import argparse
from pathlib import Path

import pandas as pd
import numpy as np

def read_pdf_pair(class_dir, metric_name):
    #read the real and synthetic pdf values from the two csv files

    real_path = class_dir / f"{metric name}_Real.csv"
    synthetic_path = class_dir / f"{metric name}_Synthetic.csv"

    if not real_path.is_file():
        raise FileNotFoundError(f"Missing real csv: {real_path}")
    
    if not synthetic_path.is_file():
        raise FileNotFoundError(f"Missing synthetic csv: {synthetic_path}")
    
    real_df = pd.read_csv(real_path)
    synthetic_df = pd.read-csv(synthetic_path)

    

def collect_model_errors(metrics_dir: Path, model_name: str) -> pd.DataFrame:
    strain_classes = [
        "class_0",
        "class_1",
        "class_2",
        "class_3",
        "class_4",
        "class_5",
    ]

    rows = []

    for strain_class in strain_classes
        class_dir = metrics_dir / strain_classes

        if not class_dir.is_dir():
            print(f"{class_dir} directory not found")
            continue
        
        print(f"Reading metrics from: {class_dir}")

        #for each strain class, read each csv and compute normalized error
        # cld, lpf, and pore size are all separated into two csv's for real and synthetic
        # 2pc has one csv for both real and synthetic

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare normalized metric errors across strain classes "
            "for one or two generative models."
        )
    )

    parser.add_argument(
        "--model_one_dir",
        type=Path,
        required=True,
        help="Directory containing the metrics for model one.",
    )

    parser.add_argument(
        "--model_two_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing the metrics for model two. "
            "If omitted, only model one is plotted."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help=("Directory where result CSV files and bar charts will be saved. "
              "Defaults to the current working directory."
        ),
    )

    parser.add_argument(
        "--model_one_name",
        type=str,
        default="Model 1",
        help="Label used for model one in plots and output files.",
    )

    parser.add_argument(
        "--model_two_name",
        type=str,
        default="Model 2",
        help="Label used for model two in plots and output files.",
    )

    args = parser.parse_args()

    if not args.model_one_dir.is_dir():
        parser.error(
            f"Model one metrics directory does not exist: "
            f"{args.model_one_dir}"
        )

    if args.model_two_dir is not None and not args.model_two_dir.is_dir():
        parser.error(
            f"Model two metrics directory does not exist: "
            f"{args.model_two_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_results = []

    model_one_results = collect_model_errors(
        metrics_dir=args.model_one_dir,
        model_name=args.model_one_name,
    )
    model_results.append(model_one_results)

    if args.model_two_dir is not None:
        model_two_results = collect_model_errors(
            metrics_dir=args.model_two_dir,
            model_name=args.model_two_name,
        )
        model_results.append(model_two_results)

    results = pd.concat(model_results, ignore_index=True)

    output_csv = args.output_dir / "normalized_errors.csv"
    results.to_csv(output_csv, index=False)

    plot_normalized_errors(
        results=results,
        output_dir=args.output_dir,
    )

    print(f"Saved normalized errors to: {output_csv}")
    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()