import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_pdf_pair(class_dir, metric_name):
    #read the real and synthetic pdf values from the two csv files

    real_path = class_dir / f"{metric_name}_Real.csv"
    synthetic_path = class_dir / f"{metric_name}_Synthetic.csv"

    if not real_path.is_file():
        raise FileNotFoundError(f"Missing real csv: {real_path}")
    
    if not synthetic_path.is_file():
        raise FileNotFoundError(f"Missing synthetic csv: {synthetic_path}")
    
    real_df = pd.read_csv(real_path)
    synthetic_df = pd.read_csv(synthetic_path)

    real_required_columns = {"bin_center", "mean_pdf"}
    synthetic_required_columns = {"bin_center", "mean_pdf"}

    if not real_required_columns.issubset(real_df.columns):
        raise ValueError(f"{real_path} must contain: {real_required_columns}")
    if not synthetic_required_columns.issubset(synthetic_df.columns):
        raise ValueError(f"{synthetic_path_path} must contain: {synthetic_required_columns_required_columns}")

    real_df = real_df[["bin_center", "mean_pdf"]].dropna()
    synthetic_df = synthetic_df[["bin_center", "mean_pdf"]].dropna()

    real_df = real_df.sort_values("bin_center")
    synthetic_df = synthetic_df.sort_values("bin_center")

    #use every x-value existing in either csv
    all_x = np.union1d(
        real_df["bin_center"].to_numpy(dtype=float),
        synthetic_df["bin_center"].to_numpy(dtype=float),
    )

    real_x = real_df["bin_center"].to_numpy(dtype=float)
    real_pdf = real_df["mean_pdf"].to_numpy(dtype=float)

    synthetic_x = synthetic_df["bin_center"].to_numpy(dtype=float)
    synthetic_pdf = synthetic_df["bin_center"].to_numpy(dtype=float)

    # Interpolate inside each curve's range.
    # Assume zero before the first point and after the final point.
    real_aligned = np.interp(
        all_x,
        real_x,
        real_pdf,
        left=0.0,
        right=0.0,
    )

    synthetic_aligned = np.interp(
        all_x,
        synthetic_x,
        synthetic_pdf,
        left=0.0,
        right=0.0,
    )

    return all_x, real_aligned, synthetic_aligned


def read_2pc(class_dir):
    #read both real and synthetic values from one csv for 2pc

    csv_path = class_dir / "Two Point Correlation.csv"

    if not csv_path.is_file():
        raise FileNotFoundErrpr(f"Missing 2PC csv: {csv_path}")
    
    df = pd.read_csv(csv_path)

    require_columns = {
        "distance",
        "Real_mean_prob",
        "Synthetic_mean_prob",
    }

    if not require_columns.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain columns: {required_columns}"
        )

    df = df[["distance", "Real_mean_prob", "Synthetic_mean_prob"]].dropna()

    df = df.sort_values("distance")

    x = df["distance"].to_numpy(dtype=float)
    real_values = df["Real_mean_prob"].to_numpy(dtype=float)
    synthetic_values = df["Synthetic_mean_prob"].to_numpy(dtype=float)

    return x, real_values, synthetic_values


def normalized_error(x, real_values, synth_values):
    #compute the normalized area btwn the real and synth curves

    x = np.asarray(x, dtype=float)
    real_values = np.asarray(real_values, dtype=float)
    synth_values = np.asarray(synth_values, dtype=float)

    if not( len(x) == len(real_values) == len(synth_values)):
        raise ValueError("x, real, synth values must have equal lengths")

    if (len(x) < 2):
        return np.nan
    
    finite_mask = (
        np.isfinite(x) & np.isfinite(real_values) & np.isfinite(synth_values)
    )

    x = x[finite_mask]
    real_values = real_values[finite_mask]
    synth_values = synth_values[finite_mask]

    if (len(x) < 2):
        return np.nan

    sort_indices = np.argsort(x)

    x = x[sort_indices]
    real_values = real_values[sort_indices]
    synth_values = synth_values[sort_indices]

    

    difference_area = np.trapz(np.abs(real_values - synth_values), x,)
    real_area = np.trapz(np.abs(real_values), x,)

    if np.isclose(real_area, 0.0):
        return np.nan

    return difference_area/ real_area


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

    for strain_class in strain_classes:
        class_dir = metrics_dir / strain_class

        if not class_dir.is_dir():
            print(f"{class_dir} directory not found")
            continue
        
        print(f"Reading metrics from: {class_dir}")

        #for each strain class, read each csv and compute normalized error
        # cld, lpf, and pore size are all separated into two csv's for real and synthetic
        # 2pc has one csv for both real and synthetic

        cld_x, cld_real, cld_synth = read_pdf_pair(class_dir, "Chord Length Distribution")
        lpf_x, lpf_real, lpf_synth = read_pdf_pair(class_dir, "Lineal Path Distribution")
        psd_x, psd_real, psd_synth = read_pdf_pair(class_dir, "Pore Size Distribution")
        tpc_x, tpc_real, tpc_synth = read_2pc(class_dir)

        cld_err = normalized_error(cld_x, cld_real, cld_synth)
        lpf_err = normalized_error(lpf_x, lpf_real, lpf_synth)
        psd_err = normalized_error(psd_x, psd_real, psd_synth)
        tpc_err = normalized_error(tpc_x, tpc_real, tpc_synth)

        rows.append({
            "model": model_name,
            "strain_class": strain_class,
            "cld_err": cld_err, 
            "lpf_err": lpf_err,
            "psd_err": psd_err, 
            "tpc_err": tpc_err})


    return pd.DataFrame(rows)


def plot_normalized_errors(results, output_dir):
    #plot one bar graph of the normalized error at each strain level for each metric
    #if a second model's data is provided, its errors will also be graphed at each strain class for comparison

    metric_columns = {
        "cld_err": "Chord Length Distribution",
        "lpf_err": "Lineal Path Distribution",
        "psd_err": "Pore Size Distribution",
        "tpc_err": "Two-Point Correlation",
    }

    strain_order = [
        "class_0",
        "class_1",
        "class_2",
        "class_3",
        "class_4",
        "class_5",
    ]

    strain_labels = [
        "0%", 
        "4%",
        "8%",
        "12%",
        "16%",
        "20%",
    ]

    models = results["model"].unique()
    x = np.arange(len(strain_order))

    bar_width = 0.8 / len(models)

    for metric_column, metric_title in metric_columns.items():
        fig, ax = plt.subplots(figsize=(9,6))

        for model_index, model_name in enumerate(models):
            model_results = results[results["model"] == model_name].copy()

            model_results["strain_class"] = pd.Categorical(model_results["strain_class"], categories=strain_order, ordered=True)
            
            model_resulsts = model_results.sort_values("strain_class")

            errors = []

            for strain_class in strain_order:
                strain_row = model_results[model_results["strain_class"] == strain_class]
                if strain_row.empty:
                    errors.append(np.nan)
                else:
                    errors.append(
                        strain_row[metric_column].iloc[0]
                    )
            
            offset = (model_index - (len(models) - 1) / 2) * bar_width

            ax.bar(x + offset,
                    errors,
                    width=bar_width,
                    label=model_name,)
            
            ax.set_xticks(x)
            ax.set_xticklabels(strain_labels)

            ax.set_xlabel("Strain")
            ax.set_ylabel("Normalized Error")
            ax.set_title(f"{metric_title}: Normalized Error by Strain")

            ax.grid(axis="y", linestyle="--", alpha=0.4,)

            if len(models) > 1:
                ax.legend()

            fig.tight_layout()

            filename = (metric_title.lower().replace(" ", "_"))
            output_path = (output_dir / f"{filename}_normalized_error.png")

            fig.savefig(output_path, dpi=300, bbox_inches="tight",)
            plt.close(fig)

            print(f"Saved plot: {output_path}")
            

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