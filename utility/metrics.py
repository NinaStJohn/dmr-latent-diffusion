import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow duplicate OpenMP (not ideal)
os.environ["OMP_NUM_THREADS"] = "1"          # optional: avoid oversubscription

import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import porespy as ps
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.append('.')
import skimage


# defining fixed axis limits and ticks to make graph scaling comparable across strain classes, 
AXIS_LIMITS = {
    "Two Point Correlation": {
        "xlim": (-10, 1100),
        "ylim": (1e-3, 1.5),
        "yscale": "log",
    },
    "Chord Length Distribution": {
        "xlim": (-10, 300),
        "ylim": (1e-5, 1.5),
        "yscale": "log",
    },
    "Lineal Path Distribution": {
        "xlim": (-10, 250),
        "ylim": (1e-5, 2e-1),
        "yscale": "log",
    },
}

PDF_TICKS = {
    "Lineal Path Distribution": {
        "xticks": [0, 50, 100, 150, 200, 250],
        "yticks": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
    },
    "Chord Length Distribution": {
        "xticks": [0, 50, 100, 150, 200, 250, 300],
        "yticks": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
    },
    "Two Point Correlation": {
        "xticks": [0, 200, 400, 600, 800, 1000],
        "yticks": [1e-3, 1e-2, 1e-1, 1e0,],
    }
}


def load_images(folder, mode):
    images = []
    if mode == "train":
        for class_name in os.listdir(folder):
            for filename in os.listdir(os.path.join(folder, class_name)):
                if filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".tif")  or filename.endswith(".tiff"):
                    img = Image.open(os.path.join(folder, class_name, filename)).convert('RGB')
                    images.append(img)
    elif mode == "test":
        for filename in os.listdir(folder):
            if filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".tif")  or filename.endswith(".tiff"):
                img = Image.open(os.path.join(folder, filename)).convert('RGB')
                images.append(img)
    return images

#plot a metric with guaranteed same domain as real images: 2pc
def plot_metric(train_data, test_data, metric_name, metrics_folder, train_folder, fixed_axis_scaling=False, model_name=None):
    print("Plotting", metric_name)
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    first_label = 'Real'
    second_label = 'Synthetic'
    #append model name to second label if provided
    second_label = f"{second_label} ({model_name})" if model_name else second_label

    train_distances = []
    train_probabilities = []
    test_distances = []
    test_probabilities = []

    for data in train_data:
        train_distances.append(data.distance)
        train_probabilities.append(data.probability)
        
    for data in test_data:
        test_distances.append(data.distance)
        test_probabilities.append(data.probability)

    train_distances = np.concatenate(train_distances)
    train_probabilities = np.concatenate(train_probabilities)
    test_distances = np.concatenate(test_distances)
    test_probabilities = np.concatenate(test_probabilities)

    distance_range = np.linspace(min(train_distances.min(), test_distances.min()), 
                                 max(train_distances.max(), test_distances.max()), 500)
    
    train_mean_prob = np.array([np.mean([np.interp(d, data.distance, data.probability) for data in train_data]) for d in distance_range])
    test_mean_prob = np.array([np.mean([np.interp(d, data.distance, data.probability) for data in test_data]) for d in distance_range])
    
    train_min_prob = np.array([min([np.interp(d, data.distance, data.probability) for data in train_data]) for d in distance_range])
    train_max_prob = np.array([max([np.interp(d, data.distance, data.probability) for data in train_data]) for d in distance_range])
    
    test_min_prob = np.array([min([np.interp(d, data.distance, data.probability) for data in test_data]) for d in distance_range])
    test_max_prob = np.array([max([np.interp(d, data.distance, data.probability) for data in test_data]) for d in distance_range])

    #save data to csv
    df = pd.DataFrame({
        'distance': distance_range,
        f'{first_label}_mean_prob': train_mean_prob,
        f'{first_label}_min_prob': train_min_prob,
        f'{first_label}_max_prob': train_max_prob,
        f'{second_label}_mean_prob': test_mean_prob,
        f'{second_label}_min_prob': test_min_prob,
        f'{second_label}_max_prob': test_max_prob
    })
    df.to_csv(os.path.join(metrics_folder, f"{metric_name}.csv"), index=False)

    ax.fill_between(distance_range, train_min_prob, train_max_prob, color='blue', alpha=0.2, label=first_label)
    ax.fill_between(distance_range, test_min_prob, test_max_prob, color='red', alpha=0.2, label=second_label)

    ax.plot(distance_range, train_mean_prob, 'b-', label='Mean_' + first_label)
    ax.plot(distance_range, test_mean_prob, 'r-', label='Mean_' + second_label)

    # change y axis to log
    ax.set_yscale("log")

    # fix axis scaling if specified
    if fixed_axis_scaling:
        limits = AXIS_LIMITS.get(metric_name)
        ticks = PDF_TICKS.get(metric_name)
        if limits:
            ax.set_xlim(*limits["xlim"])
            ax.set_ylim(*limits["ylim"])
            ax.set_yscale(limits["yscale"])
        if ticks:
            ax.set_xticks(ticks["xticks"])
            ax.set_yticks(ticks["yticks"])

    ax.set_xlabel("distance")
    ax.set_ylabel("probability")
    ax.legend()
    title = f"{metric_name} ({model_name})" if model_name else metric_name
    fig.suptitle(title)
    fig.savefig(os.path.join(metrics_folder, f"{title.replace(' ', '_')}.png"))
    
#plot a metric with possible differing domains: psd, lpd, cld
def plot_pdf_cdf_bar(data, metric_name, metrics_folder, train_folder, sigma=2, fixed_axis_scaling=False, model_name=None):
    print("Plotting", metric_name)
    fig, ax = plt.subplots(1, 2, figsize=[7, 4])
    first_label = 'Real'
    second_label = 'Synthetic'
    #append model name to second label if provided
    second_label = f"{second_label} ({model_name})" if model_name else second_label

    def compute_mean_and_range(dataset):
        #define common grid spanning full range of all images
        all_bins = np.concatenate([d.bin_centers for d in dataset])
        grid = np.linspace(all_bins.min(), all_bins.max(), 200)

        pdfs = np.array([np.interp(grid,d.bin_centers, d.pdf, left=0, right=0) for d in dataset])
        cdfs = np.array([np.interp(grid, d.bin_centers, d.cdf, left=1, right = 0) for d in dataset])

        mean_pdf, min_pdf, max_pdf = pdfs.mean(0), pdfs.min(0), pdfs.max(0)
        mean_cdf, min_cdf, max_cdf = cdfs.mean(0), cdfs.min(0), cdfs.max(0)

        # Apply Gaussian smoothing
        mean_pdf = gaussian_filter1d(mean_pdf, sigma=sigma)
        mean_cdf = gaussian_filter1d(mean_cdf, sigma=sigma)
        min_pdf = gaussian_filter1d(min_pdf, sigma=sigma)
        max_pdf = gaussian_filter1d(max_pdf, sigma=sigma)
        min_cdf = gaussian_filter1d(min_cdf, sigma=sigma)
        max_cdf = gaussian_filter1d(max_cdf, sigma=sigma)

        return grid, mean_pdf, min_pdf, max_pdf, mean_cdf, min_cdf, max_cdf

    train_bin_centers, train_mean_pdf, train_min_pdf, train_max_pdf, train_mean_cdf, train_min_cdf, train_max_cdf = compute_mean_and_range(data['train'])
    test_bin_centers, test_mean_pdf, test_min_pdf, test_max_pdf, test_mean_cdf, test_min_cdf, test_max_cdf = compute_mean_and_range(data['test'])

    # Save data to CSV
    df1 = pd.DataFrame({
        'bin_center': train_bin_centers,
        'mean_pdf': train_mean_pdf,
        'min_pdf': train_min_pdf,
        'max_pdf': train_max_pdf,

        'mean_cdf': train_mean_cdf,
        'min_cdf': train_min_cdf,
        'max_cdf': train_max_cdf,
    })
    df2 = pd.DataFrame({
        'bin_center': test_bin_centers,
        'mean_pdf': test_mean_pdf,
        'min_pdf': test_min_pdf,
        'max_pdf': test_max_pdf,

        'mean_cdf': test_mean_cdf,
        'min_cdf': test_min_cdf,
        'max_cdf': test_max_cdf
    })
    df1.to_csv(os.path.join(metrics_folder, f"{metric_name}_Real.csv"), index=False)
    df2.to_csv(os.path.join(metrics_folder, f"{metric_name}_Synthetic.csv"), index=False)

    ax[0].fill_between(train_bin_centers, train_min_pdf, train_max_pdf, color='blue', alpha=0.2, label=f'Range_{first_label}')
    ax[0].fill_between(test_bin_centers, test_min_pdf, test_max_pdf, color='red', alpha=0.2, label=f'Range_{second_label}')
    ax[0].plot(train_bin_centers, train_mean_pdf, 'b-', label=f'Mean_{first_label}')
    ax[0].plot(test_bin_centers, test_mean_pdf, 'r-', label=f'Mean_{second_label}')
    ax[0].set_title("Probability Density Function")

    ax[1].fill_between(train_bin_centers, train_min_cdf, train_max_cdf, color='blue', alpha=0.2, label=f'Range_{first_label}')
    ax[1].fill_between(test_bin_centers, test_min_cdf, test_max_cdf, color='red', alpha=0.2, label=f'Range_{second_label}')
    ax[1].plot(train_bin_centers, train_mean_cdf, 'b-', label=f'Mean_{first_label}')
    ax[1].plot(test_bin_centers, test_mean_cdf, 'r-', label=f'Mean_{second_label}')
    ax[1].set_title("Cumulative Density Function")

    for a in ax:
        a.legend()

    #if "Chord" in metric_name or "Two" in metric_name :
    for a in ax:
        a.set_yscale("log")
    
    if fixed_axis_scaling:
        limits = AXIS_LIMITS.get(metric_name)
        ticks = PDF_TICKS.get(metric_name)
        if limits:
            ax[0].set_xlim(*limits["xlim"])
            ax[0].set_ylim(*limits["ylim"])
            ax[0].set_yscale(limits["yscale"])
        if ticks:
            ax[0].set_xticks(ticks["xticks"])
            ax[0].set_yticks(ticks["yticks"])

    title = f"{metric_name} ({model_name})" if model_name else metric_name
    fig.suptitle(title)
    fig.savefig(os.path.join(metrics_folder, f"{title.replace(' ', '_')}.png"))


def main(train_folder, test_folder, org_image, rec_image, metrics_folder, fixed_axis_scaling=False, model_name=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train_folder:
        real_images = load_images(train_folder, "test")
        synth_images = load_images(test_folder, "test")

        print("Images loaded")

    os.makedirs(metrics_folder, exist_ok=True)
    metrics = {
        "two_point_correlation": {'train': [], 'test': []},
        "pore_size_distribution": {'train': [], 'test': []},
        "lineal_path_distribution": {'train': [], 'test': []},
        "chord_length_distribution": {'train': [], 'test': []}
    }

    if train_folder:
        print("Beginning metric calculations for training images")
        for i, im in enumerate(real_images):
            print(f"train: {i + 1}/{len(real_images)}")
            np_array = np.array(im.convert('L'))
            
            #threshold RGB image into binary image
            thresh = skimage.filters.threshold_otsu(np_array)
            binary = np_array > thresh

            #calc 2pc from binary img
            metrics["two_point_correlation"]['train'].append(ps.metrics.two_point_correlation(binary))

            #Note: since the strain is isotropic, we can sample on both axis and get representations of the same function
            # We can combine these for larger sample size and thus less noisy results
            paths_x = ps.filters.distance_transform_lin(binary, mode="forward", axis=0)
            paths_y = ps.filters.distance_transform_lin(binary, mode="forward", axis=1)
            combined = np.concatenate([paths_x[binary], paths_y[binary]])
            #calculate lineal path distribution
            lpf = ps.metrics.lineal_path_distribution(combined, bins = 40)
            metrics["lineal_path_distribution"]['train'].append(lpf)
            
            # apply chords filter and calc cld
            chords = ps.filters.apply_chords(binary)
            cld = ps.metrics.chord_length_distribution(chords)
            metrics["chord_length_distribution"]['train'].append(cld)

            # apply local thickness filter and calc psd
            mip = ps.filters.local_thickness(binary)        # OR ps.filters.porosimetry(binary)
            psd = ps.metrics.pore_size_distribution(mip, bins = 40)
            metrics["pore_size_distribution"]['train'].append(psd)

        
        print("Beginning metric calculations for synthetic images") # repeat analysis on synthetic images
        for i, im in enumerate(synth_images):
            print(f"test: {i + 1}/{len(synth_images)}")
            np_array = np.array(im.convert('L'))
            thresh = skimage.filters.threshold_otsu(np_array)
            binary = np_array > thresh

            metrics["two_point_correlation"]['test'].append(ps.metrics.two_point_correlation(binary))

            paths_x = ps.filters.distance_transform_lin(binary, mode="forward", axis=0)
            paths_y = ps.filters.distance_transform_lin(binary, mode="forward", axis=1)
            combined = np.concatenate([paths_x[binary], paths_y[binary]])
            lpf = ps.metrics.lineal_path_distribution(combined, bins = 40)
            metrics["lineal_path_distribution"]['test'].append(lpf)
            
            chords = ps.filters.apply_chords(binary)
            cld = ps.metrics.chord_length_distribution(chords)
            metrics["chord_length_distribution"]['test'].append(cld)

            mip = ps.filters.local_thickness(binary)
            psd = ps.metrics.pore_size_distribution(mip, bins = 40)
            metrics["pore_size_distribution"]['test'].append(psd) 

    #branch for single image analysis
    else:
        print("Beginning metric calculations for original image")
        im = Image.open(org_image)
        np_array = np.array(im.convert('L'))
        thresh = skimage.filters.threshold_otsu(np_array)

        np_array = (np_array > thresh).astype(np.uint8)
        metrics["pore_size_distribution"]['train'].append(pore_size_distribution.pore_size_distribution(np_array))
        metrics["lineal_path_distribution"]['train'].append(lineal_path_distribution.lineal_path_distribution(np_array))
        metrics["two_point_correlation"]['train'].append(two_point_correlation.two_point_correlation(np_array))
        metrics["chord_length_distribution"]['train'].append(chord_length_distribution.chord_length_distribution(np_array))

        print("Beginning metric calculations for reconstructed image")
        im = Image.open(rec_image)
        np_array = np.array(im.convert('L'))
        thresh = skimage.filters.threshold_otsu(np_array)

        np_array = (np_array > thresh).astype(np.uint8)
        metrics["pore_size_distribution"]['test'].append(pore_size_distribution.pore_size_distribution(np_array))
        metrics["lineal_path_distribution"]['test'].append(lineal_path_distribution.lineal_path_distribution(np_array))
        metrics["two_point_correlation"]['test'].append(two_point_correlation.two_point_correlation(np_array))
        metrics["chord_length_distribution"]['test'].append(chord_length_distribution.chord_length_distribution(np_array))


    #plot metrics using matplot
    plot_metric(metrics["two_point_correlation"]['train'], metrics["two_point_correlation"]['test'], "Two Point Correlation", metrics_folder, train_folder, fixed_axis_scaling=fixed_axis_scaling, model_name=model_name)
    plot_pdf_cdf_bar(metrics["pore_size_distribution"], "Pore Size Distribution", metrics_folder, train_folder, fixed_axis_scaling=fixed_axis_scaling, model_name=model_name)
    plot_pdf_cdf_bar(metrics["lineal_path_distribution"], "Lineal Path Distribution", metrics_folder, train_folder, fixed_axis_scaling=fixed_axis_scaling, model_name=model_name)
    plot_pdf_cdf_bar(metrics["chord_length_distribution"], "Chord Length Distribution", metrics_folder, train_folder, fixed_axis_scaling=fixed_axis_scaling, model_name=model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot metrics for GAN generated images.")
    parser.add_argument("--train_folder", type=str, help="Folder containing training images")
    parser.add_argument("--test_folder", type=str, help="Folder containing generated images")
    parser.add_argument("--org_image", type=str, help="Image containing original image")
    parser.add_argument("--rec_image", type=str, help="Reconstructed image from original image")
    parser.add_argument("--metrics_folder", type=str, required=True, help="Folder to save metrics plots")
    parser.add_argument("--fixed_axis_scaling", action="store_true", help="Plot all graphs with same axis scaling (no autoscaling)")
    parser.add_argument("--model_name", type=str, help="Model name for graph titles")

    args = parser.parse_args()

    if not ((args.train_folder and args.test_folder) or (args.org_image and args.rec_image)):
        parser.error("Either both train_folder and test_folder must be provided, or both org_image and rec_image must be provided.")
    main(args.train_folder, args.test_folder, args.org_image, args.rec_image, args.metrics_folder, args.fid_only, args.fixed_axis_scaling)
