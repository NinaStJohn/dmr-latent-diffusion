import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.ndimage import gaussian_filter1d
import porespy as ps
import torch
import torch.nn as nn
from torchvision import models, transforms
import sys
sys.path.append('.')
import material_metrics as mm

def load_images_from_folder(folder, subfolder_names):
    images_dict = {name: [] for name in subfolder_names}
    for subfolder in subfolder_names:
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith((".png", ".jpeg", ".jpg", ".tif", ".tiff")):
                    img = Image.open(os.path.join(subfolder_path, filename)).convert('RGB')
                    img = img.resize((2048, 2048))
                    images_dict[subfolder].append(img)
    return images_dict

# def load_images_from_conditional_test(folder, eval_name):
#     images_dict = {name: [] for name in ['eval_' + str(epoch*2500) for epoch in range(4, 27)]}
#     eval_path = os.path.join(folder, eval_name)
#     if os.path.isdir(eval_path):
#         for epoch in range(4, 27):
#             epoch_multiplied = epoch * 2500
#             epoch_path = os.path.join(eval_path, f"eval_{epoch_multiplied}", "img")
#             if os.path.isdir(epoch_path):
#                 for filename in os.listdir(epoch_path):
#                     if filename.lower().endswith((".png", ".jpeg", ".jpg", ".tif", ".tiff")):
#                         img = Image.open(os.path.join(epoch_path, filename)).convert('RGB')
#                         images_dict[f"eval_{epoch_multiplied}"].append(img)
#     return images_dict


def load_images_from_conditional_test(folder):
    images = []
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.lower().endswith((".png", ".jpeg", ".jpg", ".tif", ".tiff")):
                img = Image.open(os.path.join(folder, filename)).convert('RGB')
                img = img.resize((2048, 2048))
                images.append(img)
    return images

def preprocess_images(images):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images_preprocessed = [preprocess(img).unsqueeze(0) for img in images]
    return torch.cat(images_preprocessed, dim=0)

def get_activations(images, model, device):
    images_preprocessed = preprocess_images(images).to(device)
    with torch.no_grad():
        activations = model(images_preprocessed).cpu().numpy()
    return activations

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def plot_metric(train_data, test_data, metric_name, metrics_folder, train_folder):
    print("Plotting", metric_name)
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    first_label = 'Train' if train_folder else 'Original'
    second_label = 'Test' if train_folder else 'Reconstructed'

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
    
    # Compute mean probabilities
    train_mean_prob = np.array([np.mean([np.interp(d, data.distance, data.probability) for data in train_data]) for d in distance_range])
    test_mean_prob = np.array([np.mean([np.interp(d, data.distance, data.probability) for data in test_data]) for d in distance_range])
    
    # Compute min and max probabilities for the range
    train_min_prob = np.array([min([np.interp(d, data.distance, data.probability) for data in train_data]) for d in distance_range])
    train_max_prob = np.array([max([np.interp(d, data.distance, data.probability) for data in train_data]) for d in distance_range])
    
    test_min_prob = np.array([min([np.interp(d, data.distance, data.probability) for data in test_data]) for d in distance_range])
    test_max_prob = np.array([max([np.interp(d, data.distance, data.probability) for data in test_data]) for d in distance_range])

    # Plot the range of values
    ax.fill_between(distance_range, train_min_prob, train_max_prob, color='blue', alpha=0.2, label=first_label)
    ax.fill_between(distance_range, test_min_prob, test_max_prob, color='red', alpha=0.2, label=second_label)

    # Plot the mean values
    ax.plot(distance_range, train_mean_prob, 'b-', label='Mean_' + first_label)
    ax.plot(distance_range, test_mean_prob, 'r-', label='Mean_' + second_label)

    ax.set_xlabel("distance")
    ax.set_ylabel("probability")
    ax.legend()
    fig.suptitle(metric_name)
    fig.savefig(os.path.join(metrics_folder, f"{metric_name}.png"))


def plot_pdf_cdf_bar(data, metric_name, metrics_folder, train_folder, sigma=2):
    print("Plotting", metric_name)
    fig, ax = plt.subplots(1, 2, figsize=[7, 4])
    first_label = 'Train' if train_folder else 'Original'
    second_label = 'Test' if train_folder else 'Reconstructed'

    def compute_mean_and_range(dataset):
        bin_centers = np.concatenate([d.bin_centers for d in dataset])
        pdf = np.concatenate([d.pdf for d in dataset])
        cdf = np.concatenate([d.cdf for d in dataset])
        
        unique_bins = np.unique(bin_centers)
        mean_pdf = np.array([np.mean(pdf[bin_centers == b]) for b in unique_bins])
        mean_cdf = np.array([np.mean(cdf[bin_centers == b]) for b in unique_bins])

        min_pdf = np.array([np.min(pdf[bin_centers == b]) for b in unique_bins])
        max_pdf = np.array([np.max(pdf[bin_centers == b]) for b in unique_bins])

        min_cdf = np.array([np.min(cdf[bin_centers == b]) for b in unique_bins])
        max_cdf = np.array([np.max(cdf[bin_centers == b]) for b in unique_bins])

        # Apply Gaussian smoothing
        mean_pdf = gaussian_filter1d(mean_pdf, sigma=sigma)
        mean_cdf = gaussian_filter1d(mean_cdf, sigma=sigma)
        min_pdf = gaussian_filter1d(min_pdf, sigma=sigma)
        max_pdf = gaussian_filter1d(max_pdf, sigma=sigma)
        min_cdf = gaussian_filter1d(min_cdf, sigma=sigma)
        max_cdf = gaussian_filter1d(max_cdf, sigma=sigma)

        return unique_bins, mean_pdf, min_pdf, max_pdf, mean_cdf, min_cdf, max_cdf

    train_bin_centers, train_mean_pdf, train_min_pdf, train_max_pdf, train_mean_cdf, train_min_cdf, train_max_cdf = compute_mean_and_range(data['train'])
    test_bin_centers, test_mean_pdf, test_min_pdf, test_max_pdf, test_mean_cdf, test_min_cdf, test_max_cdf = compute_mean_and_range(data['test'])

    ax[0].fill_between(train_bin_centers, train_min_pdf, train_max_pdf, color='blue', alpha=0.2, label=f'{first_label}')
    ax[0].fill_between(test_bin_centers, test_min_pdf, test_max_pdf, color='red', alpha=0.2, label=f'{second_label}')
    ax[0].plot(train_bin_centers, train_mean_pdf, 'b-', label=f'Mean_{first_label}')
    ax[0].plot(test_bin_centers, test_mean_pdf, 'r-', label=f'Mean_{second_label}')
    ax[0].set_title("Probability Density Function")

    ax[1].fill_between(train_bin_centers, train_min_cdf, train_max_cdf, color='blue', alpha=0.2, label=f'{first_label}')
    ax[1].fill_between(test_bin_centers, test_min_cdf, test_max_cdf, color='red', alpha=0.2, label=f'{second_label}')
    ax[1].plot(train_bin_centers, train_mean_cdf, 'b-', label=f'Mean_{first_label}')
    ax[1].plot(test_bin_centers, test_mean_cdf, 'r-', label=f'Mean_{second_label}')
    ax[1].set_title("Cumulative Density Function")



def main(train_folder, test_folder, org_image, rec_image, conditional_train_folder, conditional_test_folder, metrics_folder, fid_only):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # Remove the classification head
    model.to(device)
    model.eval()

    if train_folder and test_folder:
        train_images = load_images_from_folder(train_folder, ['0%', '4%', '8%', '12%', '16%', '20%'])
        test_images = load_images_from_folder(test_folder, ['eval_0', 'eval_1', 'eval_2', 'eval_3', 'eval_4', 'eval_5', 'eval_6'])

        if not fid_only:
            os.makedirs(metrics_folder, exist_ok=True)
            metrics = {
                "two_point_correlation": {'train': [], 'test': []},
                "pore_size_distribution": {'train': [], 'test': []},
                "lineal_path_distribution": {'train': [], 'test': []},
                "chord_length_distribution": {'train': [], 'test': []}
            }

            for percentage, train_images_group in train_images.items():
                for eval_name, test_images_group in test_images.items():
                    fid_values = []
                    for epoch in range(1, 21):
                        eval_images = [img for img in test_images_group if f"eval_{epoch}" in img.filename]
                        if len(eval_images) > 0:
                            act_real = get_activations(train_images_group, model, device)
                            act_gen = get_activations(eval_images, model, device)
                            fid = calculate_fid(act_real, act_gen)
                            fid_values.append(fid)
                            # Save FID to file
                            fid_folder = os.path.join(metrics_folder, percentage, str(epoch))
                            os.makedirs(fid_folder, exist_ok=True)
                            with open(os.path.join(fid_folder, 'fid.txt'), 'w') as f:
                                f.write(f"FID: {fid}\n")

                            print(f"FID for {percentage} images vs {eval_name} eval_{epoch}: {fid}")

                    if not fid_only:
                        print("Beginning metric calculations for train images")
                        for i, im in enumerate(train_images_group):
                            print(f"train: {i+1}/{len(train_images_group)}")
                            np_array = np.array(im.convert('L'))
                            metrics["two_point_correlation"]['train'].append(two_point_correlation.two_point_correlation(np_array))
                            metrics["pore_size_distribution"]['train'].append(ps.metrics.pore_size_distribution(np_array))
                            metrics["lineal_path_distribution"]['train'].append(ps.metrics.lineal_path_distribution(np_array))
                            metrics["chord_length_distribution"]['train'].append(ps.metrics.chord_length_distribution(np_array))

                        print("Beginning metric calculations for test images")
                        for i, im in enumerate(eval_images):
                            print(f"test: {i+1}/{len(eval_images)}")
                            np_array = np.array(im.convert('L'))
                            metrics["two_point_correlation"]['test'].append(two_point_correlation.two_point_correlation(np_array))
                            metrics["pore_size_distribution"]['test'].append(ps.metrics.pore_size_distribution(np_array))
                            metrics["lineal_path_distribution"]['test'].append(ps.metrics.lineal_path_distribution(np_array))
                            metrics["chord_length_distribution"]['test'].append(ps.metrics.chord_length_distribution(np_array))

                        plot_metric(metrics["two_point_correlation"]['train'], metrics["two_point_correlation"]['test'], "Two Point Correlation", metrics_folder, train_folder)
                        plot_pdf_cdf_bar(metrics["pore_size_distribution"], "Pore Size Distribution", metrics_folder, train_folder)
                        plot_pdf_cdf_bar(metrics["lineal_path_distribution"], "Lineal Path Distribution", metrics_folder, train_folder)
                        plot_pdf_cdf_bar(metrics["chord_length_distribution"], "Chord Length Distribution", metrics_folder, train_folder)

    elif conditional_train_folder and conditional_test_folder:
        conditional_train_images = load_images_from_folder(conditional_train_folder, ['0%', '4%', '8%', '12%', '16%', '20%'])
        
        if not fid_only:
            os.makedirs(metrics_folder, exist_ok=True)
            metrics = {
                "two_point_correlation_0%": {'train': [], 'test': []},
                "two_point_correlation_4%": {'train': [], 'test': []},
                "two_point_correlation_8%": {'train': [], 'test': []},
                "two_point_correlation_12%": {'train': [], 'test': []},
                "two_point_correlation_16%": {'train': [], 'test': []},
                "two_point_correlation_20%": {'train': [], 'test': []},

                "pore_size_distribution_0%": {'train': [], 'test': []},
                "pore_size_distribution_4%": {'train': [], 'test': []},
                "pore_size_distribution_8%": {'train': [], 'test': []},
                "pore_size_distribution_12%": {'train': [], 'test': []},
                "pore_size_distribution_16%": {'train': [], 'test': []},
                "pore_size_distribution_20%": {'train': [], 'test': []},

                "lineal_path_distribution_0%": {'train': [], 'test': []},
                "lineal_path_distribution_4%": {'train': [], 'test': []},
                "lineal_path_distribution_8%": {'train': [], 'test': []},
                "lineal_path_distribution_12%": {'train': [], 'test': []},
                "lineal_path_distribution_16%": {'train': [], 'test': []},
                "lineal_path_distribution_20%": {'train': [], 'test': []},

                "chord_length_distribution_0%": {'train': [], 'test': []},
                "chord_length_distribution_4%": {'train': [], 'test': []},
                "chord_length_distribution_8%": {'train': [], 'test': []},
                "chord_length_distribution_12%": {'train': [], 'test': []},
                "chord_length_distribution_16%": {'train': [], 'test': []},
                "chord_length_distribution_20%": {'train': [], 'test': []}
            }

            print("Beginning metric calculations for train images")
            for percentage, train_images in conditional_train_images.items():
                for i, im in enumerate(train_images):
                    print(f"train: {i+1}/{len(train_images)}")
                    np_array = np.array(im.convert('L'))
                    metrics["pore_size_distribution_" + percentage]['train'].append(mm.pore_size_distribution(np_array))
                    metrics["lineal_path_distribution_" + percentage]['train'].append(mm.lineal_path_distribution(np_array))
                    np_array = (np_array > 0.65*256).astype(np.uint8) * 255  
                    metrics["two_point_correlation_" + percentage]['train'].append(mm.two_point_correlation(np_array))
                    metrics["chord_length_distribution_" + percentage]['train'].append(mm.chord_length_distribution(np_array))
                
            percentages = list(conditional_train_images.keys())
            train_images_groups = list(conditional_train_images.values())
            eval_names = ['eval_0', 'eval_1', 'eval_2', 'eval_3', 'eval_4', 'eval_5']
            for percentage, train_images_group, eval_name in zip(percentages, train_images_groups, eval_names):
                act_real = get_activations(train_images_group, model, device)
                fid_values = []
                
                for epoch in range(4, 27):
                    epoch_multiplied = 2500 * epoch
                    eval_images = load_images_from_conditional_test(os.path.join(conditional_test_folder, eval_name, f"eval_{epoch_multiplied}", "img"))
                    new_metrics_folder = os.path.join(metrics_folder, percentage, f"epoch_{epoch_multiplied}")
                    if len(eval_images) > 0:
                        act_gen = get_activations(eval_images, model, device)
                        fid = calculate_fid(act_real, act_gen)
                        fid_values.append(fid)
                        os.makedirs(new_metrics_folder, exist_ok=True)
                        with open(os.path.join(metrics_folder, percentage, 'fid.txt'), 'a') as f:
                            f.write(f"EPOCH: {epoch_multiplied} FID: {fid}\n")

                        print(f"FID for {percentage} images vs {eval_name} eval_{epoch_multiplied}: {fid}")

                    print("Beginning metric calculations for test images")
                    for i, im in enumerate(eval_images):
                        print(f"test: {i+1}/{len(eval_images)}")
                        np_array = np.array(im.convert('L'))
                        metrics["pore_size_distribution_" + percentage]['test'].append(mm.metrics.pore_size_distribution(np_array))
                        metrics["lineal_path_distribution_" + percentage]['test'].append(mm.metrics.lineal_path_distribution(np_array))
                        np_array = (np_array > 0.65*256).astype(np.uint8) * 255
                        metrics["two_point_correlation_" + percentage]['test'].append(mm.two_point_correlation(np_array))
                        metrics["chord_length_distribution_" + percentage]['test'].append(mm.metrics.chord_length_distribution(np_array))

                    plot_metric(metrics["two_point_correlation_" + percentage]['train'], metrics["two_point_correlation_" + percentage]['test'], "Two Point Correlation", new_metrics_folder, conditional_train_folder)
                    plot_pdf_cdf_bar(metrics["pore_size_distribution_" + percentage], "Pore Size Distribution", new_metrics_folder, conditional_train_folder)
                    plot_pdf_cdf_bar(metrics["lineal_path_distribution_" + percentage], "Lineal Path Distribution", new_metrics_folder, conditional_train_folder)
                    plot_pdf_cdf_bar(metrics["chord_length_distribution_" + percentage], "Chord Length Distribution", new_metrics_folder, conditional_train_folder)

    else:
        print("No valid folders provided for analysis.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot metrics for GAN generated images.")
    parser.add_argument("--train_folder", type=str, help="Folder containing training images")
    parser.add_argument("--test_folder", type=str, help="Folder containing generated images")
    parser.add_argument("--org_image", type=str, help="Image containing original image")
    parser.add_argument("--rec_image", type=str, help="Reconstructed image from original image")
    parser.add_argument("--conditional_train_folder", type=str, help="Folder containing conditional training images")
    parser.add_argument("--conditional_test_folder", type=str, help="Folder containing conditional test images")
    parser.add_argument("--metrics_folder", type=str, required=True, help="Folder to save metrics plots")
    parser.add_argument("--fid_only", action="store_true")

    args = parser.parse_args()

    if not ((args.train_folder and args.test_folder) or (args.org_image and args.rec_image) or (args.conditional_train_folder and args.conditional_test_folder)):
        parser.error("Either both train_folder and test_folder must be provided, or both org_image and rec_image must be provided, or both conditional_train_folder and conditional_test_folder must be provided.")
    main(args.train_folder, args.test_folder, args.org_image, args.rec_image, args.conditional_train_folder, args.conditional_test_folder, args.metrics_folder, args.fid_only)

