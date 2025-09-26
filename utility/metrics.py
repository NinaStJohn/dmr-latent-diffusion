import os
import argparse
import numpy as np
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
import two_point_correlation
import chord_length_distribution
import lineal_path_distribution
import pore_size_distribution

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
    
    train_mean_prob = np.array([np.mean([np.interp(d, data.distance, data.probability) for data in train_data]) for d in distance_range])
    test_mean_prob = np.array([np.mean([np.interp(d, data.distance, data.probability) for data in test_data]) for d in distance_range])
    
    train_min_prob = np.array([min([np.interp(d, data.distance, data.probability) for data in train_data]) for d in distance_range])
    train_max_prob = np.array([max([np.interp(d, data.distance, data.probability) for data in train_data]) for d in distance_range])
    
    test_min_prob = np.array([min([np.interp(d, data.distance, data.probability) for data in test_data]) for d in distance_range])
    test_max_prob = np.array([max([np.interp(d, data.distance, data.probability) for data in test_data]) for d in distance_range])

    ax.fill_between(distance_range, train_min_prob, train_max_prob, color='blue', alpha=0.2, label=first_label)
    ax.fill_between(distance_range, test_min_prob, test_max_prob, color='red', alpha=0.2, label=second_label)

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

    fig.suptitle(metric_name)
    fig.savefig(os.path.join(metrics_folder, f"{metric_name}.png"))


def main(train_folder, test_folder, org_image, rec_image, metrics_folder, fid_only):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train_folder:
        train_images = load_images(train_folder, "test")
        test_images = load_images(test_folder, "test")[:3]

        print("Images loaded")

        model = models.inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity() 
        model.to(device)
        model.eval()

        print("Getting activations")
        act_real = get_activations(train_images, model, device)
        act_gen = get_activations(test_images, model, device)

        fid = calculate_fid(act_real, act_gen)
        print(f"FID: {fid}")

    if not fid_only:
        os.makedirs(metrics_folder, exist_ok=True)
        metrics = {
            "two_point_correlation": {'train': [], 'test': []},
            "pore_size_distribution": {'train': [], 'test': []},
            "lineal_path_distribution": {'train': [], 'test': []},
            "chord_length_distribution": {'train': [], 'test': []}
        }

        if train_folder:
            print("Beginning metric calculations for training images")
            for i, im in enumerate(train_images):
                print("train: {}/{}".format(i + 1, len(train_images)))
                np_array = np.array(im.convert('L'))
                metrics["pore_size_distribution"]['train'].append(ps.metrics.pore_size_distribution(np_array))
                metrics["lineal_path_distribution"]['train'].append(ps.metrics.lineal_path_distribution(np_array))
                np_array = (np_array > 0.65*255).astype(np.uint8) * 255
                metrics["two_point_correlation"]['train'].append(two_point_correlation.two_point_correlation(np_array))
                metrics["chord_length_distribution"]['train'].append(ps.metrics.chord_length_distribution(np_array))


            print("Beginning metric calculations for test images")
            for i, im in enumerate(test_images):
                print("test: {}/{}".format(i + 1, len(test_images)))
                np_array = np.array(im.convert('L'))
                metrics["pore_size_distribution"]['test'].append(ps.metrics.pore_size_distribution(np_array))
                
                np_array = (np_array > 0.65*255).astype(np.uint8) * 255
                
                metrics["lineal_path_distribution"]['test'].append(ps.metrics.lineal_path_distribution(np_array))
                metrics["two_point_correlation"]['test'].append(two_point_correlation.two_point_correlation(np_array))
                metrics["chord_length_distribution"]['test'].append(ps.metrics.chord_length_distribution(np_array))
        else:
            print("Beginning metric calculations for original image")
            im = Image.open(org_image)
            np_array = np.array(im.convert('L'))
            np_array = (np_array > 0.65*255).astype(np.uint8)
            metrics["pore_size_distribution"]['train'].append(pore_size_distribution.pore_size_distribution(np_array))
            metrics["lineal_path_distribution"]['train'].append(lineal_path_distribution.lineal_path_distribution(np_array))
            # np.savetxt('output.txt', np_array, fmt='%d', delimiter=' ')
            metrics["two_point_correlation"]['train'].append(two_point_correlation.two_point_correlation(np_array))
            metrics["chord_length_distribution"]['train'].append(chord_length_distribution.chord_length_distribution(np_array))


            print("Beginning metric calculations for reconstructed image")
            im = Image.open(rec_image)
            np_array = np.array(im.convert('L'))
            np_array = (np_array > 0.65*255).astype(np.uint8)  
            metrics["pore_size_distribution"]['test'].append(pore_size_distribution.pore_size_distribution(np_array))
            metrics["lineal_path_distribution"]['test'].append(lineal_path_distribution.lineal_path_distribution(np_array))
            metrics["two_point_correlation"]['test'].append(two_point_correlation.two_point_correlation(np_array))
            metrics["chord_length_distribution"]['test'].append(chord_length_distribution.chord_length_distribution(np_array))
    


        plot_metric(metrics["two_point_correlation"]['train'], metrics["two_point_correlation"]['test'], "Two Point Correlation", metrics_folder, train_folder)
        plot_pdf_cdf_bar(metrics["pore_size_distribution"], "Pore Size Distribution", metrics_folder, train_folder)
        plot_pdf_cdf_bar(metrics["lineal_path_distribution"], "Lineal Path Distribution", metrics_folder, train_folder)
        plot_pdf_cdf_bar(metrics["chord_length_distribution"], "Chord Length Distribution", metrics_folder, train_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot metrics for GAN generated images.")
    parser.add_argument("--train_folder", type=str, help="Folder containing training images")
    parser.add_argument("--test_folder", type=str, help="Folder containing generated images")
    parser.add_argument("--org_image", type=str, help="Image containing original image")
    parser.add_argument("--rec_image", type=str, help="Reconstructed image from original image")
    parser.add_argument("--metrics_folder", type=str, required=True, help="Folder to save metrics plots")
    parser.add_argument("--fid_only", action="store_true")

    args = parser.parse_args()

    if not ((args.train_folder and args.test_folder) or (args.org_image and args.rec_image)):
        parser.error("Either both train_folder and test_folder must be provided, or both org_image and rec_image must be provided.")
    main(args.train_folder, args.test_folder, args.org_image, args.rec_image, args.metrics_folder, args.fid_only)
