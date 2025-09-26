import numpy as np
from scipy import fft as sp_ft
from porespy.tools import Results

def porosity(im):
    im = np.array(im, dtype=np.int64)
    Vp = np.sum(im == 1, dtype=np.int64)
    Vs = np.sum(im == 0, dtype=np.int64)
    e = Vp / (Vs + Vp)
    return e

def _radial_profile(autocorr, bins, pf=None, voxel_size=1):
    if len(autocorr.shape) == 2:
        adj = np.reshape(autocorr.shape, [2, 1, 1])
        inds = np.indices(autocorr.shape) - np.round(adj / 2)
        dt = np.sqrt(inds[0]**2 + inds[1]**2)
    elif len(autocorr.shape) == 3:
        adj = np.reshape(autocorr.shape, [3, 1, 1, 1])
        inds = np.indices(autocorr.shape) - np.round(adj / 2)
        dt = np.sqrt(inds[0]**2 + inds[1]**2 + inds[2]**2)
    else:
        raise Exception('Image dimensions must be 2 or 3')

    if np.max(bins) > np.max(dt):
        raise Exception('Bins specified distances exceeding maximum radial distance for image size.')

    bin_size = bins[1:] - bins[:-1]
    radial_sum = _get_radial_sum(dt, bins, bin_size, autocorr)
    
    # Normalize the radial sum to get probabilities
    norm_autoc_radial = radial_sum / np.max(autocorr)
    
    # Ensure values are within [0, 1] range
    norm_autoc_radial = np.clip(norm_autoc_radial, 0, 1)
    
    h = [norm_autoc_radial, bins]
    h = _parse_histogram(h, voxel_size=1)
    tpcf = Results()
    tpcf.distance = h.bin_centers * voxel_size
    tpcf.bin_centers = h.bin_centers * voxel_size
    tpcf.bin_edges = h.bin_edges * voxel_size
    tpcf.bin_widths = h.bin_widths * voxel_size
    tpcf.probability = norm_autoc_radial
    tpcf.probability_scaled = norm_autoc_radial * pf
    tpcf.pdf = h.pdf * pf
    tpcf.relfreq = h.relfreq
    return tpcf


def _get_radial_sum(dt, bins, bin_size, autocorr):
    radial_sum = np.zeros_like(bins[:-1], dtype=np.float64)  # Use float64 for large sums
    for i, r in enumerate(bins[:-1]):
        mask = (dt <= r) & (dt > (r - bin_size[i]))  # Logical AND for boolean mask
        if np.any(mask):  # Ensure there are non-zero elements in the mask
            radial_sum[i] = np.sum(autocorr[mask], dtype=np.float64) / np.sum(mask, dtype=np.float64)
    return radial_sum


def two_point_correlation(im, voxel_size=1, bins=100):
    cpus = 1
    pf = porosity(im)
    if isinstance(bins, int):
        r_max = (np.ceil(np.min(np.shape(im))) / 2).astype(int)
        bin_size = int(np.ceil(r_max / bins))
        bins = np.arange(0, r_max + bin_size, bin_size)

    with sp_ft.set_workers(cpus):
        F = sp_ft.ifftshift(sp_ft.rfftn(sp_ft.fftshift(im)))
        P = np.absolute(F**2)
        autoc = np.absolute(sp_ft.ifftshift(sp_ft.irfftn(sp_ft.fftshift(P))))

    tpcf = _radial_profile(autoc, bins, pf=pf, voxel_size=voxel_size)
    return tpcf


def _parse_histogram(h, voxel_size=1, density=True):
    delta_x = h[1]
    P = h[0]
    bin_widths = delta_x[1:] - delta_x[:-1]
    temp = P * (bin_widths)
    C = np.cumsum(temp[-1::-1])[-1::-1]
    S = P * (bin_widths)
    if not density:
        P /= np.max(P)
        temp_sum = np.sum(P * bin_widths)
        C /= temp_sum
        S /= temp_sum

    bin_edges = delta_x * voxel_size
    bin_widths = (bin_widths) * voxel_size
    bin_centers = ((delta_x[1:] + delta_x[:-1]) / 2) * voxel_size
    hist = Results()
    hist.pdf = P
    hist.cdf = C
    hist.relfreq = S
    hist.bin_centers = bin_centers
    hist.bin_edges = bin_edges
    hist.bin_widths = bin_widths
    return hist
