#!/usr/bin/env python3

"""
Bootstrap errors of reconstructions from compressed sensing, sampling rows.

Copyright (c) Facebook, Inc. and its affiliates.

This computes thrice the average of bootstrapped errors in reconstruction
from compressed sensing, providing a bootstrap estimate of the error.

Functions
---------
bootstrap
    Plots thrice the average of bootstrapped errors in reconstruction.
testbootstrap
    Tests bootstrap.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import argparse
import logging
import math

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import torch
import skimage.filters

import ctorch
import admm


def bootstrap(filein, fileout, subsampling_factor, mask, low, recon,
              recon_args, n_resamps=None, sdn=None, viz=None):
    """
    plots thrice the average of bootstrapped errors in reconstruction

    Plots and saves thrice the average of the differences between
    a reconstruction of the original image in filein and n_resamps bootstrap
    reconstructions via recon applied to the k-space subsamplings specified
    by mask and by other masks generated similarly (retaining each row of
    the image with probability given by subsampling_factor, then adding all
    frequencies between -low and low), corrupting the k-space values with
    independent and identically distributed centered complex Gaussian noise
    whose standard deviation is sdn*sqrt(2) (sdn=0 if not provided explicitly).
    Setting viz to be True yields colorized visualizations, too, including
    the error estimates overlaid over the reconstruction, the error estimates
    blurred overlaid over the reconstruction, the error estimates blurred,
    the error estimates subtracted from the reconstruction, the error estimates
    saturating the reconstruction in hue-saturation-value (HSV) color space,
    and the error estimates interpolating the reconstruction in HSV space.

    The calling sequence of recon must be  (m, n, f, mask_th, **recon_args),
    where filein contains an m x n image, f is the image in k-space subsampled
    to the mask, mask_th = torch.from_numpy(mask.astype(np.unit8)).cuda(), and
    **recon_args is the unpacking of recon_args. The function recon must return
    a torch.Tensor (the reconstruction) and a float (the corresponding loss).

    _N.B._: mask[-low+1], mask[-low+2], ..., mask[low-1] must be True.

    Parameters
    ----------
    filein : str
        path to the file containing the image to be processed (the path may be
        relative or absolute)
    fileout : str
        path to the file to which the plots will be saved (the path may be
        relative or absolute)
    subsampling_factor : float
        probability of retaining a row in the subsampling masks
    mask : ndarray of bool
        indicators of whether to include (True) or exclude (False)
        the corresponding rows in k-space of the image from filein
    low : int
        bandwidth of low frequencies included in mask (between -low to low)
    recon : function
        returns the reconstructed image
    recon_args : dict
        keyword arguments for recon
    n_resamps : int, optional
        number of bootstrap resampled reconstructions (defaults to 100)
    sdn : float, optional
        standard deviation of the noise to add (defaults to 0)
    viz : bool, optional
        indicator of whether to generate colorized visualizations
        (defaults to False)

    Returns
    -------
    float
        loss for the reconstruction using the original mask
    list of float
        losses for the reconstructions using other, randomly generated masks
    float
        square root of the sum of the square of the estimated errors
    float
        square root of the sum of the square of the estimated errors blurred
    """
    # Set default parameters.
    if n_resamps is None:
        n_resamps = 100
    if sdn is None:
        sdn = 0
    if viz is None:
        viz = False
    # Check that the mask includes all low frequencies.
    for k in range(low):
        assert mask[k]
        assert mask[-k]
    # Read the image from disk.
    with Image.open(filein) as img:
        f_orig = np.array(img).astype(np.float64) / 255.
    m = f_orig.shape[0]
    n = f_orig.shape[1]
    # Fourier transform the image.
    ff_orig = np.fft.fft2(f_orig) / np.sqrt(m * n)
    # Add noise.
    ff_noisy = ff_orig.copy()
    ff_noisy += sdn * (np.random.randn(m, n) + 1j * np.random.randn(m, n))
    # Subsample the noisy Fourier transform of the original image.
    f = ctorch.from_numpy(ff_noisy[mask]).cuda()
    logging.info('computing bootstrap resamplings -- all {}'.format(n_resamps))
    # Perform the reconstruction using the mask.
    mask_th = torch.from_numpy(mask.astype(np.uint8)).cuda()
    reconf, lossf = recon(m, n, f, mask_th, **recon_args)
    reconf = reconf.cpu().numpy()
    # Fourier transform the reconstruction.
    freconf = np.fft.fft2(reconf) / np.sqrt(m * n)
    # Perform the reconstruction resampling new masks and samples in k-space.
    recons = np.ndarray((n_resamps, m, n))
    loss = []
    for k in range(n_resamps):
        # Select which frequencies to retain.
        maski = set(np.floor(
            m * np.random.uniform(size=round(m * subsampling_factor))))
        mask1 = np.asarray([False] * m, dtype=bool)
        for i in maski:
            mask1[int(i)] = True
        # Make the optimization well-posed by including the zero frequency.
        mask1[0] = True
        # Include all low frequencies.
        for j in range(low):
            mask1[j] = True
            mask1[-j] = True
        # Subsample the Fourier transform of the reconstruction.
        f1 = ctorch.from_numpy(freconf[mask1]).cuda()
        # Reconstruct the image from the subsampled data.
        mask1_th = torch.from_numpy(mask1.astype(np.uint8)).cuda()
        recon1, loss1 = recon(m, n, f1, mask1_th, **recon_args)
        recon1 = recon1.cpu().numpy()
        # Record the results.
        recons[k, :, :] = recon1
        loss.append(loss1)
    # Calculate the sum of the bootstrap differences.
    sumboo = np.sum(recons - reconf, axis=0)
    scaled = sumboo * 3 / n_resamps
    # Blur the error estimates.
    sigma = 1
    blurred = skimage.filters.gaussian(scaled, sigma=sigma)
    rsse_estimated = np.linalg.norm(scaled, ord='fro')
    rsse_blurred = np.linalg.norm(blurred, ord='fro')

    # Plot errors.
    # Remove the ticks and spines on the axes.
    matplotlib.rcParams['xtick.top'] = False
    matplotlib.rcParams['xtick.bottom'] = False
    matplotlib.rcParams['ytick.left'] = False
    matplotlib.rcParams['ytick.right'] = False
    matplotlib.rcParams['xtick.labeltop'] = False
    matplotlib.rcParams['xtick.labelbottom'] = False
    matplotlib.rcParams['ytick.labelleft'] = False
    matplotlib.rcParams['ytick.labelright'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['axes.spines.bottom'] = False
    matplotlib.rcParams['axes.spines.left'] = False
    matplotlib.rcParams['axes.spines.right'] = False
    # Configure the colormaps.
    kwargs01 = dict(cmap='gray',
                    norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    kwargs11 = dict(cmap='gray',
                    norm=matplotlib.colors.Normalize(vmin=-1, vmax=1))
    # Separate the suffix (filetype) from the rest of the filename.
    suffix = '.' + fileout.split('.')[-1]
    rest = fileout[:-len(suffix)]
    assert fileout == rest + suffix
    # Plot the original.
    plt.figure(figsize=(5.5, 5.5))
    plt.title('Original')
    plt.imshow(np.clip(f_orig, 0, 1), **kwargs01)
    plt.savefig(rest + '_original' + suffix, bbox_inches='tight')
    # Plot the reconstruction from the original mask provided.
    plt.figure(figsize=(5.5, 5.5))
    plt.title('Reconstruction')
    plt.imshow(np.clip(reconf, 0, 1), **kwargs01)
    plt.savefig(rest + '_recon' + suffix, bbox_inches='tight')
    # Plot the difference from the original.
    plt.figure(figsize=(5.5, 5.5))
    plt.title('Error of Reconstruction')
    plt.imshow(np.clip(reconf - f_orig, -1, 1), **kwargs11)
    plt.savefig(rest + '_error' + suffix, bbox_inches='tight')
    # Plot thrice the average of the bootstrap differences.
    plt.figure(figsize=(5.5, 5.5))
    plt.title('Bootstrap')
    plt.imshow(np.clip(scaled, -1, 1), **kwargs11)
    plt.savefig(rest + '_bootstrap' + suffix, bbox_inches='tight')

    if viz:
        # Plot the reconstruction minus the bootstrap difference.
        plt.figure(figsize=(5.5, 5.5))
        plt.title('Reconstruction \u2013 Bootstrap')
        plt.imshow(np.clip(reconf - scaled, 0, 1), **kwargs01)
        plt.savefig(rest + '_corrected' + suffix, bbox_inches='tight')
        # Overlay the error estimates on the reconstruction.
        plt.figure(figsize=(5.5, 5.5))
        threshold = np.abs(scaled).flatten()
        threshold = np.sort(threshold)
        maxthresh = threshold[-1]
        threshold = threshold[round(0.98 * threshold.size)]
        hue = 2. / 3 + (scaled / maxthresh) / 4 * 2 / 3
        saturation = np.abs(scaled) > threshold
        value = reconf * (1 - saturation) + saturation
        hsv = np.dstack((hue, saturation, value))
        rgb = hsv_to_rgb(hsv)
        plt.title('Errors Over a Threshold Overlaid')
        plt.imshow(np.clip(rgb, 0, 1))
        plt.savefig(rest + '_overlaid' + suffix, bbox_inches='tight')
        # Overlay the blurred error estimates on the reconstruction.
        plt.figure(figsize=(5.5, 5.5))
        threshold = np.abs(blurred).flatten()
        threshold = np.sort(threshold)
        maxthresh = threshold[-1]
        threshold = threshold[round(0.98 * threshold.size)]
        hue = 2. / 3 + (blurred / maxthresh) / 4 * 2 / 3
        saturation = np.abs(blurred) > threshold
        value = reconf * (1 - saturation) + saturation
        hsv = np.dstack((hue, saturation, value))
        rgb = hsv_to_rgb(hsv)
        plt.title('Blurred Errors Over a Threshold Overlaid')
        plt.imshow(np.clip(rgb, 0, 1))
        plt.savefig(rest + '_blurred_overlaid' + suffix, bbox_inches='tight')
        # Plot a bootstrap-saturated reconstruction.
        plt.figure(figsize=(5.5, 5.5))
        hue = (1 - np.sign(scaled)) / 4 * 2 / 3
        saturation = np.abs(scaled)
        saturation = saturation / np.max(saturation)
        value = np.clip(reconf, 0, 1)
        hsv = np.dstack((hue, saturation, value))
        rgb = hsv_to_rgb(hsv)
        plt.title('Bootstrap-Saturated Reconstruction')
        plt.imshow(np.clip(rgb, 0, 1))
        plt.savefig(rest + '_saturated' + suffix, bbox_inches='tight')
        # Plot a bootstrap-interpolated reconstruction.
        plt.figure(figsize=(5.5, 5.5))
        hue = 7. / 12 + np.sign(scaled) * 3 / 12
        saturation = np.abs(scaled)
        saturation = saturation / np.max(saturation)
        value = np.clip(reconf, 0, 1)
        hsv = np.dstack((hue, saturation, value))
        rgb = hsv_to_rgb(hsv)
        plt.title('Bootstrap-Interpolated Reconstruction')
        plt.imshow(np.clip(rgb, 0, 1))
        plt.savefig(rest + '_interpolated' + suffix, bbox_inches='tight')
        # Plot the blurred bootstrap.
        plt.figure(figsize=(5.5, 5.5))
        plt.title('Blurred Bootstrap')
        plt.imshow(np.clip(blurred, -1, 1), **kwargs11)
        plt.savefig(rest + '_blurred' + suffix, bbox_inches='tight')

    return lossf, loss, rsse_estimated, rsse_blurred


def testbootstrap(filein, fileout, subsampling_factor, recon, recon_args,
                  n_resamps=None, sdn=None, viz=None):
    """
    tests bootstrap

    Runs bootstrap (and prints the losses) with filein, fileout,
    subsampling_factor, recon, recon_args, n_resamps, sdn, and viz, creating
    a random mask that retains each row with probability subsampling_factor,
    supplemented by all rows between -sqrt(2m) and sqrt(2m), where filein
    contains an image with m rows.

    The calling sequence of recon must be  (m, n, f, mask_th, **recon_args),
    where filein contains an m x n image, f is the image in k-space subsampled
    to the mask, mask_th = torch.from_numpy(mask.astype(np.unit8)).cuda(), and
    **recon_args is the unpacking of recon_args. The function recon must return
    a torch.Tensor (the reconstruction) and a float (the corresponding loss).

    Parameters
    ----------
    filein : str
        path to the file containing the image to be processed (the path may be
        relative or absolute)
    fileout : str
        path to the file to which the plots will be saved (the path may be
        relative or absolute)
    subsampling_factor : float
        probability of retaining a row in the subsampling masks
    recon : function
        returns the reconstructed image
    recon_args : dict
        keyword arguments for recon
    n_resamps : int, optional
        number of bootstrap resampled reconstructions (defaults to 100
        in bootstrap)
    sdn : float, optional
        standard deviation of the noise to add (defaults to 0 in bootstrap)
    viz : bool, optional
        indicator of whether to generate colorized visualizations
        (defaults to False in bootstrap)
    """
    # Obtain the size of the input image.
    with Image.open(filein) as img:
        n, m = img.size
    # Select which frequencies to retain.
    maski = set(
        np.floor(m * np.random.uniform(size=round(m * subsampling_factor))))
    mask = np.asarray([False] * m, dtype=bool)
    for i in maski:
        mask[int(i)] = True
    # Make the optimization well-posed by including the zero frequency.
    mask[0] = True
    # Include all low frequencies.
    low = round(math.sqrt(2. * m))
    for k in range(low):
        mask[k] = True
        mask[-k] = True
    # Generate bootstrap plots.
    loss, losses, rsse_estimated, rsse_blurred = bootstrap(
        filein, fileout, subsampling_factor, mask, low, recon, recon_args,
        n_resamps, sdn, viz)
    # Display the losses.
    print('loss = {}'.format(loss))
    print('losses = {}'.format(losses))
    # Display the estimated root-sum-square errors.
    print('Frobenius norm of the bootstrap = {}'.format(rsse_estimated))
    print('Frobenius norm of the blurred bootstrap = {}'.format(rsse_blurred))


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    np.random.seed(seed=1337)
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--filein', default='brain.png')
    parser.add_argument('--fileout', default='bootstrap.png')
    parser.add_argument('--subsampling_factor', type=float, default=.1)
    parser.add_argument('--mu', type=float, default=1e12)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--n_resamps', type=int, default=100)
    parser.add_argument('--sdn', type=float, default=0.02)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--viz', default=False, dest='viz', action='store_true')
    group.add_argument('--no-viz', dest='viz', action='store_false')
    args = parser.parse_args()
    # Run the test of bootstrap.
    testbootstrap(
        args.filein, args.fileout, args.subsampling_factor, admm.cs_fft,
        dict(mu=args.mu, beta=args.beta, n_iter=args.n_iter), args.n_resamps,
        args.sdn, args.viz)
