#!/usr/bin/env python3

"""
Jackknife of reconstruction from compressed sensing, sampling k-space radially.

Copyright (c) Facebook, Inc. and its affiliates.

This computes twice the sum of the leave-one-out errors of reconstructions from
compressed sensing, providing a jackknife estimate of the error.

_N.B._: Whereas the sampling patterns in jackknife.py appear often in practice,
the patterns from radialines.py used in this jackknife2.py likely are relevant
only for prototyping.

Functions
---------
jackknife2
    Plots twice the sum of the leave-one-out errors of reconstructions.
testjackknife2
    Tests jackknife2.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import argparse
import logging

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

import ctorch
import admm2
import radialines


def jackknife2(filein, fileout, angles, low, recon, recon_args, sdn=None):
    """
    plots twice the sum of the leave-one-out errors of reconstructions

    Plots and saves twice the sum of the leave-one-out differences between
    the original image in filein and the reconstruction via recon applied to
    the k-space subsampling specified by angles fed into radialines, while
    including all frequencies between -low to low in both directions,
    corrupting the k-space values with independent and identically distributed
    centered complex Gaussian noise whose standard deviation is sdn*sqrt(2)
    (sdn=0 if not provided explicitly). The "one" left out in the leave-one-out
    is a radial "line" at the angles specified by angles.

    The calling sequence of recon must be  (m, n, f, mask, **recon_args),
    where filein contains an m x n image, f is the image in k-space subsampled
    to the mask, mask is the return from calls to radialines (with angles),
    supplemented by all frequencies between -low to low in both directions, and
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
    angles : list of float
        angles of the radial "lines" in the mask that radialines will construct
    low : int
        bandwidth of low frequencies to include in the mask
        (between -low to low in both the horizontal and vertical directions)
    recon : function
        returns the reconstructed image
    recon_args : dict
        keyword arguments for recon
    sdn : float, optional
        standard deviation of the noise to add (defaults to 0)

    Returns
    -------
    float
        loss for the reconstruction using all angles
    list of float
        losses for the reconstructions using all angles except for one
    """
    # Set default parameters.
    if sdn is None:
        sdn = 0
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
    # Select which frequencies to retain.
    mask = radialines.randradialines(m, n, angles)
    # Include all low frequencies.
    for km in range(low):
        for kn in range(low):
            mask[km, kn] = True
            mask[m - 1 - km, kn] = True
            mask[km, n - 1 - kn] = True
            mask[m - 1 - km, n - 1 - kn] = True
    # Subsample the noisy Fourier transform of the original image.
    f = ctorch.from_numpy(ff_noisy[mask]).cuda()
    logging.info(
        'computing jackknife2 differences -- all {}'.format(len(angles)))
    # Perform the reconstruction using the entire mask.
    reconf, lossf = recon(m, n, f, mask, **recon_args)
    reconf = reconf.cpu().numpy()
    # Perform the reconstruction omitting different samples in k-space.
    recons = np.ndarray((angles.size, m, n))
    loss = []
    for k in range(angles.size):
        # Drop an angle.
        langles = list(angles)
        del langles[k]
        mask1 = radialines.randradialines(m, n, langles)
        # Include all low frequencies.
        for km in range(low):
            for kn in range(low):
                mask1[km, kn] = True
                mask1[m - 1 - km, kn] = True
                mask1[km, n - 1 - kn] = True
                mask1[m - 1 - km, n - 1 - kn] = True
        # Subsample the noisy Fourier transform of the original image.
        f1 = ctorch.from_numpy(ff_noisy[mask1]).cuda()
        # Reconstruct the image from the subsampled data.
        recon1, loss1 = recon(m, n, f1, mask1, **recon_args)
        recon1 = recon1.cpu().numpy()
        # Record the results.
        recons[k, :, :] = recon1
        loss.append(loss1)
    # Calculate the sum of the leave-one-out differences.
    sumloo = np.sum(recons - reconf, axis=0)
    scaled = sumloo * 2

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
    # Plot twice the sum of the leave-one-out differences.
    plt.figure(figsize=(5.5, 5.5))
    plt.title('Jackknife')
    plt.imshow(np.clip(scaled, -1, 1), **kwargs11)
    plt.savefig(rest + '_jackknife' + suffix, bbox_inches='tight')

    return lossf, loss


def testjackknife2(filein, fileout, subsampling_factor, recon, recon_args,
                   sdn=None):
    """
    tests jackknife2

    Runs jackknife2 (and prints the losses) with filein, fileout, recon,
    recon_args, and sdn, creating a random mask that retains each radial "line"
    with probability subsampling_factor.

    The calling sequence of recon must be  (m, n, f, mask, **recon_args),
    where filein contains an m x n image, f is the image in k-space subsampled
    to the mask, mask is the return from calls to radialines (with angles),
    and **recon_args is the unpacking of recon_args.
    The function recon must return a torch.Tensor (the reconstruction) and
    a float (the corresponding loss).

    Parameters
    ----------
    filein : str
        path to the file containing the image to be processed (the path may be
        relative or absolute)
    fileout : str
        path to the file to which the plots will be saved (the path may be
        relative or absolute)
    subsampling_factor : float
        probability of retaining a radial "line" in the subsampling mask
    recon : function
        returns the reconstructed image
    recon_args : dict
        keyword arguments for recon
    sdn : float, optional
        standard deviation of the noise to add (defaults to 0 in jackknife2)
    """
    # Obtain the size of the input image.
    with Image.open(filein) as img:
        n, m = img.size
    # Select which frequencies to retain.
    angles = np.random.uniform(low=0, high=(2 * np.pi),
                               size=round(2 * (m + n) * subsampling_factor))
    # Generate jackknife plots.
    low = 0
    loss, losses = jackknife2(filein, fileout, angles, low, recon, recon_args,
                              sdn)
    # Display the losses.
    print('loss = {}'.format(loss))
    print('losses = {}'.format(losses))


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    np.random.seed(seed=1337)
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--filein', default='brain.png')
    parser.add_argument('--fileout', default='jackknife2.png')
    parser.add_argument('--subsampling_factor', type=float, default=.1)
    parser.add_argument('--mu', type=float, default=1e12)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--sdn', type=float, default=0.02)
    args = parser.parse_args()
    # Run the test of jackknife2.
    testjackknife2(
        args.filein, args.fileout, args.subsampling_factor, admm2.cs_fft,
        dict(mu=args.mu, beta=args.beta, n_iter=args.n_iter), args.sdn)
