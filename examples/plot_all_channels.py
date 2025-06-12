#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: Plot All GK2A Channels

This script demonstrates how to use the `gk2go` library to fetch the
latest full disk image for all 16 AMI channels and display them in a
4x4 grid.

To run this script, first install the `gk2go` library and its dependencies:
  pip install -r ../requirements.txt
  pip install .  (from the root project directory)
"""

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from skimage import exposure
import sys

# Import the data fetcher from our installed library
try:
    from gk2go import Gk2aDataFetcher
except ImportError:
    print("FATAL ERROR: Could not import Gk2aDataFetcher.")
    print("Please ensure you have installed the gk2go library.")
    print("From the project root, run: pip install .")
    sys.exit(1)

def plot_l1b_data(fig, dataset, ax, title=''):
    """
    Plots the GK2A L1B data on a matplotlib axis, using histogram
    equalization for optimal contrast and adds a labeled colorbar.
    """
    try:
        dat = dataset.squeeze()['image_pixel_values']
        y_dim_size = dat.shape[0]
        decimation = max(1, y_dim_size // 1100) * 2
        dat_to_plot = dat[::decimation, ::decimation].load()

        if '_FillValue' in dat.attrs:
            image_data = dat_to_plot.where(dat_to_plot != dat.attrs['_FillValue'])
        else:
            image_data = dat_to_plot

        mask = ~np.isnan(image_data)
        equalized_data = np.full(image_data.shape, np.nan, dtype=float)
        if np.any(mask):
            equalized_data[mask] = exposure.equalize_hist(image_data.values[mask])

        cmap = 'gray_r' if title.startswith('IR') or title.startswith('WV') else 'gray'
        im = ax.imshow(equalized_data, cmap=cmap)
        
        units = dat.attrs.get('units', 'DN')
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(f"Equalized {units}", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_ticks([0, 0.5, 1.0])

        wavelength = dataset.attrs.get('channel_center_wavelength', 'N/A')
        ax.set_title(f"{title} ({wavelength} um)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    except Exception as e:
        print(f"ERROR: Could not plot L1B data for {title}. Error: {e}")
        ax.text(0.5, 0.5, 'Plotting Error', ha='center', va='center', color='red')
        ax.set_title(title, fontsize=10)


def main():
    """
    Main function to run the program.
    """
    # Initialize the fetcher
    fetcher = Gk2aDataFetcher()

    # Define all 16 AMI channels
    ami_channels = [
        'vi004', 'vi005', 'vi006', 'vi008', 'nr013', 'nr016', 'sw038', 'wv063',
        'wv069', 'wv073', 'ir087', 'ir096', 'ir105', 'ir112', 'ir123', 'ir133'
    ]

    # Set up the 4x4 plot
    fig, axes = plt.subplots(4, 4, figsize=(16, 18))
    axes = axes.flatten()
    fig.suptitle('Latest GK2A Full Disk Images by Channel', fontsize=20, y=0.98)

    # Loop through channels and plot
    for i, channel in enumerate(ami_channels):
        print(f"\n--- Processing Channel: {channel} ---")
        ax = axes[i]
        ds = fetcher.get_data(
            sensor='ami', product=channel, area='fd', query_type='latest'
        )
        if ds:
            plot_l1b_data(fig, ds, ax, title=channel.upper())
        else:
            ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', color='red')
            ax.set_title(channel.upper(), fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    
    # Finalize and show the plot
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
