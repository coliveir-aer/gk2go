"""
Example: Plot All 16 GK2A Channels

This script demonstrates how to use the gk2go library to fetch a
full disk image for all 16 AMI channels, perform radiometric calibration,
and display them in a 4x4 grid.
"""

import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import numpy as np
import sys
import os
import matplotlib.cm as cm

from gk2go import Gk2aDataFetcher
from gk2go.core import GK2ADefs

def plot_calibrated_data(fig, dataset, ax, title=''):
    """
    Plots calibrated GK2A data, decimating for display.
    """
    try:
        data_var = None
        cmap = 'viridis'
        units = ''
        
        # Determine which variable to plot
        if 'albedo' in dataset:
            data_var = dataset['albedo'].squeeze()
            cmap = 'gray'
            units = '%'
        elif 'brightness_temperature' in dataset:
            data_var = dataset['brightness_temperature'].squeeze()
            cmap = 'gray_r' # Inverted for temperature
            units = 'K'
        else:
            # Fallback for unexpected data
            ax.text(0.5, 0.5, 'No Calibrated Data', ha='center', va='center', color='red')
            ax.set_title(title, fontsize=10)
            return

        # Decimate the 5500x5500 (2km) data down to 550x550 for plotting
        # This triggers the Dask computation on a much smaller array
        decimation_factor = 10
        dat_to_plot = data_var[::decimation_factor, ::decimation_factor].compute()

        # Use percentile stretching for robust contrast
        vmin, vmax = np.nanpercentile(dat_to_plot, [2, 98])

        # Plot the actual calibrated data
        im = ax.imshow(dat_to_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add a labeled colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(f"{units}", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        wavelength = GK2ADefs.get_attr_scalar(dataset, 'channel_center_wavelength', 'N/A')
        ax.set_title(f"{title} ({wavelength} um)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    except Exception as e:
        print(f"ERROR: Could not plot data for {title}. Error: {e}")
        ax.text(0.5, 0.5, 'Plotting Error', ha='center', va='center', color='red')
        ax.set_title(title, fontsize=10)


def main():
    """
    Main function to run the program.
    """
    fetcher = Gk2aDataFetcher()

    ami_channels = [
        'vi004', 'vi005', 'vi006', 'vi008', 'nr013', 'nr016', 'sw038', 'wv063',
        'wv069', 'wv073', 'ir087', 'ir096', 'ir105', 'ir112', 'ir123', 'ir133'
    ]

    # Define a recent target time
    target_time_utc = datetime.utcnow() - timedelta(hours=3)

    fig, axes = plt.subplots(4, 4, figsize=(16, 18))
    axes = axes.flatten()
    fig.suptitle(f'Calibrated GK2A Full Disk Images Nearest to {target_time_utc.strftime("%Y-%m-%d %H:%M UTC")}', fontsize=16, y=0.98)

    for i, channel in enumerate(ami_channels):
        ax = axes[i]
        print(f"\n--- Processing Channel: {channel.upper()} ---")
        
        ds = fetcher.get_data(
            sensor='ami',
            product=channel,
            area='fd',
            query_type='nearest',
            target_time=target_time_utc,
            calibrate=True,
            debug=True
        )
        if ds:
            plot_calibrated_data(fig, ds, ax, title=channel.upper())
        else:
            ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', color='red')
            ax.set_title(channel.upper(), fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure as a PNG file
    output_filename = f"gk2a_fd_16ch_calibrated_{target_time_utc.strftime('%Y%m%d_%H%MUTC')}.png"
    
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved successfully as: {output_filename}")
    except Exception as e:
        print(f"ERROR: Could not save plot to file {output_filename}. Error: {e}", file=sys.stderr)

    plt.show()

if __name__ == "__main__":
    main()
