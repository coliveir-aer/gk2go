# **gk2go**

**A Python library for discovering and loading Geo-KOMPSAT-2A (GK2A) satellite data from the public AWS Open Data repository.**

gk2go provides a simple yet powerful, goes2go-inspired interface to query and access GK2A satellite data products. It allows for flexible queries to get the latest data, data nearest to a specific time, or a time-series range of data loaded directly into an xarray.Dataset with Dask for memory-efficient, lazy-loading.

## **Core Features**

* **Flexible Queries**: Fetch data by 'latest', 'nearest' time, or a 'range'.
* **On-the-fly Calibration**: Convert raw data to scientific units (Albedo or Brightness Temperature) with a simple `calibrate=True` flag.
* **Memory Efficient**: Uses s3fs, xarray, and dask to stream data directly from S3 without downloading the entire file first.
* **Time-series Ready**: 'range' queries return a 3D xarray.Dataset stacked along the time dimension, perfect for analysis and animation.

## **Installation**

You can install gk2go directly from its GitHub repository:

    pip install git+https://github.com/coliveir-aer/gk2go.git

## Quick Start: Plotting All 16 Calibrated Channels

This example demonstrates how to fetch GK2A Full Disk data for all 16 AMI channels, specifically finding the scene nearest to noon KST from yesterday. It then calibrates the data to scientific units (Albedo or Brightness Temperature) and displays them in a 4x4 grid with appropriate color scaling, saving the output as a PNG file.

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import numpy as np
import sys
import os

try:
    from gk2go import Gk2aDataFetcher
except ImportError:
    print("FATAL ERROR: Could not import Gk2aDataFetcher.")
    print("Please ensure you have installed the gk2go library.")
    print("From the project root, run: pip install .")
    sys.exit(1)

def plot_calibrated_data(fig, dataset, ax, title=''):
    """
    Intelligently plots calibrated GK2A data (Albedo or Brightness Temp).
    """
    try:
        # Determine which variable to plot
        if 'albedo' in dataset:
            dat = dataset.squeeze()['albedo']
            cmap = 'gray'
        elif 'brightness_temperature' in dataset:
            dat = dataset.squeeze()['brightness_temperature']
            cmap = 'gray_r' # Inverted for temperature
        else: # Fallback to raw pixel values
            dat = dataset.squeeze()['image_pixel_values']
            cmap = 'gray'

        y_dim_size = dat.shape[0]
        decimation = max(1, y_dim_size // 1100) * 2
        dat_to_plot = dat[::decimation, ::decimation].load()

        # Handle fill values by converting them to NaN so they are not plotted
        if '_FillValue' in dat.attrs:
            dat_to_plot = dat_to_plot.where(dat_to_plot != dat.attrs['_FillValue'])

        # Use percentile stretching for robust contrast
        vmin, vmax = np.nanpercentile(dat_to_plot, [2, 98])

        # Plot the actual calibrated data with the defined color limits
        im = ax.imshow(dat_to_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add a labeled colorbar with the correct units
        units = dat.attrs.get('units', 'N/A')
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(f"{units}", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        wavelength = dataset.attrs.get('channel_center_wavelength', 'N/A')
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

    # Define target time for yesterday's noon over Korea
    yesterday_utc_date = (datetime.utcnow() - timedelta(days=1)).date()
    # Noon KST is 03:00 UTC (KST = UTC + 9 hours)
    target_time_utc = datetime.combine(yesterday_utc_date, time(3, 0))

    target_area = 'fd'

    fig, axes = plt.subplots(4, 4, figsize=(16, 18))
    axes = axes.flatten()
    fig.suptitle(f'Calibrated GK2A Full Disk Images Nearest to {target_time_utc.strftime("%Y-%m-%d %H:%M UTC")}', fontsize=16, y=0.98)

    for i, channel in enumerate(ami_channels):
        ax = axes[i]
        ds = fetcher.get_data(
            sensor='ami',
            product=channel,
            area=target_area,
            query_type='nearest',
            target_time=target_time_utc,
            calibrate=True
        )
        if ds:
            plot_calibrated_data(fig, ds, ax, title=channel.upper())
        else:
            ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', color='red')
            ax.set_title(channel.upper(), fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure as a PNG file
    output_filename = f"gk2a_fd_16ch_{target_time_utc.strftime('%Y%m%d_%H%MUTC')}.png"
    
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved successfully as: {output_filename}")
    except Exception as e:
        print(f"ERROR: Could not save plot to file {output_filename}. Error: {e}", file=sys.stderr)

    plt.show()

if __name__ == "__main__":
    main()
```

## **Contributing**

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## **License**

This project is licensed under the MIT License.
