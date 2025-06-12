# **gk2go**

**A Python library for discovering and loading Geo-KOMPSAT-2A (GK2A) satellite data from the public AWS Open Data repository.**

gk2go provides a simple yet powerful, goes2go-inspired interface to query and access GK2A satellite data products. It allows for flexible queries to get the latest data, data nearest to a specific time, or a time-series range of data loaded directly into an xarray.Dataset with Dask for memory-efficient, lazy-loading.

## **Core Features**

* **Flexible Queries**: Fetch data by 'latest', 'nearest' time, or a 'range'.  
* **On-the-fly Calibration**: Convert raw data to scientific units (Albedo or Brightness Temperature) with a simple calibrate=True flag.  
* **Memory Efficient**: Uses s3fs, xarray, and dask to stream data directly from S3 without downloading the entire file first.  
* **Time-series Ready**: 'range' queries return a 3D xarray.Dataset stacked along the time dimension, perfect for analysis and animation.

## **Installation**

You can install gk2go directly from its GitHub repository:

    pip install git+https://github.com/coliveir-aer/gk2go.git

## **Quick Start: Plotting Calibrated Data**

Here’s a complete example of how to use gk2go to fetch the latest 10.5 µm infrared image, calibrate it to Brightness Temperature, and create a high-quality plot using the same robust techniques as the multi-channel example.

```
# First, ensure all dependencies are installed  
# !pip install gk2go matplotlib pandas numpy scikit-image

import matplotlib.pyplot as plt  
import pandas as pd  
from gk2go import Gk2aDataFetcher  
import numpy as np  
from skimage import exposure

# 1. Initialize the fetcher  
print("--- Initializing Gk2aDataFetcher ---")  
fetcher = Gk2aDataFetcher()

# 2. Get the latest calibrated image for the 10.5 µm "Clean IR" channel  
print("\n--- Fetching latest calibrated data for ir105 ---")  
ds = fetcher.get_data(  
    sensor='ami', product='ir105', area='fd', query_type='latest', calibrate=True  
)

# 3. Check if data was found and calibrated, then plot it  
if ds and 'brightness_temperature' in ds:  
    print("--- Latest Calibrated Data Found ---")  
      
    # Extract the primary data variable  
    dat = ds.squeeze()['brightness_temperature']  
      
    # --- Perform Subsampling and Contrast Enhancement ---  
    # Subsample for performance  
    y_dim_size = dat.shape[0]  
    decimation = max(1, y_dim_size // 1100) * 2  
    dat_to_plot = dat[::decimation, ::decimation].load()

    # Handle fill values  
    if '_FillValue' in dat.attrs:  
        image_data = dat_to_plot.where(dat_to_plot != dat.attrs['_FillValue'])  
    else:  
        image_data = dat_to_plot

    # Apply Histogram Equalization for optimal contrast  
    mask = ~np.isnan(image_data)  
    equalized_data = np.full(image_data.shape, np.nan, dtype=float)  
    if np.any(mask):  
        equalized_data[mask] = exposure.equalize_hist(image_data.values[mask])

    # --- Create the Plot ---  
    time_val = pd.to_datetime(ds.time.values.item())  
    time_str = time_val.strftime('%Y-%m-%d %H:%M Z')  
      
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))  
      
    # Use an inverted colormap which is standard for IR temperature  
    im = ax.imshow(equalized_data, cmap='gray_r')  
      
    # Add a colorbar  
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)  
    cbar.set_label(f"Equalized {dat.attrs.get('units', 'N/A')}")  
      
    ax.set_title(f"Latest Calibrated GK-2A (10.5 µm)\n{time_str}")  
    ax.axis('off')  
    plt.show()

else:  
    print("Could not find or calibrate the latest ir105 data.")
```

## **Contributing**

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## **License**

This project is licensed under the MIT License \- see the
