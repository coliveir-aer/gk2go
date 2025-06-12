# **gk2go**

**A Python library for discovering and loading Geo-KOMPSAT-2A (GK2A) satellite data from the public AWS Open Data repository.**

gk2go provides a simple yet powerful, goes2go-inspired interface to query and access GK2A satellite data products. It allows for flexible queries to get the latest data, data nearest to a specific time, or a time-series range of data loaded directly into an xarray.Dataset with Dask for memory-efficient, lazy-loading.

## **Core Features**

* **Flexible Queries**: Fetch data by 'latest', 'nearest' time, or a 'range'.  
* **Memory Efficient**: Uses s3fs, xarray, and dask to stream data directly from S3 without downloading the entire file first. This allows you to work with massive datasets on memory-constrained machines.  
* **Time-series Ready**: 'range' queries return a 3D xarray.Dataset stacked along the time dimension, perfect for analysis and animation.  
* **Simple Interface**: A clean, straightforward API for finding and loading the data you need.

## **Installation**

You can install gk2go directly from its GitHub repository:

    pip install git+https://github.com/coliveir-aer/gk2go.git

## **Quick Start**

The /examples directory contains scripts demonstrating how to use the library. For instance, plot\_all\_channels.py shows how to fetch and visualize the latest image for all 16 GK2A channels.

Here is a basic example:

```
import matplotlib.pyplot as plt
import pandas as pd
from gk2go import Gk2aDataFetcher

# 1. Initialize the fetcher
fetcher = Gk2aDataFetcher()

# 2. Get the latest calibrated image for the 10.5 µm "Clean IR" channel
#    Setting calibrate=True converts the raw data to Brightness Temperature.
latest_calibrated_ds = fetcher.get_data(
    sensor='ami', product='ir105', area='fd', query_type='latest', calibrate=True
)

# 3. Check if data was found and calibrated, then plot it
if latest_calibrated_ds and 'brightness_temperature' in latest_calibrated_ds:
    print("--- Latest Calibrated Data Found ---")
    print(latest_calibrated_ds)
    
    # Extract the data and the timestamp for plotting
    bt_data = latest_calibrated_ds.squeeze()['brightness_temperature']
    time_val = pd.to_datetime(latest_calibrated_ds.time.values.item())
    time_str = time_val.strftime('%Y-%m-%d %H:%M Z')
    
    # Create a simple plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Use a standard temperature colormap and display the data
    im = ax.imshow(bt_data, cmap='inferno')
    
    # Add a colorbar with the correct label and units
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(f"{bt_data.attrs['long_name']} ({bt_data.attrs['units']})")
    
    ax.set_title(f'Latest Calibrated GK-2A (10.5 µm)\n{time_str}')
    ax.axis('off') # Hide the axes for a cleaner image
    plt.show()

```

## **Contributing**

Contributions are welcome\! Please feel free to submit a pull request or open an issue.

## **License**

This project is licensed under the MIT License \- see the [LICENSE](http://docs.google.com/LICENSE) file for details.
