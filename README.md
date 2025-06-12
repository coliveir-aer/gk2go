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
from gk2go import Gk2aDataFetcher
import matplotlib.pyplot as plt

# 1. Initialize the fetcher  
fetcher = Gk2aDataFetcher()

# 2. Get the 'latest' image for the 10.5 µm channel  
latest_ds = fetcher.get_data(  
    sensor='ami', product='ir105', area='fd', query_type='latest'  
)

if latest_ds:  
    print("--- Latest Data Found ---")  
    print(latest_ds)


# 3. Display the image using matplotlib
if latest_ds:
  plt.figure(figsize=(10, 10))
  plt.imshow(latest_ds['image_pixel_values'][0], cmap='gray')
  plt.title(f"GK2A AMI IR105 Full Disk Image (Latest)")
  plt.colorbar(label='Pixel Value')
  plt.axis('off')  # Hide axes
  plt.show()
else:
  print("No data found to display.")
```

## **Contributing**

Contributions are welcome\! Please feel free to submit a pull request or open an issue.

## **License**

This project is licensed under the MIT License \- see the [LICENSE](http://docs.google.com/LICENSE) file for details.
