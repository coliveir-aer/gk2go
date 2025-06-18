# **gk2go**

**A Python library for discovering and loading Geo-KOMPSAT-2A (GK2A) satellite data from the public AWS Open Data repository.**

gk2go provides a simple yet powerful, goes2go-inspired interface to query and access GK2A satellite data products. It allows for flexible queries to get the latest data, data nearest to a specific time, or a time-series range of data loaded directly into an xarray.Dataset with Dask for memory-efficient, lazy-loading.

Geolocation functionality is not included in this package-- the official .nc files with geolocations are available from KMA at [https://datasvc.nmsc.kma.go.kr/datasvc/html/base/cmm/selectPage.do?page=static.software&lang=en](https://datasvc.nmsc.kma.go.kr/datasvc/html/base/cmm/selectPage.do?page=static.software&lang=en)

## **Core Features**

* **Flexible Queries**: Fetch data by 'latest', 'nearest' time, or a 'range'.
* **On-the-fly Calibration**: Convert raw data to scientific units (Albedo or Brightness Temperature) with a simple `calibrate=True` flag.
* **Memory Efficient**: Uses s3fs, xarray, and dask to stream data directly from S3 without downloading the entire file first.
* **Time-series Ready**: 'range' queries return a 3D xarray.Dataset stacked along the time dimension, perfect for analysis and animation.

## **Installation**

You can install gk2go directly from its GitHub repository:

    pip install git+https://github.com/coliveir-aer/gk2go.git

## **Quick Start Examples**
### These simple examples demonstrate how to query and load GK-2A data.

```
from gk2go import Gk2aDataFetcher
from datetime import datetime, timedelta

# Initialize the data fetcher
fetcher = Gk2aDataFetcher()

# --- Example 1: Get the latest available image ---
# Finds the most recent Full Disk (fd) long-wave infrared (ir105) image 
# and applies radiometric calibration to get Brightness Temperature.
latest_ir_data = fetcher.get_data(
    sensor='ami', 
    product='ir105', 
    area='fd', 
    query_type='latest', 
    calibrate=True
)

# --- Example 2: Get the image nearest to a specific time ---
# Finds the short-wave infrared (sw038) image closest to a defined target_time
# and converts the raw values to Albedo.
target_time = datetime(2024, 6, 17, 3, 0) # 3:00 AM UTC on June 17, 2024

nearest_swir_data = fetcher.get_data(
    sensor='ami',
    product='sw038',
    area='fd',
    query_type='nearest',
    target_time=target_time,
    calibrate=True
)

# --- Example 3: Get a time-series of images over a specific range ---
# Retrieves all available full disk water vapor (wv069) images between
# a start_time and end_time and stacks them into a single xarray.Dataset.
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=2)

timeseries_wv_data = fetcher.get_data(
    sensor='ami',
    product='wv069',
    area='fd',
    query_type='range',
    start_time=start_time,
    end_time=end_time,
    calibrate=True
)

# --- Example 4: Get a raw, uncalibrated image ---
# Fetches the latest high-resolution visible (vi006) image but keeps the 
# data as the original pixel values by setting calibrate=False.
raw_visible_data = fetcher.get_data(
    sensor='ami',
    product='vi006',
    area='fd',
    query_type='latest',
    calibrate=False
)
```

## **Contributing**

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## **License**

This project is licensed under the MIT License.
