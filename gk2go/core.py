"""A module for fetching, calibrating, and geolocating GK-2A L1B satellite data.

This library provides a high-level interface to access Level 1B data from
the Geostationary Korea Multi-Purpose Satellite-2A (GK-2A) Advanced
Meteorological Imager (AMI) sensor. The data is sourced from the public
NOAA PDS on Amazon S3.

The primary class, Gk2aDataFetcher, handles the entire workflow:
  - Discovering files on S3 based on user queries (latest, nearest time, or
    time range).
  - Lazily loading the NetCDF data into memory-efficient xarray.Dataset objects
    using dask.
  - Performing radiometric calibration to convert raw digital numbers into
    scientifically meaningful physical units (Albedo or Brightness Temperature)
    based on official KMA specifications.
  - Calculating and attaching geographic coordinates (latitude, longitude) to
    each pixel using a validated implementation of the official KMA GEOS
    projection algorithm.

Example:
    >>> from core import Gk2aDataFetcher
    >>> fetcher = Gk2aDataFetcher()
    >>> ds = fetcher.get_data(
    ...     sensor='ami',
    ...     product='vi006',
    ...     area='fd',
    ...     query_type='latest'
    ... )
    >>> if ds:
    ...     print(ds['albedo'])
"""

import os
import re
import sys
from datetime import datetime, timedelta
import traceback

import boto3
import xarray as xr
import s3fs
from botocore.config import Config
from botocore import UNSIGNED
import numpy as np


class EarthConstants:
    """Container for Earth-related physical constants used in projections.
    
    Attributes:
        RP (float): The semi-minor axis (polar radius) of the Earth in km,
            based on the WGS84 ellipsoid model.
        RE (float): The semi-major axis (equatorial radius) of the Earth in km,
            based on the WGS84 ellipsoid model.
    """
    RP = 6356.75231414
    RE = 6378.137

class GK2ADefs:
    """Defines constants and static methods for GK-2A data processing.
    
    This class centralizes constants such as S3 bucket names, filename parsing
    regular expressions, and data quality thresholds to ensure consistency
    and ease of maintenance.
    """
    AMI_FILENAME_RE = re.compile(
        r"^(?P<satellite>gk2a)_"
        r"(?P<sensor>ami)_"
        r"(?P<level>le1b)_"
        r"(?P<product>[a-z]{2}\d{3})_"
        r"(?P<area>fd|la)"
        r"(?P<resolution>\d{3})"
        r"(?P<projection>ge)_"
        r"(?P<timestamp>\d{12})\."
        r"(?P<format>nc|png)$"
    )
    S3_BUCKET = "noaa-gk2a-pds"
    DQF_ERROR_THRESHOLD = 49151

    @staticmethod
    def parse_filename(filename):
        """Parses a GK-2A AMI filename to extract metadata.

        Args:
            filename (str): The filename to parse (e.g., 
                'gk2a_ami_le1b_vi006_fd010ge_202401010000.nc').

        Returns:
            dict or None: A dictionary containing the parsed components of the
                filename if it matches the expected format, otherwise None.
                The dictionary includes a 'datetime' object for the timestamp.
        """
        ami_match = GK2ADefs.AMI_FILENAME_RE.match(filename)
        if ami_match:
            data = ami_match.groupdict()
            try:
                data['datetime'] = datetime.strptime(data['timestamp'], '%Y%m%d%H%M')
            except ValueError:
                data['datetime'] = None
            return data
        return None

    @staticmethod
    def get_attr_scalar(ds, attr_name, default=None):
        """Safely retrieves a scalar attribute from an xarray.Dataset.

        This helper function handles cases where the attribute might be stored
        as a single-element array or list, returning a standard Python scalar.

        Args:
            ds (xarray.Dataset): The dataset from which to retrieve the attribute.
            attr_name (str): The name of the attribute.
            default: The default value to return if the attribute is not found.

        Returns:
            The scalar value of the attribute, or the default value.
        """
        value = ds.attrs.get(attr_name, default)
        if isinstance(value, (np.ndarray, xr.DataArray)):
            return value.item() if value.size == 1 else value.flatten()[0].item()
        return value

def _compute_geos_latlon(image_width, image_height, resolution_km):
    """Computes the latitude/longitude grid from pixel coordinates.

    This function is a direct, faithful implementation of the official KMA
    geolocation algorithm. It converts pixel line and column numbers into
    geographic coordinates based on a GEOS projection. The projection
    constants are specific to each spatial resolution.

    Args:
        image_width (int): The number of columns in the image.
        image_height (int): The number of lines in the image.
        resolution_km (float): The spatial resolution of the image (0.5, 1.0, or 2.0).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two 2D numpy arrays:
        (latitude, longitude). The arrays have the shape (image_height, image_width).
        
    Raises:
        ValueError: If an unsupported resolution is provided.
    """
    sub_lon = 128.2
    h = 42164.0
    k = 1.006739501
    h_sq_minus_req_sq = 1737122264

    # Select KMA projection constants based on resolution
    if resolution_km == 2.0:
        coff, loff = 2750.5, 2750.5
        cfac, lfac = 20425338.90333935, 20425338.90333935
    elif resolution_km == 1.0:
        coff, loff = 5500.5, 5500.5
        cfac, lfac = 40850677.8066787, 40850677.8066787
    elif resolution_km == 0.5:
        coff, loff = 11000.5, 11000.5
        cfac, lfac = 81701355.61335742, 81701355.61335742
    else:
        raise ValueError(f"Unsupported resolution for geolocation: {resolution_km} km")

    cols = np.arange(image_width, dtype=np.float64)
    lines = np.arange(image_height, dtype=np.float64)
    col_grid, lin_grid = np.meshgrid(cols, lines)

    # Mathematical implementation from KMA sample code
    x = np.deg2rad((col_grid - coff) * (2**16 / cfac))
    y = np.deg2rad((lin_grid - loff) * (2**16 / lfac))

    with np.errstate(invalid='ignore'):
        sd = np.sqrt(
            (h * np.cos(x) * np.cos(y))**2 - (np.cos(y)**2 + k * np.sin(y)**2) * h_sq_minus_req_sq
        )
    
    sn = (h * np.cos(x) * np.cos(y) - sd) / (np.cos(y)**2 + k * np.sin(y)**2)
    s1 = h - sn * np.cos(x) * np.cos(y)
    s2 = sn * np.sin(x) * np.cos(y)
    s3 = -sn * np.sin(y)
    sxy = np.sqrt(s1**2 + s2**2)
    
    lon = np.rad2deg(np.arctan2(s2, s1) + np.deg2rad(sub_lon))
    lat = np.rad2deg(np.arctan2(k * s3, sxy))

    invalid_mask = np.isnan(lat)
    lat[invalid_mask] = -999.0
    lon[invalid_mask] = -999.0

    return lat, lon


class S3Utils:
    """A collection of utility methods for interacting with AWS S3.

    This class abstracts away the low-level details of using boto3 and s3fs
    to find and access files in the public GK-2A data repository.
    """
    def __init__(self):
        """Initializes the S3 utility class.
        
        Sets up a boto3 client for unsigned (public) access and an s3fs
        filesystem object for integration with xarray.
        """
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.s3_fs = s3fs.S3FileSystem(anon=True)

    def list_s3_objects(self, bucket, prefix):
        """Lists all object keys in an S3 bucket that match a given prefix.

        Args:
            bucket (str): The name of the S3 bucket.
            prefix (str): The prefix (folder path) to search within.

        Returns:
            list[str]: A list of S3 object keys.
        """
        objects = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    objects.append(obj["Key"])
        return objects

class Gk2aDataFetcher:
    """The main interface for fetching, processing, and accessing GK-2A L1B data.

    This class orchestrates the entire data acquisition and processing workflow.
    It provides a single `get_data` method that handles file discovery,
    loading, calibration, and geolocation based on user-provided parameters.
    """
    def __init__(self, cache_dir=None):
        """Initializes the data fetcher.

        Args:
            cache_dir (str, optional): The directory to use for storing
                persistent geolocation cache files. If None, a default
                directory `~/.gk2go_cache` will be created and used.
        """
        self.s3_utils = S3Utils()
        self.s3_bucket = GK2ADefs.S3_BUCKET
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(os.path.expanduser('~'), '.gk2go_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_s3_prefixes(self, sensor, product, start_time, end_time, area):
        """Generates the S3 prefixes for a given time range.
        
        The GK-2A S3 bucket is organized by YYYYMM/DD/HH/. This method
        iterates through each hour in the time range to produce the
        correct folder paths to search for data.

        Args:
            sensor (str): The sensor name (e.g., 'ami').
            product (str): The product identifier (e.g., 'vi006').
            start_time (datetime): The beginning of the time range.
            end_time (datetime): The end of the time range.
            area (str): The geographic area (e.g., 'fd').

        Yields:
            str: An S3 prefix string for a specific hour.
        """
        base_prefix = f"{sensor.upper()}/L1B/{area.upper()}/"
        current_time = start_time
        while current_time <= end_time:
            yield f"{base_prefix}{current_time.strftime('%Y%m/%d/%H/')}"
            current_time += timedelta(hours=1)

    def _find_files(self, sensor, product, start_time, end_time, area):
        """Finds all relevant data files within a specified time range.

        Args:
            sensor (str): The sensor name.
            product (str): The product identifier.
            start_time (datetime): The start of the time range.
            end_time (datetime): The end of the time range.
            area (str): The geographic area.

        Returns:
            list[dict]: A sorted list of dictionaries, where each dictionary
                contains parsed metadata for a found file.
        """
        all_files = []
        for prefix in self._generate_s3_prefixes(sensor, product, start_time, end_time, area):
            for obj_key in self.s3_utils.list_s3_objects(self.s3_bucket, prefix):
                filename = os.path.basename(obj_key)
                if product in filename:
                    parsed_data = GK2ADefs.parse_filename(filename)
                    if parsed_data and start_time <= parsed_data['datetime'] <= end_time:
                        parsed_data['s3_key'] = obj_key
                        all_files.append(parsed_data)
        all_files.sort(key=lambda x: x['datetime'])
        return all_files

    def _load_as_xarray(self, s3_path, debug=False):
        """Loads a NetCDF file from S3 into a lazy-loaded xarray.Dataset.

        Args:
            s3_path (str): The full S3 object key (path) of the file.
            debug (bool): If True, prints status messages.

        Returns:
            xarray.Dataset or None: The loaded dataset, or None on failure.
        """
        try:
            if debug: print(f"Loading data from: {s3_path}")
            remote_file = self.s3_utils.s3_fs.open(s3_path, 'rb')
            return xr.open_dataset(remote_file, engine='h5netcdf', chunks='auto')
        except Exception as e:
            print(f"Error loading xarray dataset from {s3_path}: {e}", file=sys.stderr)
            if debug: traceback.print_exc()
            return None

    def _calibrate(self, ds, product_name, debug=False):
        """Performs radiometric calibration on the raw data.

        This method implements the scientifically correct calibration workflow:
        1. Filters out bad-quality pixels using the DQF.
        2. Applies a channel-specific bit mask to isolate valid data.
        3. Converts the cleaned digital numbers to radiance, then to either
           Albedo or Brightness Temperature.

        Args:
            ds (xarray.Dataset): The dataset containing `image_pixel_values`.
            product_name (str): The product identifier (e.g., 'vi006').
            debug (bool): If True, prints detailed calibration steps.

        Returns:
            xarray.Dataset: The dataset with a new calibrated data variable added
                (e.g., 'albedo' or 'brightness_temperature').
        """
        if debug: print(f"--- Entering Calibration for {product_name} ---")
        try:
            pixel_values = ds['image_pixel_values'].values.astype(np.float32)
            dqf_mask = pixel_values > GK2ADefs.DQF_ERROR_THRESHOLD
            if np.any(dqf_mask):
                pixel_values[dqf_mask] = np.nan
                if debug: print(f"  [FIX] Filtered {np.sum(dqf_mask)} DQF pixels.")
            
            valid_bits = int(GK2ADefs.get_attr_scalar(ds, 'valid_bits', 16))
            bit_mask = (2**valid_bits) - 1
            processed_pixels = np.bitwise_and(np.nan_to_num(pixel_values).astype(int), bit_mask).astype(np.float32)
            processed_pixels[dqf_mask] = np.nan
            if debug: print(f"  [FIX] Applied {valid_bits}-bit mask.")

            gain = GK2ADefs.get_attr_scalar(ds, 'DN_to_Radiance_Gain')
            offset = GK2ADefs.get_attr_scalar(ds, 'DN_to_Radiance_Offset')
            radiance = processed_pixels * gain + offset
            
            channel_type = product_name[:2]
            if channel_type in ['vi', 'nr']:
                c = GK2ADefs.get_attr_scalar(ds, 'Radiance_to_Albedo_c')
                albedo = np.clip(radiance * c, 0.0, 1.0) * 100
                ds['albedo'] = (ds['image_pixel_values'].dims, albedo)
            elif channel_type in ['sw', 'ir', 'wv']:
                radiance[radiance <= 0] = np.nan
                cval = GK2ADefs.get_attr_scalar(ds, 'light_speed')
                kval = GK2ADefs.get_attr_scalar(ds, 'Boltzmann_constant_k')
                hval = GK2ADefs.get_attr_scalar(ds, 'Plank_constant_h')
                wn = (10000.0 / GK2ADefs.get_attr_scalar(ds, 'channel_center_wavelength')) * 100.0
                radiance_for_planck = radiance * 1e-5
                e1 = (2.0 * hval * cval**2) * (wn**3)
                teff = ((hval * cval / kval) * wn) / np.log((e1 / radiance_for_planck) + 1.0)
                c0 = GK2ADefs.get_attr_scalar(ds, 'Teff_to_Tbb_c0')
                c1 = GK2ADefs.get_attr_scalar(ds, 'Teff_to_Tbb_c1')
                c2 = GK2ADefs.get_attr_scalar(ds, 'Teff_to_Tbb_c2')
                tbb = c2 * teff**2 + c1 * teff + c0
                ds['brightness_temperature'] = (ds['image_pixel_values'].dims, tbb)
            return ds
        except Exception as e:
            print(f"CALIBRATION FAILED for {product_name}: {e}", file=sys.stderr)
            if debug: traceback.print_exc()
            return ds

    def _add_geolocation(self, ds, debug=False):
        """Calculates and attaches geolocation coordinates to the dataset.

        This method uses a persistent file cache to store computed geolocation
        grids, avoiding re-computation on subsequent runs. It adds both
        geographic (lat/lon) and native geostationary (x/y) coordinates.

        Args:
            ds (xarray.Dataset): The dataset to add coordinates to.
            debug (bool): If True, prints status messages.

        Returns:
            xarray.Dataset: The dataset with added coordinate variables.
        """
        try:
            image_width = int(GK2ADefs.get_attr_scalar(ds, 'number_of_columns'))
            image_height = int(GK2ADefs.get_attr_scalar(ds, 'number_of_lines'))
            resolution_km = float(GK2ADefs.get_attr_scalar(ds, 'channel_spatial_resolution'))
            
            cache_filename = f"geoloc_{image_width}x{image_height}_res{resolution_km}km.npz"
            cache_filepath = os.path.join(self.cache_dir, cache_filename)

            if os.path.exists(cache_filepath):
                if debug: print(f"Loading geolocation from cache: {cache_filepath}")
                with np.load(cache_filepath) as data:
                    lat, lon = data['latitude'], data['longitude']
            else:
                print(f"NOTICE: Geolocation cache not found. Creating file: {cache_filepath}")
                lat, lon = _compute_geos_latlon(image_width, image_height, resolution_km)
                np.savez_compressed(cache_filepath, latitude=lat.astype(np.float32), longitude=lon.astype(np.float32))

            dims = ds['image_pixel_values'].dims
            ds = ds.assign_coords(latitude=(dims, lat), longitude=(dims, lon))

            sat_height = GK2ADefs.get_attr_scalar(ds, 'nominal_satellite_height')
            MRAD_2KM_BASE = 56.0e-6
            if resolution_km == 2.0: MRAD = MRAD_2KM_BASE
            elif resolution_km == 1.0: MRAD = MRAD_2KM_BASE * 2.0
            elif resolution_km == 0.5: MRAD = MRAD_2KM_BASE * 4.0
            
            x_rad = (np.arange(image_width) - (image_width / 2.0)) * MRAD
            y_rad = (np.arange(image_height) - (image_height / 2.0)) * MRAD
            ds = ds.assign_coords(x=(dims[1], x_rad * sat_height))
            ds = ds.assign_coords(y=(dims[0], y_rad * sat_height))
            
            return ds
        except Exception as e:
            print(f"GEOLOCATION FAILED: {e}", file=sys.stderr)
            if debug: traceback.print_exc()
            return ds

    def get_data(self, sensor, product, area, query_type, target_time=None,
                 start_time=None, end_time=None, calibrate=True, geolocation_enabled=True, debug=False):
        """Fetches, calibrates, and geolocates GK-2A data.

        This is the main public method of the class. It orchestrates the entire
        workflow based on the user's query.

        Args:
            sensor (str): The sensor name (e.g., 'ami').
            product (str): The product identifier (e.g., 'vi006').
            area (str): The geographic area (e.g., 'fd').
            query_type (str): The type of query: 'latest', 'nearest', or 'range'.
            target_time (datetime, optional): The target time for a 'nearest' query.
            start_time (datetime, optional): The start of the time range for a 'range' query.
            end_time (datetime, optional): The end of the time range for a 'range' query.
            calibrate (bool): If True, perform radiometric calibration.
            geolocation_enabled (bool): If True, calculate and add geolocation coordinates.
            debug (bool): If True, print verbose processing logs.

        Returns:
            xarray.Dataset or None: A dataset containing the processed data, or
            None if no files were found for the query. For 'range' queries,
            datasets are concatenated along a new 'time' dimension.
        """
        
        found_files = []
        if query_type == 'latest':
            found_files = self._find_files(sensor, product, datetime.utcnow() - timedelta(hours=24), datetime.utcnow(), area)
            if found_files: found_files = [found_files[-1]]
        elif query_type == 'nearest':
            if not target_time: raise ValueError("`target_time` required for 'nearest' query.")
            found_files = self._find_files(sensor, product, target_time - timedelta(hours=6), target_time + timedelta(hours=6), area)
            if found_files: found_files = [min(found_files, key=lambda x: abs(x['datetime'] - target_time))]
        elif query_type == 'range':
            if not start_time or not end_time: raise ValueError("`start_time` and `end_time` required for 'range' query.")
            found_files = self._find_files(sensor, product, start_time, end_time, area)

        if not found_files:
            print(f"No '{product}' data found for the specified query.")
            return None

        datasets = []
        for file_info in found_files:
            ds = self._load_as_xarray(f"s3://{self.s3_bucket}/{file_info['s3_key']}", debug=debug)
            if ds is not None:
                if calibrate:
                    ds = self._calibrate(ds, product, debug=debug)
                if geolocation_enabled:
                    ds = self._add_geolocation(ds, debug=debug)
                ds = ds.expand_dims(time=[file_info['datetime']])
                datasets.append(ds)
        
        if not datasets: return None
        return xr.concat(datasets, dim='time') if len(datasets) > 1 else datasets[0]
