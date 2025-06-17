"""
This module provides a suite of tools for fetching, parsing, and calibrating
Level-1B data from the Geostationary Korea Multi-Purpose Satellite-2A (GK-2A)
Advanced Meteorological Imager (AMI) sensor, made available via the NOAA PDS
on Amazon S3.

It handles the complexities of navigating the S3 bucket structure, parsing
filenames, and applying calibration algorithms to convert raw digital numbers
into scientifically meaningful values like radiance, albedo, or brightness
temperature.

Key Features:
- Fetch the latest available data, data nearest to a specific time, or data
  within a given time range.
- On-the-fly calibration of raw data to physical units.
- Uses xarray for powerful, labeled, multi-dimensional data handling.
- Robust error handling for S3 interactions and calibration processes.
- **NEW**: Optional addition of latitude and longitude coordinates to the
  xarray Dataset.

Classes:
    GK2ADefs: Defines constants and static methods for GK-2A data.
    S3Utils: Provides utility functions for interacting with the S3 bucket.
    Gk2aDataFetcher: The main class for finding and loading GK-2A data.
"""

import os
import re
import sys
from datetime import datetime, timedelta
import traceback

import boto3
import pandas as pd
import xarray as xr
import s3fs
from botocore.config import Config
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from dateutil.parser import parse as parse_date
import numpy as np

# Import the new geolocation module
# Assumes geolocation.py is in the same directory or accessible via Python path
try:
    from . import geolocation
except ImportError:
    # Fallback for direct script execution or if not part of a package
    import geolocation


class GK2ADefs:
    """
    A container for constants and static definitions related to GK-2A AMI data.

    This class centralizes key information such as S3 bucket details, AWS region,
    and filename parsing logic to ensure consistency across the module.
    """
    # Regex to parse the standard GK-2A AMI LE1B filename format.
    # It captures key metadata components like satellite, sensor, product, area,
    # resolution, and timestamp.
    AMI_FILENAME_RE = re.compile(
        r"^(?P<satellite>gk2a)_"
        r"(?P<sensor>ami)_"
        r"(?P<level>le1b)_"
        r"(?P<product>[a-z]{2}\d{3})_"
        r"(?P<area>fd|ela|la)"
        r"(?P<resolution>\d{3})"
        r"(?P<projection>ge)_"
        r"(?P<timestamp>\d{12})\."
        r"(?P<format>nc|png)$"
    )
    # The public S3 bucket hosting the NOAA GK-2A PDS.
    S3_BUCKET = "noaa-gk2a-pds"
    # The AWS region where the S3 bucket is located.
    AWS_REGION = "us-east-1"

    @staticmethod
    def parse_filename(filename):
        """
        Parses a GK-2A AMI filename to extract metadata.

        Args:
            filename (str): The filename to parse.

        Returns:
            dict: A dictionary containing the parsed components of the filename,
                  including a 'datetime' object. Returns None if the filename
                  does not match the expected format.
        """
        ami_match = GK2ADefs.AMI_FILENAME_RE.match(filename)
        if ami_match:
            data = ami_match.groupdict()
            try:
                # Convert the timestamp string into a Python datetime object.
                data['datetime'] = datetime.strptime(data['timestamp'], '%Y%m%d%H%M')
            except ValueError:
                # Handle cases where the timestamp is not a valid date.
                data['datetime'] = None
            return data
        return None


class S3Utils:
    """
    A utility class for handling interactions with Amazon S3.

    This class abstracts the setup of boto3 and s3fs clients for anonymous
    access to public S3 buckets. It provides a simplified interface for
    listing objects within a specified prefix.
    """
    def __init__(self):
        """
        Initializes the S3 clients for anonymous access.

        Raises:
            Exception: If there is an error during client initialization.
        """
        try:
            # s3fs is used by xarray for opening S3 objects directly.
            self.fs = s3fs.S3FileSystem(anon=True)
            # boto3 is used for more granular S3 operations like listing.
            self.s3_client = boto3.client(
                's3',
                region_name=GK2ADefs.AWS_REGION,
                config=Config(signature_version=UNSIGNED)
            )
        except Exception as e:
            print(f"FATAL: Error initializing S3 clients: {e}", file=sys.stderr)
            raise

    def list_files_in_prefix(self, prefix):
        """
        Lists all files in a given S3 bucket and prefix.

        This method uses a paginator to efficiently handle prefixes that may
        contain a large number of objects.

        Args:
            prefix (str): The S3 prefix (folder path) to search within.

        Yields:
            dict: A dictionary representing an S3 object, as returned by
                  the boto3 list_objects_v2 operation.
        """
        paginator = self.s3_client.get_paginator('list_objects_v2')
        try:
            # Paginate through results to handle large directories.
            for page in paginator.paginate(Bucket=GK2ADefs.S3_BUCKET, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        yield obj
        except ClientError as e:
            # Suppress 'NoSuchKey' errors, as it's valid for a prefix to not exist.
            # Log other client errors.
            if e.response['Error']['Code'] != 'NoSuchKey':
                print(f"ERROR: S3 client error on prefix '{prefix}': {e}", file=sys.stderr)
            return


class Gk2aDataFetcher:
    """
    A class to fetch and process GK-2A satellite data from Amazon S3.

    This is the main entry point for users to acquire data. It orchestrates
    the process of finding the correct files based on user criteria, loading
    them into xarray Datasets, and optionally applying calibration.
    """
    def __init__(self):
        """Initializes the Gk2aDataFetcher with an S3 utility instance."""
        self.s3_utils = S3Utils()

    def _calibrate(self, ds, product_name):
        """
        Calibrates raw data in the xarray.Dataset to scientific units.

        This internal method applies the appropriate calibration formulas based
        on the channel type (Visible, Infrared, etc.). It converts the raw
        'image_pixel_values' (Digital Numbers or DNs) into either Albedo (%)
        for visible channels or Brightness Temperature (K) for thermal channels.

        The method is designed to be robust, with extensive error handling and
        debugging output to diagnose issues with metadata attributes. If
        calibration fails, it returns the original, uncalibrated dataset.

        Args:
            ds (xr.Dataset): The input dataset containing the raw pixel values
                             and necessary calibration coefficients as attributes.
            product_name (str): The product identifier (e.g., 'vi004', 'ir133')
                                 which determines the calibration path.

        Returns:
            xr.Dataset: The dataset with a new calibrated data variable
                        ('albedo' or 'brightness_temperature') added. Returns the
                        original dataset if calibration fails.
        """
        print(f"--- Entering Calibration for {product_name} ---")
        try:
            # --- Extensive Debugging Prints ---
            dn_variable = ds['image_pixel_values']
            gain_attr = ds.attrs.get('DN_to_Radiance_Gain', 'N/A')
            offset_attr = ds.attrs.get('DN_to_Radiance_Offset', 'N/A')

            print(f"\n[DEBUG] Raw 'image_pixel_values' info:")
            print(f"  - Type: {type(dn_variable)}")
            print(f"  - Shape: {dn_variable.shape}")
            print(f"  - Dtype: {dn_variable.dtype}")
            print(f"  - Underlying data type: {type(dn_variable.data)}")


            print(f"\n[DEBUG] Raw 'DN_to_Radiance_Gain' attribute info:")
            print(f"  - Type: {type(gain_attr)}")
            print(f"  - Value: {repr(gain_attr)}")

            print(f"\n[DEBUG] Raw 'DN_to_Radiance_Offset' attribute info:")
            print(f"  - Type: {type(offset_attr)}")
            print(f"  - Value: {repr(offset_attr)}")

            def get_scalar(attr_name):
                """
                Helper to robustly extract a scalar float from dataset attributes.
                Handles attributes that might be lists, tuples, or numpy arrays.
                """
                val = ds.attrs[attr_name]
                print(f"  [get_scalar] Initial value for '{attr_name}': {repr(val)} (type: {type(val)})")
                # Some files store coefficients in a list/array, extract the first element.
                while isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) == 0:
                        raise ValueError(f"Calibration coefficient {attr_name} is an empty sequence.")
                    val = val[0]
                    print(f"  [get_scalar] Unwrapped to: {repr(val)} (type: {type(val)})")
                # Ensure the final value is a float.
                return float(val)

            # Convert Digital Number (DN) to Radiance. This is the first step for all channels.
            gain = get_scalar('DN_to_Radiance_Gain')
            offset = get_scalar('DN_to_Radiance_Offset')

            print("\n[DEBUG] Processed scalar coefficients:")
            print(f"  - Gain: {gain} (type: {type(gain)})")
            print(f"  - Offset: {offset} (type: {type(offset)})")

            print("\n[DEBUG] Attempting multiplication: radiance = dn_variable * gain + offset")
            radiance = dn_variable * gain + offset
            radiance.attrs['units'] = 'W m-2 sr-1 um-1' # Note: This is radiance per-micrometer
            print("[DEBUG] DN to Radiance conversion successful.")

            # --- Channel-Specific Calibration ---
            channel_type = product_name[:2]  # e.g., 'vi', 'nr', 'sw', 'ir', 'wv'

            if channel_type in ['vi', 'nr']: # Visible and Near-Infrared channels
                # Convert Radiance to Albedo
                c = get_scalar('Radiance_to_Albedo_c')
                fractional_albedo = radiance * c
                # Clip to physically valid range [0, 1] and scale to percent.
                albedo = np.clip(fractional_albedo, 0.0, 1.0) * 100
                albedo.attrs['long_name'] = 'Albedo'
                albedo.attrs['units'] = '%'
                ds['albedo'] = albedo
                print(f"[DEBUG] Albedo: {albedo.min().compute().item():.2f}% to {albedo.max().compute().item():.2f}%")


            elif channel_type in ['sw', 'ir', 'wv']: # Thermal channels
                # Convert Radiance to Brightness Temperature via inverse Planck function.
                # The function is only defined for positive radiance.
                positive_radiance = radiance.where(radiance > 0)

                # Get physical constants from dataset attributes.
                cval = get_scalar('light_speed')
                kval = get_scalar('Boltzmann_constant_k')
                hval = get_scalar('Plank_constant_h')

                # Calculate wavenumber (wn) in m^-1.
                # The channel_center_wavelength is in micrometers (um).
                wn = (10000.0 / get_scalar('channel_center_wavelength')) * 100.0
                print(f"[DEBUG] Calculated Wavenumber (wn): {wn} (m^-1)")

                # Convert radiance to units required by the Planck function formula.
                radiance_for_planck = positive_radiance * 1e-5
                print(f"[DEBUG] Radiance for Planck (radiance_for_planck): {radiance_for_planck.min().compute().item():.3e} to {radiance_for_planck.max().compute().item():.3e}")

                # Inverse Planck function to get Effective Temperature (Teff)
                e1 = (2.0 * hval * cval * cval) * np.power(wn, 3.0)
                e2 = radiance_for_planck
                # Guard against division by zero or log of non-positive numbers.
                e2 = e2.where(e2 > 0)
                term_in_log = (e1 / e2) + 1.0
                term_in_log = term_in_log.where(term_in_log > 0)

                teff = ((hval * cval / kval) * wn) / np.log(term_in_log)
                print(f"[DEBUG] Effective Temperature (teff): {teff.min().compute().item():.2f} K to {teff.max().compute().item():.2f} K")


                # Apply standard polynomial correction to get Brightness Temperature (Tbb)
                c0 = get_scalar('Teff_to_Tbb_c0')
                c1_t = get_scalar('Teff_to_Tbb_c1')
                c2_t = get_scalar('Teff_to_Tbb_c2')

                tbb = c2_t * teff**2 + c1_t * teff + c0
                tbb.attrs['long_name'] = 'Brightness Temperature'
                tbb.attrs['units'] = 'K'
                ds['brightness_temperature'] = tbb
                print(f"[DEBUG] Brightness Temperature (tbb): {tbb.min().compute().item():.2f} K to {tbb.max().compute().item():.2f} K")

            print(f"--- Calibration successful for {product_name} ---")
            return ds

        except Exception as e:
            # If any step fails, log the error and return the original data.
            print(f"\n---!!! CALIBRATION FAILED for {product_name} !!!---", file=sys.stderr)
            print(f"ERROR MESSAGE: {e}", file=sys.stderr)
            print("\n--- Full Stack Trace ---", file=sys.stderr)
            traceback.print_exc()
            print("------------------------", file=sys.stderr)
            print("Returning uncalibrated data.", file=sys.stderr)
            return ds

    def _add_geolocation(self, ds, area, product_name):
        """
        Adds latitude and longitude coordinates to the xarray.Dataset using the
        geolocation module.

        Args:
            ds (xarray.Dataset): The xarray dataset to add coordinates to.
            area (str): The observation area (e.g., 'fd', 'la').
            product_name (str): The product identifier (e.g., 'ir087'), used
                                to derive the resolution.

        Returns:
            xarray.Dataset: The dataset with 'latitude' and 'longitude' coordinates added.
        """
        def _get_resolution_from_product(prod_name):
            """
            Internal helper to derive resolution string (e.g., '0.5km', '1km', '2km')
            from the product name (e.g., 'ir087').
            This mapping is based on typical GK2A channel resolutions.
            """
            # This mapping must be kept in sync with actual GK2A resolutions
            # and the AMI_FILENAME_RE in GK2ADefs.
            if 'vi004' in prod_name or 'vi005' in prod_name or 'vi006' in prod_name:
                return '0.5km'
            elif 'vi008' in prod_name or 'vi016' in prod_name:
                return '1km'
            elif 'ir' in prod_name or 'wv' in prod_name or 'sw' in prod_name:
                # All thermal/water vapor/shortwave channels are 2km in standard products
                return '2km'
            return 'unknown'

        resolution = _get_resolution_from_product(product_name.lower())
        if resolution == 'unknown':
            print(f"Warning: Could not determine resolution for product '{product_name}'. Geolocation may not work correctly.", file=sys.stderr)
            return ds

        # Determine the dimension names for columns and lines in the dataset.
        # Common names are 'x', 'y' or 'col', 'row', or derived from image shape.
        # It's assumed 'image_pixel_values' is the main data variable,
        # and its dimensions define the image shape.
        data_var_name = 'image_pixel_values'
        if data_var_name not in ds.data_vars:
            print(f"Error: Expected data variable '{data_var_name}' not found in dataset. Cannot geolocate.", file=sys.stderr)
            return ds

        column_dim_name = None
        line_dim_name = None

        # Try common dimension names for image data
        if 'x' in ds[data_var_name].dims and 'y' in ds[data_var_name].dims:
            column_dim_name = 'x'
            line_dim_name = 'y'
        # Check if the dimensions are named 'column' and 'line'
        elif 'column' in ds[data_var_name].dims and 'line' in ds[data_var_name].dims:
            column_dim_name = 'column'
            line_dim_name = 'line'
        # Fallback for generic 2D data: assume last two dimensions are (line, column)
        elif len(ds[data_var_name].dims) >= 2:
            line_dim_name = ds[data_var_name].dims[-2]
            column_dim_name = ds[data_var_name].dims[-1]
        else:
            print(f"Warning: Could not identify suitable 2D dimensions for geolocation in data variable '{data_var_name}'. Expected 'x'/'y' or 'column'/'line', or at least 2 dimensions. Skipping geolocation.", file=sys.stderr)
            return ds

        columns = np.arange(ds[data_var_name].sizes[column_dim_name])
        lines = np.arange(ds[data_var_name].sizes[line_dim_name])

        # Create a meshgrid for all pixels to get all (column, line) pairs
        cols_mesh, lines_mesh = np.meshgrid(columns, lines)

        # Convert to lat/lon using the geolocation module
        # Note: sat_time_utc parameter is no longer explicitly passed to geolocation.to_latlon,
        # as it's not needed for the GEOS projection used by FD/LA.
        latitudes, longitudes = geolocation.to_latlon(
            cols_mesh, lines_mesh, area, resolution
        )

        # Add as new coordinates to the xarray dataset.
        # Ensure the dimensions match the original data's spatial dimensions.
        ds = ds.assign_coords(latitude=((line_dim_name, column_dim_name), latitudes))
        ds = ds.assign_coords(longitude=((line_dim_name, column_dim_name), longitudes))

        return ds


    @staticmethod
    def _generate_s3_prefixes(sensor, product, start_time, end_time, area=None):
        """
        Generates the S3 prefixes to search for files within a time range.

        The GK-2A S3 bucket is organized by .../YYYYMM/DD/HH/. This method
        iterates through each hour in the specified time range and yields the
        corresponding S3 prefix.

        Args:
            sensor (str): The sensor name (e.g., 'ami').
            product (str): The product identifier (e.g., 'vi004').
            start_time (datetime): The beginning of the time range.
            end_time (datetime): The end of the time range.
            area (str): The geographic area (e.g., 'fd', 'ela', 'la').

        Yields:
            str: An S3 prefix string for a specific hour.

        Raises:
            ValueError: If 'area' is not provided.
        """
        if not area:
            raise ValueError("The 'area' parameter is required for L1B data.")
        # Note: The AMI_L1B_PREFIX from GK2ADefs is no longer directly used here as per user's provided code structure.
        # The structure of the S3 path derived from the user's provided code is:
        # {SENSOR.upper()}/L1B/{AREA.upper()}/{YYYYMM}/{DD}/{HH}/gk2a_{SENSOR.lower()}_le1b_{PRODUCT.lower()}_{AREA.lower()}
        base_prefix_path = f"{sensor.upper()}/L1B/{area.upper()}/"
        current_time = start_time
        while current_time <= end_time:
            time_path = current_time.strftime(f"%Y%m/%d/%H/")
            filename_prefix = f"gk2a_{sensor.lower()}_le1b_{product.lower()}_{area.lower()}"
            full_prefix = f"{base_prefix_path}{time_path}{filename_prefix}"
            yield full_prefix
            current_time += timedelta(hours=1)

    def _find_files(self, sensor, product, start_time, end_time, area=None):
        """
        Finds all relevant data files within a specified time range.

        This method queries S3 using the prefixes generated by
        _generate_s3_prefixes, filters the results to match the exact time
        window, parses the filenames, and returns a sorted list of file metadata.

        Args:
            sensor (str): The sensor name.
            product (str): The product identifier.
            start_time (datetime): The start of the time range.
            end_time (datetime): The end of the time range.
            area (str): The geographic area.

        Returns:
            list[dict]: A sorted list of dictionaries, where each dictionary
                        contains parsed metadata for a found file. Returns an
                        empty list if no files are found.
        """
        all_files = []
        # Generate prefixes using the static method
        prefixes_to_search = list(self._generate_s3_prefixes(sensor, product, start_time, end_time, area))
        for prefix in prefixes_to_search:
            s3_objects = self.s3_utils.list_files_in_prefix(prefix)
            for obj in s3_objects:
                filename = os.path.basename(obj['Key'])
                parsed_data = GK2ADefs.parse_filename(filename)

                if not parsed_data:
                    continue

                file_time = parsed_data.get('datetime')
                if not file_time:
                    continue
                    
                # Final check to ensure file is within the precise time range.
                if not (start_time <= file_time <= end_time):
                    continue

                parsed_data['s3_key'] = obj['Key']
                all_files.append(parsed_data)

        # Sort files chronologically.
        all_files.sort(key=lambda x: x.get('datetime', datetime.min))
        return all_files
    
    def _load_as_xarray(self, s3_path):
        """
        Loads a single NetCDF file from S3 into an xarray.Dataset.

        Args:
            s3_path (str): The full S3 path (e.g., 's3://bucket/key').

        Returns:
            xr.Dataset: The loaded dataset. Returns None if loading fails.
        """
        try:
            # Use the s3fs filesystem object to open a remote file handle.
            remote_file = self.s3_utils.fs.open(s3_path, 'rb')
            # Open the dataset with automatic chunking for Dask integration.
            ds = xr.open_dataset(remote_file, chunks='auto')
            return ds
        except Exception as e:
            print(f"ERROR: Could not open S3 object {s3_path} as xarray.Dataset: {e}", file=sys.stderr)
            return None

    def get_data(self, sensor, product, area, query_type='latest', target_time=None, start_time=None, end_time=None, calibrate=False, geolocation_enabled=False):
        """
        Fetches GK-2A data based on specified criteria.

        This is the main public method of the class. It provides three ways to
        query for data: 'latest', 'nearest', and 'range'.

        Args:
            sensor (str): The sensor name, e.g., 'ami'.
            product (str): The product identifier, e.g., 'vi004', 'ir133'.
            area (str): The geographical area, e.g., 'fd' (Full Disk),
                        'ela' (Extended Local Area), 'la' (Local Area).
            query_type (str, optional): The type of query. One of 'latest',
                'nearest', or 'range'. Defaults to 'latest'.
            target_time (datetime, optional): The target time for a 'nearest'
                query. Required if query_type is 'nearest'.
            start_time (datetime, optional): The start time for a 'range'
                query. Required if query_type is 'range'.
            end_time (datetime, optional): The end time for a 'range'
                query. Required if query_type is 'range'.
            calibrate (bool, optional): If True, applies calibration to the
                loaded data. Defaults to False.
            geolocation_enabled (bool, optional): If True, adds latitude and
                longitude coordinates to the dataset. Defaults to False.

        Returns:
            xr.Dataset or None: An xarray.Dataset containing the requested data.
            The dataset will have a 'time' dimension. Returns None if no data
            is found or if an error occurs.

        Raises:
            ValueError: If `query_type` is unknown or if required time
            parameters for 'nearest' or 'range' queries are missing.
        """
        if query_type == 'latest':
            # Search the last 2 hours for the most recent file.
            search_end = datetime.utcnow()
            search_start = search_end - timedelta(hours=2)
            found_files = self._find_files(sensor, product, search_start, search_end, area)
            if not found_files: return None
            
            latest_file = found_files[-1]
            ds = self._load_as_xarray(f"s3://{GK2ADefs.S3_BUCKET}/{latest_file['s3_key']}")
            if not ds: return None
            
            if calibrate:
                ds = self._calibrate(ds, product)
            if geolocation_enabled: # Call geolocation if flag is True
                ds = self._add_geolocation(ds, area, product) # Pass product to _add_geolocation
            # Add a time dimension to the dataset for consistency.
            return ds.expand_dims(time=[latest_file['datetime']])

        elif query_type == 'nearest':
            if not target_time: raise ValueError("`target_time` is required for 'nearest' query.")
            # Search a 4-hour window around the target time.
            search_start = target_time - timedelta(hours=2)
            search_end = target_time + timedelta(hours=2)
            found_files = self._find_files(sensor, product, search_start, search_end, area)
            if not found_files: return None

            # Find the file with the minimum time difference from the target.
            nearest_file = min(found_files, key=lambda x: abs(x['datetime'] - target_time))
            ds = self._load_as_xarray(f"s3://{GK2ADefs.S3_BUCKET}/{nearest_file['s3_key']}")
            if not ds: return None
            
            if calibrate:
                ds = self._calibrate(ds, product)
            if geolocation_enabled: # Call geolocation if flag is True
                ds = self._add_geolocation(ds, area, product) # Pass product to _add_geolocation
            return ds.expand_dims(time=[nearest_file['datetime']])

        elif query_type == 'range':
            if not start_time or not end_time: raise ValueError("`start_time` and `end_time` are required for 'range' query.")
            found_files = self._find_files(sensor, product, start_time, end_time, area)
            if not found_files: return None

            datasets = []
            for file_info in found_files:
                ds = self._load_as_xarray(f"s3://{GK2ADefs.S3_BUCKET}/{file_info['s3_key']}")
                if ds:
                    # To ensure concatenation works, we might need to handle datasets
                    # with multiple variables. Here we simplify by just taking the main one.
                    # A more robust implementation might select variables explicitly.
                    main_var = next(iter(ds.data_vars))
                    ds_clean = ds[[main_var]]
                    if calibrate:
                        ds_clean = self._calibrate(ds, product)
                    if geolocation_enabled: # Call geolocation if flag is True
                        ds_clean = self._add_geolocation(ds_clean, area, product) # Pass product to _add_geolocation
                    # Add time dimension before appending for concatenation.
                    datasets.append(ds_clean.expand_dims(time=[file_info['datetime']]))
            
            if not datasets: return None
            # Combine all individual datasets into a single timeseries dataset.
            return xr.concat(datasets, dim='time')

        else:
            raise ValueError(f"Unknown query_type: '{query_type}'. Must be 'latest', 'nearest', or 'range'.")
