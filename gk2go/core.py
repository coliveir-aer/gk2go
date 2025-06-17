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
- On-the-fly calibration: *requires external calibration coefficients to be provided
  as attributes within the xarray.Dataset itself*.
- Uses xarray for powerful, labeled, multi-dimensional data handling.
- Robust error handling for S3 interactions and calibration processes.
- **NEW**: Optional addition of latitude and longitude coordinates to the
  xarray Dataset using detailed projection models for 'FD' and 'LA' areas.

Classes:
    GK2ADefs: Defines constants and static methods for GK2A data.
    S3Utils: Provides utility functions for interacting with the S3 bucket.
    Gk2aDataFetcher: The main class for finding and loading GK2A data.
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
    Defines constants and static methods related to GK2A satellite data.
    """
    S3_BUCKET = "noaa-ghe-pds" # S3 bucket name
    AMI_L1B_PREFIX = "GKO/AMI/L1B" # Prefix for AMI Level 1B data

    # AMI Channels and their resolutions. This is a simplified mapping;
    # actual resolution can depend on area.
    # Ref: 20190618_gk2a_level_1b_data_user_manual_eng.pdf (Page 5)
    AMI_CHANNELS_RESOLUTION = {
        'vi004': '0.5km', 'vi005': '0.5km', 'vi006': '0.5km', 'vi008': '1km',
        'vi016': '1km', 'ir039': '2km', 'wv062': '2km', 'wv069': '2km',
        'wv073': '2km', 'ir087': '2km', 'ir096': '2km', 'ir105': '2km',
        'ir112': '2km', 'ir123': '2km', 'ir133': '2km', 'sw037': '2km'
    }
  
    @staticmethod
    def get_product_prefix(sensor, product, area):
        """Constructs the S3 object key prefix for a given sensor, product, and area."""
        # Example: GKO/AMI/L1B/FD/2019/07/20/AMI_L1B_FD020_20190720_000000.nc
        return f"{GK2ADefs.AMI_L1B_PREFIX}/{area.upper()}"

    @staticmethod
    def get_resolution(product_name):
        """Returns the resolution string for a given product."""
        return GK2ADefs.AMI_CHANNELS_RESOLUTION.get(product_name.lower(), 'unknown')
      

class S3Utils:
    """
    Utility functions for interacting with the AWS S3 bucket.
    """
    def __init__(self):
        # Configure botocore to use anonymous access for public S3 buckets
        self.config = Config(signature_version=UNSIGNED)
        self.s3_client = boto3.client("s3", config=self.config)
        self.s3_filesystem = s3fs.S3FileSystem(anon=True)

    def list_files(self, prefix):
        """
        Lists all objects within a given S3 prefix.

        Args:
            prefix (str): The S3 prefix to list objects from.

        Returns:
            list: A list of S3 object keys (full paths).
        """
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=GK2ADefs.S3_BUCKET, Prefix=prefix)
            keys = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        keys.append(obj["Key"])
            return keys
        except ClientError as e:
            print(f"Error listing S3 files for prefix {prefix}: {e}")
            return []

    def get_latest_file(self, prefix):
        """
        Retrieves the latest (most recently modified) file from an S3 prefix.

        Args:
            prefix (str): The S3 prefix to search.

        Returns:
            dict or None: A dictionary containing 's3_key' and 'last_modified'
                          datetime object, or None if no files are found.
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=GK2ADefs.S3_BUCKET,
                Prefix=prefix,
                MaxKeys=1000 # Fetch a reasonable number to find the latest
            )
            if "Contents" in response:
                # Sort by LastModified timestamp in descending order
                latest_object = max(response["Contents"], key=lambda obj: obj["LastModified"])
                return {
                    "s3_key": latest_object["Key"],
                    "last_modified": latest_object["LastModified"],
                }
            return None
        except ClientError as e:
            print(f"Error getting latest S3 file for prefix {prefix}: {e}")
            return None


class Gk2aDataFetcher:
    """
    Main class for discovering and loading GK2A satellite data.
    """
    def __init__(self):
        self.s3_utils = S3Utils()

    def _parse_filename(self, s3_key):
        """
        Parses an S3 object key to extract datetime and product information.
        Assumes GK2A L1B file naming convention:
        GK2_AMI_L1B_{AREA}{CHANNEL}_{YYYYMMDD}_{HHMMSS}.nc

        Args:
            s3_key (str): Full S3 object key (path to the .nc file).

        Returns:
            dict or None: A dictionary with 'datetime', 'area', 'product', and 's3_key',
                          or None if parsing fails.
        """
        # Example: GKO/AMI/L1B/FD/2019/07/20/AMI_L1B_FD020_20190720_000000.nc
        # Updated regex to correctly capture 'la' in filename like gk2a_ami_le1b_ir087_la020ge_202406151406.nc
        match = re.search(r'AMI_L1B_([A-Za-z]{2,3})(\d{3}[A-Za-z]{0,2})_(\d{8})_(\d{6})\.nc$', s3_key)
        if match:
            area = match.group(1).lower()
            # Channel code might include letters for local areas, e.g., '020ge'
            channel_id = match.group(2)
            date_str = match.group(3)
            time_str = match.group(4)
            try:
                dt_obj = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                return {
                    "datetime": dt_obj,
                    "area": area,
                    "channel_id": channel_id, # Store channel_id to help infer product if needed
                    "s3_key": s3_key,
                }
            except ValueError:
                return None
        return None

    def _find_files(self, sensor, product, start_time, end_time, area):
        """
        Finds relevant files within a time range for a given product and area.
        This method constructs the S3 prefix path based on year, month, day.
        """
        found_files = []
        current_time = start_time
        while current_time <= end_time:
            # Construct S3 prefix for the specific day
            daily_prefix = (
                f"{GK2ADefs.get_product_prefix(sensor, product, area)}/"
                f"{current_time.strftime('%Y/%m/%d')}/"
            )
            all_files_in_day = self.s3_utils.list_files(daily_prefix)

            for s3_key in all_files_in_day:
                file_info = self._parse_filename(s3_key)
                if file_info and product in s3_key and area in s3_key: # Basic check
                    if start_time <= file_info["datetime"] <= end_time:
                        found_files.append(file_info)
            current_time += timedelta(days=1)

        # Sort files by datetime
        found_files.sort(key=lambda x: x["datetime"])
        return found_files

    def _load_as_xarray(self, s3_path):
        """
        Loads a NetCDF4 file from S3 directly into an xarray.Dataset.
        Uses s3fs for memory-efficient streaming.

        Args:
            s3_path (str): Full S3 path to the .nc file (e.g., "s3://bucket/key.nc").

        Returns:
            xarray.Dataset or None: The loaded dataset, or None if an error occurs.
        """
        try:
            with self.s3_utils.s3_filesystem.open(s3_path, mode="rb") as f:
                # Use h5netcdf engine for NetCDF4 files
                ds = xr.open_dataset(f, engine="h5netcdf", chunks={})
                return ds
        except Exception as e:
            print(f"Error loading {s3_path} into xarray: {e}")
            traceback.print_exc()
            return None

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

    def _add_geolocation(self, ds, area, resolution, sat_time_utc=None):
        """
        Adds latitude and longitude coordinates to the xarray.Dataset using the
        geolocation module.

        Args:
            ds (xarray.Dataset): The xarray dataset to add coordinates to.
            area (str): The observation area (e.g., 'fd', 'la').
            resolution (str): The resolution of the data (e.g., '0.5km', '1km', '2km').
            sat_time_utc (datetime.datetime, optional): Satellite observation time in UTC.
                                                       *Not used in current GEOS-only implementation for FD/LA*,
                                                       but kept for consistency in function signature.

        Returns:
            xarray.Dataset: The dataset with 'latitude' and 'longitude' coordinates added.
        """
        # Determine the dimension names for columns and lines in the dataset.
        data_var_name = list(ds.data_vars.keys())[0] # Get the first data variable

        column_dim = None
        line_dim = None

        if 'x' in ds[data_var_name].dims and 'y' in ds[data_var_name].dims:
            column_dim = 'x'
            line_dim = 'y'
        elif 'column' in ds[data_var_name].dims and 'line' in ds[data_var_name].dims:
            column_dim = 'column'
            line_dim = 'line'
        elif len(ds[data_var_name].dims) >= 2: # Fallback for generic 2D data (last two dims are spatial)
            line_dim = ds[data_var_name].dims[-2]
            column_dim = ds[data_var_name].dims[-1]
        else:
            print(f"Warning: Could not identify suitable 2D dimensions for geolocation in data variable '{data_var_name}' (expected 'x'/'y' or 'column'/'line'). Skipping geolocation.")
            return ds

        columns = ds[column_dim].values
        lines = ds[line_dim].values

        # Create a meshgrid for all pixels to get all (column, line) pairs
        cols_mesh, lines_mesh = np.meshgrid(columns, lines)

        # Convert to lat/lon using the geolocation module
        # Note: sat_time_utc is no longer directly used by `geolocation.to_latlon` for GEOS,
        # but the parameter is kept in the signature for broader compatibility if needed.
        latitudes, longitudes = geolocation.to_latlon(
            cols_mesh, lines_mesh, area, resolution
        )

        # Add as new coordinates to the xarray dataset.
        # Ensure the dimensions match the original data's spatial dimensions.
        ds = ds.assign_coords(latitude=((line_dim, column_dim), latitudes))
        ds = ds.assign_coords(longitude=((line_dim, column_dim), longitudes))

        return ds

    def get_data(self, sensor, product, area, query_type,
                 target_time=None, start_time=None, end_time=None,
                 calibrate=False, geolocation_enabled=False): # Added geolocation_enabled option
        """
        Fetches GK2A satellite data.

        Args:
            sensor (str): The sensor name (e.g., 'ami').
            product (str): The data product name (e.g., 'ir105').
            area (str): The observation area ('fd' or 'la').
            query_type (str): Type of query: 'latest', 'nearest', 'range'.
            target_time (datetime.datetime, optional): Target time for 'nearest' query.
                                                      Required for 'nearest' query.
            start_time (datetime.datetime, optional): Start time for 'range' query.
                                                     Required for 'range' query.
            end_time (datetime.datetime, optional): End time for 'range' query.
                                                   Required for 'range' query.
            calibrate (bool): If True, calibrate raw data to physical units.
            geolocation_enabled (bool): If True, add latitude and longitude coordinates
                                        to the dataset. Defaults to False.

        Returns:
            xarray.Dataset: The requested satellite data. Returns None if data not found
                            or an error occurs.
        """
        if sensor.lower() != 'ami':
            raise ValueError("Only 'ami' sensor is supported at this time.")

        # Validate requested area
        if area.lower() not in ['fd', 'la']:
            raise ValueError(f"Unsupported observation area: '{area}'. Must be 'fd' or 'la'.")

        # Determine resolution based on product for geolocation
        resolution = GK2ADefs.get_resolution(product)
        if resolution == 'unknown':
            print(f"Warning: Unknown resolution for product '{product}'. Geolocation may not work correctly.")

        if query_type == 'latest':
            today_prefix = (
                f"{GK2ADefs.get_product_prefix(sensor, product, area)}/"
                f"{datetime.utcnow().strftime('%Y/%m/%d')}/"
            )
            latest_file_info = self.s3_utils.get_latest_file(today_prefix)

            if latest_file_info:
                parsed_info = self._parse_filename(latest_file_info['s3_key'])
                if parsed_info:
                    ds = self._load_as_xarray(f"s3://{GK2ADefs.S3_BUCKET}/{parsed_info['s3_key']}")
                    if ds:
                        if calibrate:
                            ds = self._calibrate(ds, product)
                        if geolocation_enabled:
                            # sat_time_utc is no longer used by GEOS projection, can pass None or actual time.
                            ds = self._add_geolocation(ds, area, resolution, sat_time_utc=parsed_info['datetime'])
                        return ds
                print(f"Error parsing latest file info or loading data for {today_prefix}.")
            else:
                print(f"No latest file found for prefix: {today_prefix}")
            return None

        elif query_type == 'nearest':
            if not target_time:
                raise ValueError("`target_time` is required for 'nearest' query.")

            search_start_time = target_time - timedelta(days=1)
            search_end_time = target_time + timedelta(days=1)
            found_files = self._find_files(sensor, product, search_start_time, search_end_time, area)

            if not found_files:
                print(f"No files found near {target_time} for product {product}, area {area}.")
                return None

            nearest_file_info = min(found_files, key=lambda x: abs(x["datetime"] - target_time))

            ds = self._load_as_xarray(f"s3://{GK2ADefs.S3_BUCKET}/{nearest_file_info['s3_key']}")
            if ds:
                if calibrate:
                    ds = self._calibrate(ds, product)
                if geolocation_enabled:
                    # sat_time_utc is no longer used by GEOS projection, can pass None or actual time.
                    ds = self._add_geolocation(ds, area, resolution, sat_time_utc=nearest_file_info['datetime'])
                return ds.expand_dims(time=[nearest_file_info['datetime']])
            return None

        elif query_type == 'range':
            if not start_time or not end_time:
                raise ValueError("`start_time` and `end_time` are required for 'range' query.")

            found_files = self._find_files(sensor, product, start_time, end_time, area)
            if not found_files:
                print(f"No files found in range {start_time} to {end_time} for product {product}, area {area}.")
                return None

            datasets = []
            for file_info in found_files:
                ds = self._load_as_xarray(f"s3://{GK2ADefs.S3_BUCKET}/{file_info['s3_key']}")
                if ds:
                    main_var = list(ds.data_vars.keys())[0]
                    ds_processed = ds[[main_var]]
                    ds_processed[main_var].attrs = ds[main_var].attrs

                    if calibrate:
                        ds_processed = self._calibrate(ds_processed, product)
                    if geolocation_enabled:
                        # sat_time_utc is no longer used by GEOS projection, can pass None or actual time.
                        ds_processed = self._add_geolocation(ds_processed, area, resolution, sat_time_utc=file_info['datetime'])

                    datasets.append(ds_processed.expand_dims(time=[file_info['datetime']]))

            if not datasets:
                return None
            return xr.concat(datasets, dim='time')

        else:
            raise ValueError(f"Unknown query_type: '{query_type}'. Must be 'latest', 'nearest', or 'range'.")

