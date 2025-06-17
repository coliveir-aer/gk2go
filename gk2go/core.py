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

Classes:
    GK2ADefs: Defines constants and static methods for GK-2A data.
    S3Utils: Provides utility functions for interacting with the S3 bucket.
    Gk2aDataFetcher: The main class for finding and loading GK-2A data.
"""

import os
import re
import sys
import math
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


class EarthConstants:
    """Defines constants for Earth's dimensions for geolocation calculations."""
    RP = 6356.75231414  # Radius of Earth across Poles (WGS84 semi-minor axis) in km
    RE = 6378.137       # Radius of Earth across Equator (WGS84 semi-major axis) in km
    ECC_SQ = 0.00669438440 # 1st eccentricity squared (from GK2A readme_latlon.txt)

# --- GEOS Projection Core Function ---
_geolocation_cache = {} # Module-level cache for computed lat/lon grids

def _compute_geos_latlon(image_width, image_height, resolution_km, satellite_height_km, sub_satellite_lon_deg, debug=False):
    """
    Computes geodetic latitude and longitude arrays from image pixel coordinates
    using the GEOS projection model, precisely adopting a standard geostationary
    satellite imaging geometry implementation.
    Caches results to avoid re-computation for the same grid parameters.

    Args:
        image_width (int): Number of pixels in the X dimension (columns).
        image_height (int): Number of pixels in the Y dimension (rows).
        resolution_km (float): Spatial resolution of the channel in kilometers (e.g., 0.5, 1.0, 2.0).
        satellite_height_km (float): Height of the satellite from Earth's center in km.
        sub_satellite_lon_deg (float): Sub-satellite longitude in degrees.
        debug (bool): If True, print debug messages.

    Returns:
        tuple: (lat_deg, lon_deg) 2D NumPy arrays of latitudes and longitudes in degrees.
               Returns NaN for points outside the Earth's view.
    """
    # Cache key based on parameters that define the grid geometry
    cache_key = (image_width, image_height, resolution_km, satellite_height_km, sub_satellite_lon_deg)
    if cache_key in _geolocation_cache:
        if debug:
            print(f"[DEBUG Geoloc Cache] Loading from cache for resolution {resolution_km}km, dims {image_width}x{image_height}.")
        return _geolocation_cache[cache_key]

    # Dynamically determine MRAD (microradians per pixel) based on resolution scaling.
    # The baseline MRAD (56.0e-6) is assumed for 2km resolution for a 5500x5500 image.
    # This implies a total angular span of 5500 * 56.0e-6 = 0.308 radians for the 5500x5500 image.
    # For GK2A, image dimensions scale with resolution (e.g., 1km is 11000x11000).
    # Thus, the MRAD (angular resolution per pixel) should scale inversely with the resolution value (km).
    # This maintains a consistent total angular field of view regardless of pixel count.

    # Calculate base MRAD as if for a 2km resolution image
    MRAD_2KM_BASE = 56.0e-6 # This is the base angular resolution per pixel for 2km resolution

    # Adjust MRAD based on the current image's resolution relative to the 2km base.
    # A smaller `resolution_km` value (higher spatial resolution) means the sensor's
    # instantaneous field of view (MRAD) is proportionally smaller for each pixel.
    if resolution_km == 0.5:
        MRAD = MRAD_2KM_BASE / 4.0
    elif resolution_km == 1.0:
        MRAD = MRAD_2KM_BASE / 2.0
    elif resolution_km == 2.0:
        MRAD = MRAD_2KM_BASE
    else:
        # Fallback or raise error for unsupported resolutions
        if debug:
            print(f"Warning: Unsupported channel spatial resolution '{resolution_km}km'. Using MRAD for 2km resolution as fallback.", file=sys.stderr)
        MRAD = MRAD_2KM_BASE

    XP = image_width / 2.0
    YP = image_height / 2.0
    SLON_RAD = np.deg2rad(sub_satellite_lon_deg)

    REQ = EarthConstants.RE
    RPOL = EarthConstants.RP
    ECC_SQ = EarthConstants.ECC_SQ

    # Create 2D arrays for pixel indices
    pixel_x_indices = np.arange(image_width, dtype=np.float64)
    pixel_y_indices = np.arange(image_height, dtype=np.float64)
    pixel_x_grid, pixel_y_grid = np.meshgrid(pixel_x_indices, pixel_y_indices)

    # 1. Shift pixel coordinates to have 0 in the middle and convert to radians
    # This directly mirrors the initial steps of the GEOS pixel-to-lat/lon conversion.
    x_rad = (pixel_x_grid - XP) * MRAD
    y_rad = (pixel_y_grid - YP) * MRAD

    # 2. Setup quadratic equation coefficients 'a', 'b', 'c' for ray intersection with ellipsoid
    # These formulas are derived from the GEOS projection model.
    a_val = np.sin(x_rad)**2 + np.cos(x_rad)**2 * (np.cos(y_rad)**2 + (REQ**2/RPOL**2) * np.sin(y_rad)**2)
    b_val = -2 * satellite_height_km * np.cos(x_rad) * np.cos(y_rad)
    c_val = satellite_height_km**2 - REQ**2

    # 3. Solve the quadratic equation for 'rs' (distance along ray from satellite to Earth point)
    discriminant = b_val**2 - 4 * a_val * c_val
    rs = np.full_like(discriminant, np.nan, dtype=np.float64)

    # Only calculate 'rs' where the discriminant is non-negative (i.e., ray intersects Earth)
    valid_discriminant_mask = (discriminant >= 0)
    if np.any(valid_discriminant_mask):
        # Taking the negative root for the physically meaningful intersection (closer to satellite)
        rs_temp = (-b_val[valid_discriminant_mask] - np.sqrt(discriminant[valid_discriminant_mask])) / (2 * a_val[valid_discriminant_mask])
        rs[valid_discriminant_mask] = rs_temp

    # If rs is NaN, the pixel is off-Earth, so lat/lon will remain NaN.

    # 4. Calculate 'sx', 'sy', 'sz' (components of the vector from satellite to Earth point in satellite coordinates)
    sx = rs * np.cos(x_rad) * np.cos(y_rad)
    sy = rs * np.sin(x_rad)
    sz = rs * np.cos(x_rad) * np.sin(y_rad)

    # 5. Convert to geodetic latitude and longitude
    lat_deg = np.full_like(rs, np.nan, dtype=np.float64)
    lon_deg = np.full_like(rs, np.nan, dtype=np.float64)

    # Identify points where the calculated Earth-point coordinates are valid for conversion
    # Add a small epsilon to avoid division by zero or near-zero for denominator
    epsilon = 1e-9
    valid_conversion_mask = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(sz) & \
                            (np.abs(satellite_height_km - sx) > epsilon) & \
                            (((satellite_height_km - sx)**2 + sy**2) > epsilon)

    if np.any(valid_conversion_mask):
        # Apply longitude conversion
        lon_rad_calc = SLON_RAD + np.arctan2(sy[valid_conversion_mask], (satellite_height_km - sx)[valid_conversion_mask])
        lon_deg[valid_conversion_mask] = np.rad2deg(lon_rad_calc)

        # Apply latitude conversion
        # Note: (REQ**2/RPOL**2) is equivalent to 1 / (1 - ECC_SQ)
        lat_rad_calc = np.arctan2((REQ**2/RPOL**2) * sz[valid_conversion_mask], np.sqrt((satellite_height_km - sx)[valid_conversion_mask]**2 + sy[valid_conversion_mask]**2))
        lat_deg[valid_conversion_mask] = np.rad2deg(lat_rad_calc)
    
    # Store in cache before returning
    _geolocation_cache[cache_key] = (lat_deg, lon_deg)
    if debug:
        print(f"[DEBUG Geoloc Cache] Stored in cache for resolution {resolution_km}km, dims {image_width}x{image_height}.")
    return lat_deg, lon_deg


class GK2ADefs:
    """
    Defines constants and static methods for GK-2A data, including S3 bucket details,
    product types, and helper functions for parsing filenames and attributes.
    """
    # Regex to parse the standard GK-2A AMI LE1B filename format.
    # It captures key metadata components like satellite, sensor, product, area,
    # resolution, and timestamp.
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
    # The public S3 bucket hosting the NOAA GK-2A PDS.
    S3_BUCKET = "noaa-gk2a-pds"
    # The AWS region where the S3 bucket is located.
    AWS_REGION = "us-east-1"

    # GK2A sensor (AMI) product types and areas
    AMI_PRODUCTS = [
        'vi004',
        'vi005',
        'vi006',
        'vi008',
        'nr013',
        'nr016',
        'sw038',
        'wv063',
        'wv069',
        'wv073',
        'ir087',
        'ir096',
        'ir105',
        'ir112',
        'ir123',
        'ir133',
    ]
    AMI_AREAS = ['fd', 'la']

    # Regex pattern to parse GK2A Level 1B filenames
    # Example: gk2a_ami_le1b_ir087_fd020ge_202506010000.nc
    GK2A_FILENAME_PATTERN = re.compile(
        r'gk2a_(?P<sensor>ami|ksem)_le1b_(?P<product>[a-z0-9]+)_'
        r'(?P<area>[a-z]+)(?P<res>\d{3})ge_'
        r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
        r'(?P<hour>\d{2})(?P<minute>\d{2})\.nc$'
    )

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

    @staticmethod
    def get_attr_scalar(ds, attr_name, default=None, debug=False):
        """
        Safely retrieves a scalar attribute from an xarray Dataset,
        handling cases where the attribute might be an array or missing.
        """
        if debug:
            print(f"  [get_scalar] Initial value for '{attr_name}': {ds.attrs.get(attr_name, default)} (type: {type(ds.attrs.get(attr_name, default))})")
        value = ds.attrs.get(attr_name, default)
        if isinstance(value, (np.ndarray, xr.DataArray)):
            if value.size == 1:
                return value.item() # Get scalar value from single-element array
            else:
                if debug:
                    print(f"Warning: Attribute '{attr_name}' is an array with size > 1. Returning first element.", file=sys.stderr)
                return value.flatten()[0].item()
        return value # Return value directly, don't force float conversion here


class S3Utils:
    """
    Provides utility functions for interacting with an S3 bucket, specifically
    for listing and filtering objects.
    """
    def __init__(self):
        # Configure botocore to use UNSIGNED requests for public S3 buckets
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.s3_fs = s3fs.S3FileSystem(anon=True) # Used by xarray for direct S3 access

    def list_s3_objects(self, bucket, prefix, debug=False):
        """
        Lists objects in an S3 bucket with a given prefix.
        Args:
            bucket (str): The S3 bucket name.
            prefix (str): The prefix to filter objects by.
            debug (bool): If True, print debug messages.
        Returns:
            list: A list of S3 object keys (full paths) that match the prefix.
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
    """
    Main class for discovering, fetching, and processing GK-2A satellite data.
    """
    def __init__(self):
        self.s3_utils = S3Utils()
        self.s3_bucket = GK2ADefs.S3_BUCKET

    def _get_s3_prefix(self, sensor, area, product, dt=None):
        """
        Constructs the S3 prefix path based on sensor, area, product, and optional datetime.
        Now follows AMI/L1B/FD/YYYYMM/DD/HH structure.
        """
        base_prefix = f"{sensor.upper()}/L1B/{area.upper()}"
        if dt:
            # Construct path as L1B/{AREA}/{YYYYMM}/{DD}/{HH}/
            return f"{base_prefix}/{dt.strftime('%Y%m')}/{dt.strftime('%d')}/{dt.strftime('%H')}/"
        return base_prefix + "/" # For listing all files in an area

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
        base_prefix = f"{sensor.upper()}/L1B/{area.upper()}/"
        current_time = start_time
        while current_time <= end_time:
            time_path = current_time.strftime(f"%Y%m/%d/%H/")
            filename_prefix = f"gk2a_{sensor.lower()}_le1b_{product.lower()}_{area.lower()}"
            full_prefix = f"{base_prefix}{time_path}{filename_prefix}"
            yield full_prefix
            current_time += timedelta(hours=1)

    def _find_files(self, sensor, product, start_time, end_time, area=None, debug=False):
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
        prefixes = list(self._generate_s3_prefixes(sensor, product, start_time, end_time, area))
        for prefix in prefixes:
            s3_objects = self.s3_utils.list_s3_objects(self.s3_bucket, prefix)
            for obj_key in s3_objects: # obj_key is already the string key, not a dict
                filename = os.path.basename(obj_key) # Corrected: Use obj_key directly
                parsed_data = GK2ADefs.parse_filename(filename)

                if not parsed_data:
                    continue

                file_time = parsed_data.get('datetime')
                if not file_time:
                    continue

                # Final check to ensure file is within the precise time range.
                if not (start_time <= file_time <= end_time):
                    continue

                parsed_data['s3_key'] = obj_key # Corrected: Use obj_key directly
                all_files.append(parsed_data)

        # Sort files chronologically.
        all_files.sort(key=lambda x: x.get('datetime', datetime.min))
        return all_files

    def _load_as_xarray(self, s3_path, debug=False):
        """
        Loads a NetCDF file directly from S3 into an xarray Dataset.
        """
        try:
            if debug:
                print(f"Loading data from: {s3_path}", file=sys.stderr)
              
            # Use the s3fs filesystem object to open a remote file handle.
            remote_file = self.s3_utils.s3_fs.open(s3_path, 'rb')
            # Open the dataset with automatic chunking for Dask integration.
            ds = xr.open_dataset(remote_file, chunks='auto')
            return ds
          
            # Use s3fs to open the file directly via xarray
            # Use engine='h5netcdf' as these are NetCDF4 files, and 'scipy' often struggles.
            #with self.s3_utils.s3_fs.open(s3_path, mode='rb') as f:
            #    # Use 'h5netcdf' engine for NetCDF4 files, and allow lazy loading with 'chunks=auto'
            #    ds = xr.open_dataset(f, engine='h5netcdf', decode_coords="all", chunks='auto')
            #    # CRITICAL FIX: Force data to be loaded into memory before 'f' is closed
            #    ds.load() 
            #return ds
        except Exception as e:
            print(f"Error loading xarray dataset from {s3_path}: {e}", file=sys.stderr)
            if debug:
                traceback.print_exc()
            return None

    def _calibrate(self, ds, product_name, debug=False):
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
            debug (bool): If True, print debug messages.

        Returns:
            xr.Dataset: The dataset with a new calibrated data variable
                        ('albedo' or 'brightness_temperature') added. Returns the
                        original dataset if calibration fails.
        """
        if debug:
            print(f"--- Entering Calibration for {product_name} ---", file=sys.stderr)
        try:
            dn_variable = ds['image_pixel_values']

            if debug:
                print(f"[DEBUG] Raw 'image_pixel_values' info:", file=sys.stderr)
                print(f"  - Type: {type(dn_variable)}", file=sys.stderr)
                print(f"  - Shape: {dn_variable.shape}", file=sys.stderr)
                print(f"  - Dtype: {dn_variable.dtype}", file=sys.stderr)
                if hasattr(dn_variable.data, 'dask'):
                    print(f"  - Underlying data type (dask array): {type(dn_variable.data)}", file=sys.stderr)
            
            def get_scalar(attr_name, default=None):
                """
                Helper to robustly extract a scalar float from dataset attributes.
                Handles attributes that might be lists, tuples, or numpy arrays.
                """
                val = ds.attrs.get(attr_name, default)
                if debug:
                    print(f"  [get_scalar] Initial value for '{attr_name}': {repr(val)} (type: {type(val)})", file=sys.stderr)
                
                # Some files store coefficients in a list/array, extract the first element.
                while isinstance(val, (list, tuple, np.ndarray, xr.DataArray)):
                    if len(val) == 0:
                        raise ValueError(f"Calibration coefficient {attr_name} is an empty sequence.")
                    val = val[0]
                    if debug:
                        print(f"  [get_scalar] Unwrapped to: {repr(val)} (type: {type(val)})", file=sys.stderr)
                
                # Ensure the final value is a float.
                return float(val)

            # Convert Digital Number (DN) to Radiance. This is the first step for all channels.
            gain = get_scalar('DN_to_Radiance_Gain')
            offset = get_scalar('DN_to_Radiance_Offset')

            if debug:
                print("\n[DEBUG] Processed scalar coefficients:", file=sys.stderr)
                print(f"  - Gain: {gain} (type: {type(gain)})", file=sys.stderr)
                print(f"  - Offset: {offset} (type: {type(offset)})", file=sys.stderr)

            if gain is None or offset is None:
                print("Error: Calibration coefficients (Gain/Offset) not found. Cannot calibrate.", file=sys.stderr)
                return ds # Return original dataset if calibration not possible

            if debug:
                print("[DEBUG] Attempting multiplication: radiance = dn_variable * gain + offset", file=sys.stderr)
            radiance = dn_variable * gain + offset
            radiance.attrs['units'] = 'W m-2 sr-1 um-1' # Note: This is radiance per-micrometer
            if debug:
                print("[DEBUG] DN to Radiance conversion successful.", file=sys.stderr)
                print(f"[DEBUG] Radiance min/max: {radiance.min().compute().item():.3e} to {radiance.max().compute().item():.3e}", file=sys.stderr)

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
                if debug:
                    print(f"[DEBUG] Albedo: {albedo.min().compute().item():.2f}% to {albedo.max().compute().item():.2f}%", file=sys.stderr)


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
                # Convert micrometers to cm^-1 (10000 / um) then to m^-1 ( * 100)
                wn = (10000.0 / get_scalar('channel_center_wavelength')) * 100.0
                if debug:
                    print(f"[DEBUG] Calculated Wavenumber (wn): {wn} (m^-1)", file=sys.stderr)

                # Convert radiance to units required by the Planck function formula.
                # This 1e-5 scaling factor is critical for GK2A radiance units with SI constants.
                radiance_for_planck = positive_radiance * 1e-5 
                if debug:
                    print(f"[DEBUG] Radiance for Planck (radiance_for_planck): {radiance_for_planck.min().compute().item():.3e} to {radiance_for_planck.max().compute().item():.3e}", file=sys.stderr)

                # Inverse Planck function to get Effective Temperature (Teff)
                # Term 'e1' corresponds to 2hc^2nu^3
                e1 = (2.0 * hval * cval * cval) * np.power(wn, 3.0)
                # Term 'e2' is the adjusted radiance
                e2 = radiance_for_planck
                # Guard against division by zero or log of non-positive numbers.
                e2 = e2.where(e2 > 0) # Ensure no non-positive values for division
                term_in_log = (e1 / e2) + 1.0
                term_in_log = term_in_log.where(term_in_log > 0) # Ensure no non-positive values for log

                teff = ((hval * cval / kval) * wn) / np.log(term_in_log)
                if debug:
                    print(f"[DEBUG] Effective Temperature (teff): {teff.min().compute().item():.2f} K to {teff.max().compute().item():.2f} K", file=sys.stderr)


                # Apply standard polynomial correction to get Brightness Temperature (Tbb)
                c0 = get_scalar('Teff_to_Tbb_c0')
                c1_t = get_scalar('Teff_to_Tbb_c1')
                c2_t = get_scalar('Teff_to_Tbb_c2')

                tbb = c2_t * teff**2 + c1_t * teff + c0
                tbb.attrs['long_name'] = 'Brightness Temperature'
                tbb.attrs['units'] = 'K'
                ds['brightness_temperature'] = tbb
                if debug:
                    print(f"[DEBUG] Brightness Temperature (tbb): {tbb.min().compute().item():.2f} K to {tbb.max().compute().item():.2f} K", file=sys.stderr)

            if debug:
                print(f"--- Calibration successful for {product_name} ---", file=sys.stderr)
            return ds

        except Exception as e:
            # If any step fails, log the error and return the original data.
            print(f"\n---!!! CALIBRATION FAILED for {product_name} !!!---", file=sys.stderr)
            print(f"ERROR MESSAGE: {e}", file=sys.stderr)
            if debug:
                print("\n--- Full Stack Trace ---", file=sys.stderr)
                traceback.print_exc()
                print("------------------------", file=sys.stderr)
            print("Returning uncalibrated data.", file=sys.stderr)
            return ds


    def _add_geolocation(self, ds, sensor, debug=False):
        """
        Adds latitude and longitude coordinates to the xarray Dataset using the
        GEOS projection method. Caches results for efficiency.
        """
        # It's crucial to remove existing lat/lon coords that might be loaded
        # from the NetCDF, as they are often incorrect or too limited for full disk.
        #if 'latitude' in ds.coords and 'longitude' in ds.coords:
        #    if debug:
        #        print("[DEBUG Core Geoloc] Removing existing 'latitude' and 'longitude' coordinates.", file=sys.stderr)
        #    # Create a new dataset without these coordinates if they exist
        #    ds = ds.drop_vars(['latitude', 'longitude'], errors='ignore')

        if sensor != 'ami':
            if debug:
                print("Warning: Geolocation currently only implemented for 'ami' sensor.", file=sys.stderr)
            return ds

        # Extract necessary parameters from dataset attributes
        # Corrected attribute names based on the dataset info provided by user
        image_width = GK2ADefs.get_attr_scalar(ds, 'number_of_columns', None, debug=debug)
        image_height = GK2ADefs.get_attr_scalar(ds, 'number_of_lines', None, debug=debug)
        # Convert channel_spatial_resolution to float if it's a string
        resolution_km = float(GK2ADefs.get_attr_scalar(ds, 'channel_spatial_resolution', 2.0, debug=debug)) # Default to 2.0km if not found

        # Ensure these core attributes are available
        if image_width is None or image_height is None:
            print("Error: 'number_of_columns' or 'number_of_lines' attributes not found in dataset. Cannot add geolocation.", file=sys.stderr)
            return ds

        # GK2A specific satellite parameters (constants in our GEOS implementation)
        # 'nominal_satellite_height' is in meters, convert to km
        satellite_height_m = GK2ADefs.get_attr_scalar(ds, 'nominal_satellite_height', 42164000.0, debug=debug) # Default value in meters
        satellite_height_km = satellite_height_m / 1000.0 # Convert to kilometers
        
        # image_center_longitude (sub_longitude) is the satellite longitude, given in radians
        # Need to convert to degrees before passing to _compute_geos_latlon
        sub_satellite_lon_rad = GK2ADefs.get_attr_scalar(ds, 'image_center_longitude', None, debug=debug)
        if sub_satellite_lon_rad is None:
             # Fallback to 'sub_longitude' if 'image_center_longitude' is not found
             sub_satellite_lon_rad = GK2ADefs.get_attr_scalar(ds, 'sub_longitude', np.deg2rad(128.2), debug=debug) # Default for GK2A in radians
        
        sub_satellite_lon_deg = np.rad2deg(sub_satellite_lon_rad)
        if debug:
             print(f"[DEBUG Core Geoloc] Sub-satellite longitude: {sub_satellite_lon_deg:.4f} degrees (from {sub_satellite_lon_rad} radians)", file=sys.stderr)


        if debug:
            print(f"[DEBUG Core Geoloc] Parameters for _compute_geos_latlon:", file=sys.stderr)
            print(f"  image_width: {image_width}", file=sys.stderr)
            print(f"  image_height: {image_height}", file=sys.stderr)
            print(f"  resolution_km: {resolution_km}", file=sys.stderr)
            print(f"  satellite_height_km: {satellite_height_km}", file=sys.stderr)
            print(f"  sub_satellite_lon_deg: {sub_satellite_lon_deg}", file=sys.stderr)


        try:
            lat_coords, lon_coords = _compute_geos_latlon(
                image_width=image_width,
                image_height=image_height,
                resolution_km=resolution_km,
                satellite_height_km=satellite_height_km,
                sub_satellite_lon_deg=sub_satellite_lon_deg,
                debug=debug
            )

            # Add these as coordinates to the dataset
            # Ensure the dimensions match the image data dimensions
            # We assume image_pixel_values is the primary data variable
            if 'image_pixel_values' in ds.data_vars:
                # Use actual dimensions names from the image data array
                y_dim_name = ds['image_pixel_values'].dims[-2]
                x_dim_name = ds['image_pixel_values'].dims[-1]
            else:
                # Fallback if image_pixel_values is not present or has different dims
                y_dim_name = 'dim_image_y'
                x_dim_name = 'dim_image_x'
                if debug:
                    print(f"Warning: 'image_pixel_values' not found. Using default dimension names: {y_dim_name}, {x_dim_name}", file=sys.stderr)

            # Assign coordinates. xarray handles broadcasting if lat_coords/lon_coords are 2D.
            # Make sure the time dimension is preserved if the dataset originally had it.
            # The lat/lon coords are computed for the 2D image, so we add them at the image dimension level.
            # Corrected: Both latitude and longitude should typically use the same (y_dim_name, x_dim_name) order
            ds = ds.assign_coords(latitude=((y_dim_name, x_dim_name), lat_coords))
            ds = ds.assign_coords(longitude=((y_dim_name, x_dim_name), lon_coords))


            if debug:
                print("[DEBUG Core Geoloc] Successfully added GEOS-based geolocation coordinates.", file=sys.stderr)
                print(f"  Final Latitudes Array Info:\n  Shape: {lat_coords.shape}\n  Min: {np.nanmin(lat_coords):.4f} deg\n  Max: {np.nanmax(lat_coords):.4f} deg\n  NaN Count: {np.isnan(lat_coords).sum().item()} / {lat_coords.size} ({np.isnan(lat_coords).sum().item()/lat_coords.size*100:.2f}%)", file=sys.stderr)
                print(f"  Final Longitudes Array Info:\n  Shape: {lon_coords.shape}\n  Min: {np.nanmin(lon_coords):.4f} deg\n  Max: {np.nanmax(lon_coords):.4f} deg\n  NaN Count: {np.isnan(lon_coords).sum().item()} / {lon_coords.size} ({np.isnan(lon_coords).sum().item()/lon_coords.size*100:.2f}%)", file=sys.stderr)

        except Exception as e:
            print(f"Error computing geolocation with GEOS-based method: {e}", file=sys.stderr)
            if debug:
                traceback.print_exc()
        return ds


    def get_data(self, sensor, product, area, query_type, target_time=None,
                 start_time=None, end_time=None, calibrate=True, geolocation_enabled=False, debug=False):
        """
        Fetches GK2A satellite data.
        """
        ds = None # Initialize ds to None for scope
        file_info = None # Initialize file_info

        if query_type == 'latest':
            # For 'latest', search a recent time window (e.g., last 24 hours)
            # to find the most recent file.
            _end_time = datetime.utcnow()
            _start_time = _end_time - timedelta(hours=24) # Look back 24 hours
            found_files = self._find_files(sensor, product, _start_time, _end_time, area, debug=debug)
            
            if not found_files:
                print(f"No '{product}' data found for '{area}' area in the last 24 hours.", file=sys.stderr)
                return None
            
            # Get the very latest file
            file_info = found_files[-1]
            ds = self._load_as_xarray(f"s3://{self.s3_bucket}/{file_info['s3_key']}", debug=debug)
            
        elif query_type == 'nearest':
            if not target_time: raise ValueError("`target_time` is required for 'nearest' query.")
            
            # For 'nearest', search a window around the target time (e.g., +/- 6 hours)
            _start_time = target_time - timedelta(hours=6)
            _end_time = target_time + timedelta(hours=6)
            found_files = self._find_files(sensor, product, _start_time, _end_time, area, debug=debug)
            
            if not found_files:
                print(f"No '{product}' data found for '{area}' area near {target_time.strftime('%Y-%m-%d %H:%M UTC')}.", file=sys.stderr)
                return None
            
            # Find the file closest to target_time
            file_info = min(found_files, key=lambda x: abs(x['datetime'] - target_time))
            if debug:
                print(f"Nearest file found: {file_info['s3_key']}", file=sys.stderr)
            ds = self._load_as_xarray(f"s3://{self.s3_bucket}/{file_info['s3_key']}", debug=debug)
            
        elif query_type == 'range':
            if not start_time or not end_time: raise ValueError("`start_time` and `end_time` are required for 'range' query.")
            found_files = self._find_files(sensor, product, start_time, end_time, area, debug=debug)
            if not found_files: return None

            datasets = []
            for file_info_single in found_files: # Renamed to avoid conflict with outer file_info
                ds_single = self._load_as_xarray(f"s3://{self.s3_bucket}/{file_info_single['s3_key']}", debug=debug)
                if ds_single:
                    if calibrate:
                        ds_single = self._calibrate(ds_single, product, debug=debug)
                    
                    # To ensure concatenation works, we might need to handle datasets
                    # with multiple variables. Here we simplify by just taking the main one.
                    # A more robust implementation might select variables explicitly.
                    # For consistency, ensure brightness_temperature or albedo is primary if calibrated
                    if 'brightness_temperature' in ds_single.data_vars:
                        main_var = 'brightness_temperature'
                    elif 'albedo' in ds_single.data_vars:
                        main_var = 'albedo'
                    else:
                        main_var = next(iter(ds_single.data_vars)) # Fallback to first data variable

                    ds_clean = ds_single[[main_var]]
                    # Add time dimension before appending for concatenation.
                    datasets.append(ds_clean.expand_dims(time=[file_info_single['datetime']]))
            
            if not datasets: return None
            # Combine all individual datasets into a single timeseries dataset.
            ds = xr.concat(datasets, dim='time')

        else:
            raise ValueError(f"Unknown query_type: '{query_type}'. Must be 'latest', 'nearest', or 'range'.")

        # Common post-loading steps
        if ds is not None:
            if calibrate and query_type != 'range': # Only calibrate once if not a range query (already done per-file in range)
                ds = self._calibrate(ds, product, debug=debug)
            
            # Add time dimension if not present (for consistency in 'latest' and 'nearest' after calibration)
            if query_type in ['latest', 'nearest'] and 'time' not in ds.dims and file_info:
                ds = ds.expand_dims(time=[file_info['datetime']])

            # Apply geolocation if enabled and data was successfully fetched
            if geolocation_enabled:
                if debug:
                    print(f"\n--- Adding Geolocation for {product.upper()} Data ---", file=sys.stderr)
                ds = self._add_geolocation(ds, sensor, debug=debug)
            
        return ds
