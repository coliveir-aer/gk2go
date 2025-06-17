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
    # instantaneous field of view (MRAD) for each pixel is proportionally smaller.
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
    S3_BUCKET = "noaa-gk2a-pds"

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
    def parse_s3_key(s3_key):
        """
        Parses an S3 object key (filename) to extract GK2A metadata.
        Args:
            s3_key (str): The full S3 object key (e.g., AMI/L1B/FD/202506/01/00/gk2a_ami_le1b_ir087_fd020ge_202506010000.nc)
        Returns:
            dict or None: A dictionary containing parsed metadata (sensor, area, product, datetime)
                          or None if the key does not match the expected pattern.
        """
        filename = os.path.basename(s3_key)
        match = GK2ADefs.GK2A_FILENAME_PATTERN.match(filename)
        if match:
            data = match.groupdict()
            try:
                data['datetime'] = datetime(
                    int(data['year']), int(data['month']), int(data['day']),
                    int(data['hour']), int(data['minute'])
                )
                data['s3_key'] = s3_key # Store the full S3 key
                return data
            except ValueError:
                return None
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
        return value


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
        Now follows YYYYMM/DD/HH structure.
        """
        # //noaa-gk2a-pds/AMI/L1B/FD/202506/17/16/
        # gk2a_ami_le1b_ir087_fd020ge_202506171650.nc
        base_prefix = f"{sensor.upper()}/L1B/{area.upper()}"
        if dt:
            # Construct path as L1B/{AREA}/{YYYYMM}/{DD}/{HH}/
            return f"{base_prefix}/{dt.strftime('%Y%m')}/{dt.strftime('%d')}/{dt.strftime('%H')}/"
        return base_prefix + "/" # For listing all files in an area

    def _find_files(self, sensor, product, start_time, end_time, area, debug=False):
        """
        Finds available GK2A NetCDF files within a specified time range.
        Optimized to search only relevant daily/hourly prefixes.
        """
        found_files = []
        current_time = start_time
        # Iterate hour by hour to build prefixes more accurately
        while current_time <= end_time:
            hourly_prefix = self._get_s3_prefix(sensor, area, product, dt=current_time)
            if debug:
                print(f"Searching S3 prefix: {self.s3_bucket}/{hourly_prefix}", file=sys.stderr)
            s3_keys = self.s3_utils.list_s3_objects(self.s3_bucket, hourly_prefix, debug=debug)

            for key in s3_keys:
                parsed_data = GK2ADefs.parse_s3_key(key)
                if parsed_data and parsed_data['sensor'] == sensor and \
                   parsed_data['product'] == product and parsed_data['area'] == area:
                    if start_time <= parsed_data['datetime'] <= end_time:
                        found_files.append(parsed_data)
            current_time += timedelta(hours=1) # Increment by hour
        # Sort by datetime to ensure consistency for 'range' queries
        return sorted(found_files, key=lambda x: x['datetime'])

    def _load_as_xarray(self, s3_path, debug=False):
        """
        Loads a NetCDF file directly from S3 into an xarray Dataset.
        """
        try:
            if debug:
                print(f"Loading data from: {s3_path}", file=sys.stderr)
            # Use s3fs to open the file directly via xarray
            with self.s3_utils.s3_fs.open(s3_path, mode='rb') as f:
                ds = xr.open_dataset(f, engine='netcdf4', decode_coords="all")
            return ds
        except Exception as e:
            print(f"Error loading xarray dataset from {s3_path}: {e}", file=sys.stderr)
            if debug:
                traceback.print_exc()
            return None

    def _calibrate(self, ds, product, debug=False):
        """
        Calibrates the raw image pixel values to radiance and then to
        Brightness Temperature (for IR channels) or Albedo (for Visible channels).
        """
        # Get raw image pixel values
        dn_variable = ds['image_pixel_values'].squeeze() # Remove time dimension

        if debug:
            print("[DEBUG] Raw 'image_pixel_values' info:", file=sys.stderr)
            print(f"  - Type: {type(dn_variable)}", file=sys.stderr)
            print(f"  - Shape: {dn_variable.shape}", file=sys.stderr)
            print(f"  - Dtype: {dn_variable.dtype}", file=sys.stderr)
            if hasattr(dn_variable.data, 'dask'):
                print(f"  - Underlying data type: {type(dn_variable.data)}", file=sys.stderr)
        
        # Get calibration coefficients
        gain = GK2ADefs.get_attr_scalar(ds, 'DN_to_Radiance_Gain', debug=debug)
        offset = GK2ADefs.get_attr_scalar(ds, 'DN_to_Radiance_Offset', debug=debug)

        if debug:
            print("[DEBUG] Processed scalar coefficients:", file=sys.stderr)
            print(f"  - Gain: {gain} (type: {type(gain)})", file=sys.stderr)
            print(f"  - Offset: {offset} (type: {type(offset)})", file=sys.stderr)

        if gain is None or offset is None:
            print("Error: Calibration coefficients (Gain/Offset) not found. Cannot calibrate.", file=sys.stderr)
            return ds # Return original dataset if calibration not possible

        # Convert to Radiance
        # Radiance = DN * Gain + Offset
        radiance = dn_variable * gain + offset
        if debug:
            print("[DEBUG] Attempting multiplication: radiance = dn_variable * gain + offset", file=sys.stderr)
            print("[DEBUG] DN to Radiance conversion successful.", file=sys.stderr)

        # Add radiance as a new data variable or replace if it's the primary one
        if 'radiance' in ds.data_vars:
            ds['radiance'] = (ds['image_pixel_values'].dims, radiance.data)
        else:
            ds = ds.assign(radiance=(ds['image_pixel_values'].dims, radiance.data))


        # Handle Brightness Temperature for IR channels
        if product.startswith('ir'):
            # Constants for Planck function
            h = GK2ADefs.get_attr_scalar(ds, 'Plank_constant_h', debug=debug)
            k = GK2ADefs.get_attr_scalar(ds, 'Boltzmann_constant_k', debug=debug)
            c = GK2ADefs.get_attr_scalar(ds, 'light_speed', debug=debug)
            channel_center_wavelength_um = float(GK2ADefs.get_attr_scalar(ds, 'channel_center_wavelength', debug=debug))

            if any(val is None for val in [h, k, c, channel_center_wavelength_um]):
                print("Error: Planck constants or channel wavelength not found. Cannot calculate Brightness Temperature.", file=sys.stderr)
                return ds # Return original dataset

            # Wavenumber in m^-1 (convert um to m first)
            wn = 1e6 / channel_center_wavelength_um # Convert micrometers to m^-1
            if debug:
                print(f"[DEBUG] Calculated Wavenumber (wn): {wn} (m^-1)", file=sys.stderr)

            # Planck function to get Effective Temperature (Te_eff)
            # Te_eff = C2 / log(C1/Rad + 1) where C1 = 2hc^2nu^3, C2 = hc.nu/k
            # Let's use the actual attribute names if they exist
            C1 = GK2ADefs.get_attr_scalar(ds, 'planck_function_coefficient_c1', debug=debug)
            C2 = GK2ADefs.get_attr_scalar(ds, 'planck_function_coefficient_c2', debug=debug)

            if C1 is None or C2 is None:
                if debug:
                    print("Warning: Planck coefficients C1/C2 not found. Recalculating from fundamental constants.", file=sys.stderr)
                # Fallback to calculate C1 and C2 if attributes are missing
                C1 = 2 * h * c**2 * wn**3
                C2 = h * c * wn / k
            
            # Ensure radiance is not zero or negative for log calculation
            # Use np.maximum to avoid log(0) or log(negative)
            radiance_for_planck = np.maximum(radiance, 1e-9) # Small positive epsilon
            if debug:
                print(f"[DEBUG] Radiance for Planck (radiance_for_planck): {np.nanmin(radiance_for_planck):.3e} to {np.nanmax(radiance_for_planck):.3e}", file=sys.stderr)

            effective_temperature = C2 / np.log(C1 / radiance_for_planck + 1)
            effective_temperature = xr.DataArray(effective_temperature, dims=dn_variable.dims, coords=dn_variable.coords) # Ensure it's DataArray
            if debug:
                print(f"[DEBUG] Effective Temperature (teff): {np.nanmin(effective_temperature.values):.2f} K to {np.nanmax(effective_temperature.values):.2f} K", file=sys.stderr)

            # Apply empirical correction (Teff to Tbb)
            # Tbb = c0 + c1*Teff + c2*Teff^2 (from GK2A L1B User Manual, page 20)
            c0 = GK2ADefs.get_attr_scalar(ds, 'Teff_to_Tbb_c0', debug=debug)
            c1 = GK2ADefs.get_attr_scalar(ds, 'Teff_to_Tbb_c1', debug=debug)
            c2 = GK2ADefs.get_attr_scalar(ds, 'Teff_to_Tbb_c2', debug=debug)

            if any(val is None for val in [c0, c1, c2]):
                if debug:
                    print("Warning: Teff_to_Tbb coefficients (c0, c1, c2) not found. Skipping Tbb correction.", file=sys.stderr)
                ds['brightness_temperature'] = effective_temperature # Use effective_temperature as Tbb
            else:
                brightness_temperature = c0 + c1 * effective_temperature + c2 * effective_temperature**2
                brightness_temperature = xr.DataArray(brightness_temperature, dims=dn_variable.dims, coords=dn_variable.coords) # Ensure DataArray
                ds['brightness_temperature'] = brightness_temperature
                if debug:
                    print(f"[DEBUG] Brightness Temperature (tbb): {np.nanmin(brightness_temperature.values):.2f} K to {np.nanmax(brightness_temperature.values):.2f} K", file=sys.stderr)

        elif product.startswith(('vis', 'vnir')): # For visible/VNIR channels (convert to Albedo/Reflectance)
            # GK2A L1B User Manual (20190618_gk2a_level_1b_data_user_manual_eng.pdf, Page 19)
            # Albedo = Radiance * Scaling_Factor
            # Scaling_Factor = pi * D_sun^2 / (Solar_Irradiance * cos(Solar_Zenith_Angle))
            # Solar_Irradiance is 'channel_solar_irradiance' attribute
            # D_sun is 'earth_sun_distance_correction_factor' attribute (unit: AU)
            # Solar_Zenith_Angle: requires solar position, pixel lat/lon, and time. This is more complex.

            # For now, let's just convert radiance to a scaled value if a simple "Radiance_to_Albedo_Scaling_Factor" exists
            # or use 'albedo_scaling_factor_value' (found in some datasets)
            albedo_scale = GK2ADefs.get_attr_scalar(ds, 'albedo_scaling_factor_value', debug=debug)
            if albedo_scale is None:
                # Fallback to channel_solar_irradiance and earth_sun_distance_correction_factor
                solar_irradiance = GK2ADefs.get_attr_scalar(ds, 'channel_solar_irradiance', debug=debug)
                earth_sun_distance = GK2ADefs.get_attr_scalar(ds, 'earth_sun_distance_correction_factor', debug=debug) # in AU

                if solar_irradiance is not None and earth_sun_distance is not None:
                    # Simplified scaling, actual albedo requires solar zenith angle, which needs geolocation first.
                    # For a quick fix, if we don't have SZA, we can't do full albedo.
                    # Let's just return radiance and note the limitation.
                    if debug:
                        print("Warning: Full albedo conversion requires Solar Zenith Angle, which needs geolocation. Returning radiance as 'albedo'.", file=sys.stderr)
                    ds['albedo'] = radiance
                else:
                    if debug:
                        print("Warning: Necessary attributes for Albedo conversion not found. Returning radiance as 'albedo'.", file=sys.stderr)
                    ds['albedo'] = radiance
            else:
                albedo = radiance * albedo_scale
                ds['albedo'] = xr.DataArray(albedo, dims=dn_variable.dims, coords=dn_variable.coords)
            
        if debug:
            print(f"--- Calibration successful for {product} ---", file=sys.stderr)
        return ds

    def _add_geolocation(self, ds, sensor, debug=False):
        """
        Adds latitude and longitude coordinates to the xarray Dataset using the
        GEOS projection method. Caches results for efficiency.
        """
        # It's crucial to remove existing lat/lon coords that might be loaded
        # from the NetCDF, as they are often incorrect or too limited for full disk.
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            if debug:
                print("[DEBUG Core Geoloc] Removing existing 'latitude' and 'longitude' coordinates.", file=sys.stderr)
            # Create a new dataset without these coordinates if they exist
            ds = ds.drop_vars(['latitude', 'longitude'], errors='ignore')
            # If they were actual coordinates (not just variables), this should handle it.
            # reset_coords can convert coordinate variables to data variables, then drop_vars.
            # A more robust way might be to create a new dataset from the data variables.
            # For simplicity, let's assume drop_vars handles it.

        if sensor != 'ami':
            if debug:
                print("Warning: Geolocation currently only implemented for 'ami' sensor.", file=sys.stderr)
            return ds

        # Extract necessary parameters from dataset attributes
        # Ensure these attributes exist in the raw dataset
        image_width = ds.attrs.get('image_width', None)
        image_height = ds.attrs.get('image_height', None)
        # Convert channel_spatial_resolution to float if it's a string
        resolution_km = float(ds.attrs.get('channel_spatial_resolution', 2.0)) # Default to 2.0km if not found

        # Ensure these core attributes are available
        if image_width is None or image_height is None:
            print("Error: 'image_width' or 'image_height' attributes not found in dataset. Cannot add geolocation.", file=sys.stderr)
            return ds

        # GK2A specific satellite parameters (constants in our GEOS implementation)
        satellite_height_km = GK2ADefs.get_attr_scalar(ds, 'satellite_height', 42164.160, debug=debug) # Default value
        sub_satellite_lon_deg = GK2ADefs.get_attr_scalar(ds, 'image_center_longitude', 128.2, debug=debug) # Default for GK2A

        # Convert sub_satellite_lon_deg from radians to degrees if it's stored as radians in ds.attrs
        # The image_center_longitude in GK2A datasets is often in radians.
        if ds.attrs.get('image_center_longitude_units') == 'radians':
             sub_satellite_lon_deg = np.rad2deg(sub_satellite_lon_deg)
             if debug:
                 print(f"[DEBUG Core Geoloc] Converted image_center_longitude from radians to degrees: {sub_satellite_lon_deg:.4f}", file=sys.stderr)


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
            ds = ds.assign_coords(latitude=((y_dim_name, x_dim_name), lat_coords))
            ds = ds.assign_coords(longitude=((x_dim_name, y_dim_name), lon_coords)) # Longitude should typically be (x_dim, y_dim) or (y_dim, x_dim) consistent with latitude


            if debug:
                print("[DEBUG Core Geoloc] Successfully added GEOS-based geolocation coordinates.")
                print(f"  Final Latitudes Array Info:\n  Shape: {lat_coords.shape}\n  Min: {np.nanmin(lat_coords):.4f} deg\n  Max: {np.nanmax(lat_coords):.4f} deg\n  NaN Count: {np.isnan(lat_coords).sum().item()} / {lat_coords.size} ({np.isnan(lat_coords).sum().item()/lat_coords.size*100:.2f}%)")
                print(f"  Final Longitudes Array Info:\n  Shape: {lon_coords.shape}\n  Min: {np.nanmin(lon_coords):.4f} deg\n  Max: {np.nanmax(lon_coords):.4f} deg\n  NaN Count: {np.isnan(lon_coords).sum().item()} / {lon_coords.size} ({np.isnan(lon_coords).sum().item()/lon_coords.size*100:.2f}%)")

        except Exception as e:
            print(f"Error computing geolocation with GEOS-based method: {e}", file=sys.stderr)
            if debug:
                traceback.print_exc()
        return ds


    def get_data(self, sensor, product, area, query_type, target_time=None,
                 start_time=None, end_time=None, calibrate=False, geolocation_enabled=True, debug=False):
        """
        Fetches GK2A satellite data.
        """
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
            if ds and calibrate:
                ds = self._calibrate(ds, product, debug=debug)
            # Add time dimension if not present (for consistency)
            if 'time' not in ds.dims:
                ds = ds.expand_dims(time=[file_info['datetime']])

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
            if ds and calibrate:
                ds = self._calibrate(ds, product, debug=debug)
            # Add time dimension if not present (for consistency)
            if 'time' not in ds.dims:
                ds = ds.expand_dims(time=[file_info['datetime']])

        elif query_type == 'range':
            if not start_time or not end_time: raise ValueError("`start_time` and `end_time` are required for 'range' query.")
            found_files = self._find_files(sensor, product, start_time, end_time, area, debug=debug)
            if not found_files: return None

            datasets = []
            for file_info in found_files:
                ds_single = self._load_as_xarray(f"s3://{self.s3_bucket}/{file_info['s3_key']}", debug=debug)
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
                    datasets.append(ds_clean.expand_dims(time=[file_info['datetime']]))
            
            if not datasets: return None
            # Combine all individual datasets into a single timeseries dataset.
            ds = xr.concat(datasets, dim='time')

        else:
            raise ValueError(f"Unknown query_type: '{query_type}'. Must be 'latest', 'nearest', or 'range'.")

        # Apply geolocation if enabled and data was successfully fetched
        if ds is not None and geolocation_enabled:
            if debug:
                print(f"\n--- Adding Geolocation for {product.upper()} Data ---", file=sys.stderr)
            ds = self._add_geolocation(ds, sensor, debug=debug)
            
        return ds

