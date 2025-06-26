"""A module for fetching and radiometrically calibrating GK-2A L1B data.

This library provides a high-level interface to access Level 1B data from
the Geostationary Korea Multi-Purpose Satellite-2A (GK-2A) Advanced
Meteorological Imager (AMI) sensor.

The primary class, Gk2aDataFetcher, handles the following workflow:
  - Discovering files on S3 based on user queries.
  - Lazily loading NetCDF data into memory-efficient xarray.Datasets.
  - Downsampling high-resolution visible/NIR imagery to a standard 2km grid
    to ensure memory safety and stability.
  - Performing radiometric calibration to convert raw digital numbers into
    Albedo or Brightness Temperature.
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

class GK2ADefs:
    """Defines constants and static methods for GK-2A data processing."""
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
    HIGH_RES_PRODUCTS = ['vi004', 'vi005', 'vi006', 'vi008', 'nr013', 'nr016']

    @staticmethod
    def parse_filename(filename):
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
        value = ds.attrs.get(attr_name, default)
        if isinstance(value, (np.ndarray, xr.DataArray)):
            return value.item() if value.size == 1 else value.flatten()[0].item()
        return value

class S3Utils:
    """A collection of utility methods for interacting with AWS S3."""
    def __init__(self):
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.s3_fs = s3fs.S3FileSystem(anon=True)

    def list_s3_objects(self, bucket, prefix):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        return [obj["Key"] for page in pages if "Contents" in page for obj in page["Contents"]]

    def download_file(self, bucket, key, local_path):
        """Downloads a file from S3 to a local path."""
        try:
            self.s3_client.download_file(bucket, key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading {key} from S3: {e}", file=sys.stderr)
            return False

class Gk2aDataFetcher:
    """The main interface for fetching and processing GK-2A L1B data."""
    def __init__(self):
        self.s3_utils = S3Utils()
        self.s3_bucket = GK2ADefs.S3_BUCKET

    def _find_files(self, sensor, product, start_time, end_time, area):
        base_prefix = f"{sensor.upper()}/L1B/{area.upper()}/"
        current_time = start_time
        all_files = []
        while current_time <= end_time:
            prefix = f"{base_prefix}{current_time.strftime('%Y%m/%d/%H/')}"
            for obj_key in self.s3_utils.list_s3_objects(self.s3_bucket, prefix):
                if product in os.path.basename(obj_key):
                    parsed = GK2ADefs.parse_filename(os.path.basename(obj_key))
                    if parsed and start_time <= parsed['datetime'] <= end_time:
                        parsed['s3_key'] = obj_key
                        all_files.append(parsed)
            current_time += timedelta(hours=1)
        all_files.sort(key=lambda x: x['datetime'])
        return all_files

    def _load_as_xarray(self, path, debug=False):
        """Loads a NetCDF file into an xarray.Dataset from a local path or S3."""
        try:
            if debug: print(f"Loading data from: {path}")
            if path.startswith("s3://"):
                remote_file = self.s3_utils.s3_fs.open(path, 'rb')
                return xr.open_dataset(remote_file, engine='h5netcdf', chunks='auto')
            else:
                return xr.open_dataset(path, engine='h5netcdf', chunks='auto')
        except Exception as e:
            print(f"Error loading xarray dataset from {path}: {e}", file=sys.stderr)
            if debug: traceback.print_exc()
            return None

    def _calibrate(self, ds, product_name, debug=False):
        """Performs radiometric calibration on the xarray dataset."""
        if debug: print(f"--- Calibrating {product_name} ---")
        try:
            pixel_da = ds['image_pixel_values']
            pixel_da_filtered = pixel_da.where(pixel_da <= GK2ADefs.DQF_ERROR_THRESHOLD)
            valid_bits = int(GK2ADefs.get_attr_scalar(ds, 'valid_bits', 16))
            bit_mask = (2**valid_bits) - 1
            processed_pixels = np.bitwise_and(pixel_da_filtered.astype(int), bit_mask).astype(np.float32)
            
            gain = GK2ADefs.get_attr_scalar(ds, 'DN_to_Radiance_Gain')
            offset = GK2ADefs.get_attr_scalar(ds, 'DN_to_Radiance_Offset')
            radiance = processed_pixels * gain + offset
            ds['radiance'] = radiance

            channel_type = product_name[:2]
            if channel_type in ['vi', 'nr']:
                c = GK2ADefs.get_attr_scalar(ds, 'Radiance_to_Albedo_c')
                ds['albedo'] = (radiance * c).clip(0.0, 1.0) * 100
            elif channel_type in ['sw', 'ir', 'wv']:
                radiance_pos = radiance.where(radiance > 0)
                wavelength = float(GK2ADefs.get_attr_scalar(ds, 'channel_center_wavelength'))
                cval = GK2ADefs.get_attr_scalar(ds, 'light_speed')
                kval = GK2ADefs.get_attr_scalar(ds, 'Boltzmann_constant_k')
                hval = GK2ADefs.get_attr_scalar(ds, 'Plank_constant_h')
                wn = (10000.0 / wavelength) * 100.0
                e1 = (2.0 * hval * cval**2) * (wn**3)
                teff = ((hval * cval / kval) * wn) / np.log((e1 / (radiance_pos * 1e-5)) + 1.0)
                c0 = GK2ADefs.get_attr_scalar(ds, 'Teff_to_Tbb_c0')
                c1 = GK2ADefs.get_attr_scalar(ds, 'Teff_to_Tbb_c1')
                c2 = GK2ADefs.get_attr_scalar(ds, 'Teff_to_Tbb_c2')
                ds['brightness_temperature'] = c2 * teff**2 + c1 * teff + c0
            return ds
        except Exception as e:
            print(f"CALIBRATION FAILED: {e}", file=sys.stderr)
            return ds

    def _save_dataset(self, ds, local_path, debug=False):
        """Saves an xarray dataset to a NetCDF file."""
        try:
            if debug: print(f"Saving dataset to: {local_path}")
            ds.to_netcdf(local_path)
            return True
        except Exception as e:
            print(f"Error saving dataset to {local_path}: {e}", file=sys.stderr)
            return False

    def get_data(self, sensor, product, area, query_type, target_time=None,
                 start_time=None, end_time=None, calibrate=True,
                 download=False, save_calibrated=False, load_xarray=True,
                 download_dir='.', debug=False):
        """
        Fetches and processes GK-2A L1B data.

        Args:
            sensor (str): The satellite sensor (e.g., 'ami').
            product (str): The data product (e.g., 'vi008').
            area (str): The geographical area (e.g., 'fd' for full disk).
            query_type (str): Type of query: 'latest', 'nearest', or 'range'.
            target_time (datetime, optional): Required for 'nearest' query.
            start_time (datetime, optional): Required for 'range' query.
            end_time (datetime, optional): Required for 'range' query.
            calibrate (bool): Whether to perform radiometric calibration. Defaults to True.
            download (bool): If True, downloads the NetCDF file to `download_dir`. Defaults to False.
            save_calibrated (bool): If True, saves the calibrated dataset as NetCDF. Defaults to False.
            load_xarray (bool): If True, loads data into an xarray.Dataset. Defaults to True.
            download_dir (str): Directory to download raw files to. Defaults to current directory.
            debug (bool): If True, prints debugging information. Defaults to False.

        Returns:
            xr.Dataset or None: The processed xarray dataset, or None if no data found.
        """
        
        # Validate input flags
        if not load_xarray and calibrate:
            print("ERROR: Calibration cannot occur unless data is loaded into xarray. "
                  "Please set 'load_xarray' to True if you want to calibrate.", file=sys.stderr)
            return None
        
        # Ensure download directory exists
        if download_dir and download:
            os.makedirs(download_dir, exist_ok=True)

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
            print(f"No '{product}' data found for the query.")
            return None

        datasets = []
        for file_info in found_files:
            s3_full_path = f"s3://{self.s3_bucket}/{file_info['s3_key']}"
            local_raw_filename = os.path.basename(file_info['s3_key'])
            local_raw_path = os.path.join(download_dir, local_raw_filename) if download_dir else local_raw_filename
            
            ds = None
            if os.path.exists(local_raw_path):
                print(f"File already exists locally: {local_raw_path}. Loading from local disk.")
                if load_xarray:
                    ds = self._load_as_xarray(local_raw_path, debug=debug)
            elif download:
                print(f"Downloading file to: {local_raw_path}")
                if self.s3_utils.download_file(self.s3_bucket, file_info['s3_key'], local_raw_path):
                    if load_xarray:
                        ds = self._load_as_xarray(local_raw_path, debug=debug)
                else:
                    print(f"Failed to download raw file: {local_raw_path}", file=sys.stderr)
            else:
                # If not downloading and not exists locally, just confirm existence in S3
                print(f"File exists on S3: {s3_full_path}. Not downloading or loading into xarray as requested.")
                continue # Move to the next file or finish

            if ds is None and load_xarray: # If loading into xarray was requested but failed or didn't happen from local/downloaded
                # Fallback to S3 direct load if loading was requested but no local file was loaded
                print(f"Attempting to load from S3 directly: {s3_full_path}")
                ds = self._load_as_xarray(s3_full_path, debug=debug)
            
            if ds is None: # If after all attempts, ds is still None, skip this file
                continue

            if product in GK2ADefs.HIGH_RES_PRODUCTS:
                dims = ds['image_pixel_values'].dims
                original_height = len(ds[dims[0]])
                if original_height > 5500:
                    factor = original_height // 5500
                    if debug: print(f"Decimating high-res imagery (factor {factor}) to 2km grid.")
                    coarsen_dims = {dims[0]: factor, dims[1]: factor}
                    ds = ds.coarsen(coarsen_dims, boundary="trim").mean()
            
            if calibrate:
                ds = self._calibrate(ds, product, debug=debug)
            
            ds = ds.expand_dims(time=[file_info['datetime']])
            datasets.append(ds)

            # Handle save_calibrated flag
            if save_calibrated:
                calibrated_filename = f"calibrated_{local_raw_filename}"
                calibrated_filepath = os.path.join(download_dir, calibrated_filename) if download_dir else calibrated_filename
                if not self._save_dataset(ds, calibrated_filepath, debug=debug):
                    print(f"Failed to save calibrated dataset: {calibrated_filepath}", file=sys.stderr)
        
        if not datasets:
            if not load_xarray and not download: # if no datasets were loaded and no download was requested, just confirm existence.
                print(f"No datasets processed for the query with current flags. File existence was confirmed where applicable.")
            else:
                print(f"No datasets processed for the query with current flags.")
            return None
        
        return xr.concat(datasets, dim='time') if len(datasets) > 1 else datasets[0]

