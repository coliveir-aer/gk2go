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

    def _load_as_xarray(self, s3_path, debug=False):
        try:
            if debug: print(f"Loading data from: {s3_path}")
            remote_file = self.s3_utils.s3_fs.open(s3_path, 'rb')
            return xr.open_dataset(remote_file, engine='h5netcdf', chunks='auto')
        except Exception as e:
            print(f"Error loading xarray dataset: {e}", file=sys.stderr)
            if debug: traceback.print_exc()
            return None

    def _calibrate(self, ds, product_name, debug=False):
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

    def get_data(self, sensor, product, area, query_type, target_time=None,
                 start_time=None, end_time=None, calibrate=True, debug=False):
        
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
            ds = self._load_as_xarray(f"s3://{self.s3_bucket}/{file_info['s3_key']}", debug=debug)
            if ds is None: continue

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
        
        if not datasets: return None
        return xr.concat(datasets, dim='time') if len(datasets) > 1 else datasets[0]

