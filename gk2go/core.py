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

class GK2ADefs:
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
    S3_BUCKET = "noaa-gk2a-pds"
    AWS_REGION = "us-east-1"

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

class S3Utils:
    def __init__(self):
        try:
            self.fs = s3fs.S3FileSystem(anon=True)
            self.s3_client = boto3.client(
                's3', region_name=GK2ADefs.AWS_REGION, config=Config(signature_version=UNSIGNED)
            )
        except Exception as e:
            print(f"FATAL: Error initializing S3 clients: {e}", file=sys.stderr)
            raise

    def list_files_in_prefix(self, prefix):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=GK2ADefs.S3_BUCKET, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        yield obj
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                print(f"ERROR: S3 client error on prefix '{prefix}': {e}", file=sys.stderr)
            return

class Gk2aDataFetcher:
    def __init__(self):
        self.s3_utils = S3Utils()

    def _calibrate(self, ds, product_name):
        """
        Calibrates the raw data in the dataset to scientific units.
        """
        print(f"--- Entering Calibration for {product_name} ---")
        try:
            # --- Definitive helper to get a scalar value ---
            def get_scalar(attr_name):
                val = ds.attrs[attr_name]
                print(f"  [get_scalar] Initial value for '{attr_name}': {repr(val)} (type: {type(val)})")
                while isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) == 0:
                        raise ValueError(f"Calibration coefficient {attr_name} is an empty sequence.")
                    val = val[0]
                    print(f"  [get_scalar] Unwrapped to: {repr(val)} (type: {type(val)})")
                return float(val)

            gain = get_scalar('DN_to_Radiance_Gain')
            offset = get_scalar('DN_to_Radiance_Offset')
            
            # Operate on the raw dask/numpy array to avoid potential xarray overhead issues
            dn_array = ds['image_pixel_values'].data
            radiance_array = dn_array * gain + offset
            
            radiance = xr.DataArray(
                radiance_array,
                coords=ds['image_pixel_values'].coords,
                dims=ds['image_pixel_values'].dims,
                attrs={'units': 'W m-2 sr-1 um-1'}
            )
            print("[DEBUG] DN to Radiance conversion successful.")
            
            # --- Channel-Specific Calibration ---
            channel_type = product_name[:2]

            if channel_type in ['vi', 'nr']:
                c = get_scalar('Radiance_to_Albedo_c')
                albedo = radiance * c * 100
                albedo.attrs['long_name'] = 'Albedo'
                albedo.attrs['units'] = '%'
                ds['albedo'] = albedo
                
            elif channel_type in ['sw', 'ir', 'wv']:
                h = get_scalar('Plank_constant_h')
                k = get_scalar('Boltzmann_constant_k')
                c_light = get_scalar('light_speed')
                lambda_c = get_scalar('channel_center_wavelength') * 1e-6
                
                c1_planck = 2 * h * c_light**2
                c2_planck = (h * c_light) / k
                wavenumber_m = (1 / lambda_c)
                teff = (c2_planck * wavenumber_m) / np.log(1 + (c1_planck * wavenumber_m**3) / radiance)

                c0 = get_scalar('Teff_to_Tbb_c0')
                c1_t = get_scalar('Teff_to_Tbb_c1')
                c2_t = get_scalar('Teff_to_Tbb_c2')
                
                tbb = c2_t * teff**2 + c1_t * teff + c0
                tbb.attrs['long_name'] = 'Brightness Temperature'
                tbb.attrs['units'] = 'K'
                ds['brightness_temperature'] = tbb

            print(f"--- Calibration successful for {product_name} ---")
            return ds

        except Exception as e:
            print(f"\n---!!! CALIBRATION FAILED for {product_name} !!!---", file=sys.stderr)
            print(f"ERROR MESSAGE: {e}", file=sys.stderr)
            print("\n--- Full Stack Trace ---", file=sys.stderr)
            traceback.print_exc()
            print("------------------------", file=sys.stderr)
            print("Returning uncalibrated data.", file=sys.stderr)
            return ds


    @staticmethod
    def _generate_s3_prefixes(sensor, product, start_time, end_time, area=None):
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

    def _find_files(self, sensor, product, start_time, end_time, area=None):
        all_files = []
        prefixes = list(self._generate_s3_prefixes(sensor, product, start_time, end_time, area))
        for prefix in prefixes:
            s3_objects = self.s3_utils.list_files_in_prefix(prefix)
            for obj in s3_objects:
                filename = os.path.basename(obj['Key'])
                parsed_data = GK2ADefs.parse_filename(filename)
                if not parsed_data: continue
                if not (start_time <= parsed_data.get('datetime', start_time - timedelta(seconds=1)) <= end_time): continue
                parsed_data['s3_key'] = obj['Key']
                all_files.append(parsed_data)
        all_files.sort(key=lambda x: x.get('datetime', datetime.min))
        return all_files
    
    def _load_as_xarray(self, s3_path):
        try:
            remote_file = self.s3_utils.fs.open(s3_path, 'rb')
            ds = xr.open_dataset(remote_file, chunks='auto')
            return ds
        except Exception as e:
            print(f"ERROR: Could not open S3 object {s3_path} as xarray.Dataset: {e}", file=sys.stderr)
            return None

    def get_data(self, sensor, product, area, query_type='latest', target_time=None, start_time=None, end_time=None, calibrate=False):
        print(f"\n--- Entering get_data for product: {product}, query: {query_type} ---")
        
        if query_type == 'latest':
            search_end = datetime.utcnow()
            search_start = search_end - timedelta(hours=2)
            found_files = self._find_files(sensor, product, search_start, search_end, area)
            if not found_files: 
                print("[INFO] No files found for 'latest' query.")
                return None
            
            latest_file = found_files[-1]
            print(f"[DEBUG] Latest file found: {latest_file['s3_key']}")
            
            print("[DEBUG] Calling _load_as_xarray...")
            ds = self._load_as_xarray(f"s3://{GK2ADefs.S3_BUCKET}/{latest_file['s3_key']}")
            
            if not ds: 
                print("[ERROR] _load_as_xarray returned None.")
                return None
            
            print(f"[DEBUG] Dataset loaded. Calibrate flag is: {calibrate}")
            if calibrate:
                print("[DEBUG] Entering _calibrate function call...")
                ds = self._calibrate(ds, product)
            
            print("[DEBUG] Expanding dims and returning dataset.")
            return ds.expand_dims(time=[latest_file['datetime']])

        elif query_type == 'nearest':
            if not target_time: raise ValueError("`target_time` is required for 'nearest' query.")
            search_start = target_time - timedelta(hours=2)
            search_end = target_time + timedelta(hours=2)
            found_files = self._find_files(sensor, product, search_start, search_end, area)
            if not found_files: return None

            nearest_file = min(found_files, key=lambda x: abs(x['datetime'] - target_time))
            ds = self._load_as_xarray(f"s3://{GK2ADefs.S3_BUCKET}/{nearest_file['s3_key']}")
            if not ds: return None
            
            if calibrate:
                ds = self._calibrate(ds, product)

            return ds.expand_dims(time=[nearest_file['datetime']])

        elif query_type == 'range':
            if not start_time or not end_time: raise ValueError("`start_time` and `end_time` are required for 'range' query.")
            found_files = self._find_files(sensor, product, start_time, end_time, area)
            if not found_files: return None

            datasets = []
            for file_info in found_files:
                ds = self._load_as_xarray(f"s3://{GK2ADefs.S3_BUCKET}/{file_info['s3_key']}")
                if ds:
                    main_var = next(iter(ds.data_vars))
                    ds_clean = ds[[main_var]]
                    if calibrate:
                        ds_clean = self._calibrate(ds, product)
                    datasets.append(ds_clean.expand_dims(time=[file_info['datetime']]))
            
            if not datasets: return None
            return xr.concat(datasets, dim='time')

        else:
            raise ValueError(f"Unknown query_type: '{query_type}'. Must be 'latest', 'nearest', or 'range'.")
