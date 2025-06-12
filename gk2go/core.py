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
                val = ds.attrs[attr_name]
                print(f"  [get_scalar] Initial value for '{attr_name}': {repr(val)} (type: {type(val)})")
                while isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) == 0:
                        raise ValueError(f"Calibration coefficient {attr_name} is an empty sequence.")
                    val = val[0]
                    print(f"  [get_scalar] Unwrapped to: {repr(val)} (type: {type(val)})")
                # Cast to float to handle cases where metadata gives a string (e.g., '10.5')
                return float(val)

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
            channel_type = product_name[:2]

            if channel_type in ['vi', 'nr']:
                c = get_scalar('Radiance_to_Albedo_c')
                albedo = radiance * c * 100
                albedo.attrs['long_name'] = 'Albedo'
                albedo.attrs['units'] = '%'
                ds['albedo'] = albedo

            elif channel_type in ['sw', 'ir', 'wv']:
                # The Planck function is only defined for positive radiance.
                # Where the linear calibration produces non-physical (negative or zero) radiance,
                # we must treat it as invalid data (NaN) to prevent a math error.
                positive_radiance = radiance.where(radiance > 0)

                # Get constants from dataset attributes
                cval = get_scalar('light_speed')
                kval = get_scalar('Boltzmann_constant_k')
                hval = get_scalar('Plank_constant_h')

                # Calculate wavenumber (wn) in m^-1
                # channel_center_wavelength is in micrometers (um)
                # 10000 / wavelength_in_um gives wavenumber in cm^-1
                # * 100 converts cm^-1 to m^-1.
                # So wn is in m^-1
                wn = (10000.0 / get_scalar('channel_center_wavelength')) * 100.0
                print(f"[DEBUG] Calculated Wavenumber (wn): {wn} (m^-1)")

                # Convert radiance to the specific units expected by Planck calculation (W m-2 sr-1 m-1)
                # Therefore, we should apply `* 1e-5` to our `positive_radiance` (which is already in W m-2 sr-1 um-1).
                radiance_for_planck = positive_radiance * 1e-5
                # Fix: Add .compute() to force evaluation before .item()
                print(f"[DEBUG] Radiance for Planck (radiance_for_planck): {radiance_for_planck.min().compute().item():.3e} to {radiance_for_planck.max().compute().item():.3e}")

                # Inverse Planck function to get Effective Temperature (Teff)
                e1 = (2.0 * hval * cval * cval) * np.power(wn, 3.0)
                e2 = radiance_for_planck

                # Guard against division by zero or non-positive e2 values in the log.
                # The `positive_radiance` already handles `radiance > 0`.
                # If `radiance_for_planck` (e2) becomes zero or negative due to further floating point issues,
                # np.log will produce errors. So apply the `.where` again for `e2`.
                e2 = e2.where(e2 > 0)

                term_in_log = (e1 / e2) + 1.0
                # Ensure term_in_log is positive to avoid log domain error
                term_in_log = term_in_log.where(term_in_log > 0)
                teff = ((hval * cval / kval) * wn) / np.log(term_in_log)
                # Fix: Add .compute() to force evaluation before .item()
                print(f"[DEBUG] Effective Temperature (teff): {teff.min().compute().item():.2f} K to {teff.max().compute().item():.2f} K")


                # This is a standard polynomial correction to get Brightness Temperature (Tbb)
                c0 = get_scalar('Teff_to_Tbb_c0')
                c1_t = get_scalar('Teff_to_Tbb_c1')
                c2_t = get_scalar('Teff_to_Tbb_c2')

                tbb = c2_t * teff**2 + c1_t * teff + c0
                tbb.attrs['long_name'] = 'Brightness Temperature'
                tbb.attrs['units'] = 'K'
                ds['brightness_temperature'] = tbb
                # Fix: Add .compute() to force evaluation before .item()
                print(f"[DEBUG] Brightness Temperature (tbb): {tbb.min().compute().item():.2f} K to {tbb.max().compute().item():.2f} K")

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
                
                if not parsed_data: 
                    continue
                
                file_time = parsed_data.get('datetime')
                if not file_time:
                    continue
                    
                if not (start_time <= file_time <= end_time): 
                    continue

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
        if query_type == 'latest':
            search_end = datetime.utcnow()
            search_start = search_end - timedelta(hours=2)
            found_files = self._find_files(sensor, product, search_start, search_end, area)
            if not found_files: return None
            
            latest_file = found_files[-1]
            ds = self._load_as_xarray(f"s3://{GK2ADefs.S3_BUCKET}/{latest_file['s3_key']}")
            if not ds: return None
            
            if calibrate:
                ds = self._calibrate(ds, product)
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
