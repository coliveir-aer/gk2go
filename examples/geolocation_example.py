import sys
import os
from datetime import datetime, timedelta
import traceback

# Import the Gk2aDataFetcher from our local gk2go.core module
# Ensure that gk2go.core is accessible in your Python environment's path
from gk2go.core import Gk2aDataFetcher

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.cm as cm # For colormaps


# Set this to True to enable detailed debug prints from the gk2go.core module
ENABLE_CORE_DEBUG = True
# --- End Configuration ---

def print_dataset_info(ds, title):
    """Prints key information about an xarray Dataset, including all attributes."""
    print(f"\n--- Dataset Info for: {title} ---", file=sys.stderr) # Directing to stderr for cleaner output
    if ds is None:
        print("Dataset is None.", file=sys.stderr)
        print("-----------------------------------", file=sys.stderr)
        return

    print(f"Dataset Dimensions: {ds.dims}", file=sys.stderr)
    print(f"Dataset Data Variables: {list(ds.data_vars.keys())}", file=sys.stderr)
    print(f"Dataset Coordinates: {list(ds.coords.keys())}", file=sys.stderr)

    print("\n--- All Dataset Attributes (.attrs): ---", file=sys.stderr)
    for attr, value in ds.attrs.items():
        # Handle cases where attribute value might be a numpy array or DataArray
        if isinstance(value, (np.ndarray, xr.DataArray)):
            if value.size == 1:
                print(f"  {attr}: {value.item()} (Type: {type(value.item())})", file=sys.stderr)
            else:
                # For arrays with multiple elements, print shape, dtype, min, max
                print(f"  {attr}: Array Shape {value.shape}, Dtype {value.dtype} (Min: {np.nanmin(value)}, Max: {np.nanmax(value)})", file=sys.stderr)
        else:
            print(f"  {attr}: {value} (Type: {type(value)})", file=sys.stderr)
    print("-----------------------------------", file=sys.stderr)


def main():
    """
    Main function to fetch, calibrate, and generate georeferenced plots
    for both Full Disk (FD) and Local Area (LA) GK2A scenes.
    Prints all dataset metadata for verification.
    """

    fetcher = Gk2aDataFetcher()
    sensor = 'ami'
    product = 'ir087' # Infrared channel for brightness temperature
    target_time = datetime.utcnow() - timedelta(hours=3) # Target a recent time

    # --- Fetch Full Disk (FD) Data ---
    print(f"\n===== Processing Full Disk (FD) Data =====", file=sys.stderr)
    ds_fd = None

    try:
        print(f"Fetching FD {product.upper()} data...", file=sys.stderr)
        ds_fd = fetcher.get_data(
            sensor=sensor,
            product=product,
            area='fd',
            query_type='latest',
            target_time=target_time,
            calibrate=True,
            geolocation_enabled=True, # IMPORTANT: Set to True to get lat/lon coordinates
            debug=ENABLE_CORE_DEBUG   # Control module's debug prints
        )

    except Exception as e:
        print(f"An error occurred during FD data fetch: {e}", file=sys.stderr)
        traceback.print_exc()
        ds_fd = None


    # --- Analysis and Plotting ---
    if ds_fd:
        print_dataset_info(ds_fd, "Full Disk (FD) Data")

        # Check for calibrated data
        data_to_plot = None
        plot_title = ""
        cmap_choice = None
        cbar_label = ""

        if 'brightness_temperature' in ds_fd.data_vars:
            data_to_plot = ds_fd['brightness_temperature'].squeeze()
            plot_title = f"GK2A AMI {product.upper()} Brightness Temperature"
            cmap_choice = cm.viridis # A perceptually uniform colormap, good for temperature
            cbar_label = "Brightness Temperature (K)"
            vmin_val = 180 # Example min temperature for IR
            vmax_val = 320 # Example max temperature for IR
        elif 'albedo' in ds_fd.data_vars:
            data_to_plot = ds_fd['albedo'].squeeze()
            plot_title = f"GK2A AMI {product.upper()} Albedo"
            cmap_choice = cm.gray_r # Reversed grayscale for albedo (higher albedo = brighter)
            cbar_label = "Albedo (%)"
            vmin_val = 0 # Albedo range
            vmax_val = 100 # Albedo range
        else:
            print("No calibrated data (brightness_temperature or albedo) found for plotting.", file=sys.stderr)
            data_to_plot = None # Ensure data_to_plot is None if no calibrated variable exists


        if data_to_plot is not None and 'longitude' in ds_fd.coords and 'latitude' in ds_fd.coords:
            print("\n--- Generating Map Plot ---", file=sys.stderr)
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

            # Get the actual latitude and longitude NumPy arrays
            lat_coords = ds_fd['latitude'].values
            lon_coords = ds_fd['longitude'].values

            # Calculate the bounding box for the valid (non-NaN) data
            # Use np.nanmin/max to ignore NaN values
            min_lon = np.nanmin(lon_coords)
            max_lon = np.nanmax(lon_coords)
            min_lat = np.nanmin(lat_coords)
            max_lat = np.nanmax(lat_coords)

            # Define the extent for imshow in PlateCarree coordinates
            # Note: imshow's extent expects [left, right, bottom, top]
            image_extent = [min_lon, max_lon, min_lat, max_lat]

            # Ensure data is loaded from Dask before plotting to avoid I/O issues during rendering
            data_values = data_to_plot.compute().values # Get the actual NumPy array values

            # Plot the data using imshow.
            # imshow takes the image data directly (data_values).
            # The 'transform' argument describes the coordinate system of the 'extent'.
            # The 'origin' determines where the [0,0] pixel is (usually 'upper' for satellite images).
            img = ax.imshow(
                data_values,
                transform=ccrs.PlateCarree(), # The extent is defined in PlateCarree (lat/lon)
                extent=image_extent,           # Bounding box of the data
                cmap=cmap_choice,
                vmin=vmin_val,
                vmax=vmax_val,
                origin='upper', # Adjust if image origin is bottom-left, typically 'upper' for satellite imagery
                interpolation='none' # Avoid interpolation artifacts
            )

            # Add a colorbar
            cbar = fig.colorbar(img, ax=ax, orientation='vertical', pad=0.05, label=cbar_label)


            ax.set_title(plot_title)
            ax.coastlines(resolution='50m', color='black', linewidth=0.8) # Add coastlines
            ax.gridlines(draw_labels=True, linestyle='--', alpha=0.6) # Add gridlines with labels
            ax.set_global() # Set the map extent to show the whole world

            print("Map plot generated successfully using imshow.", file=sys.stderr)
        else:
            print("Geolocation coordinates (latitude, longitude) or data to plot are missing, skipping map plot.", file=sys.stderr)
    else:
        print("\nFull Disk data not available for analysis or plotting.", file=sys.stderr)

    print("\n--- Example Script Finished ---", file=sys.stderr)
    plt.show() # Display all generated plots

if __name__ == "__main__":
    main()

