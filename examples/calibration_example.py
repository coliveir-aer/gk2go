"""
This example script demonstrates the functionality of the gk2go library.
It fetches GK-2A data, performs radiometric calibration, and displays the
resulting data as a simple image on its native pixel grid.
"""
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from gk2go import Gk2aDataFetcher

def main():
    """Main function to run the example."""
    fetcher = Gk2aDataFetcher()

    print("\n--- Fetching and Calibrating L1B Data ---")
    
    ds = fetcher.get_data(
        sensor='ami',
        product='ir105', # An infrared channel
        area='fd',
        query_type='latest',
        calibrate=True,
        debug=True
    )

    if ds is None:
        print("Could not retrieve data. Exiting.")
        return

    # --- Plotting the Calibrated Data ---
    data_to_plot = None
    plot_title = "Calibrated GK-2A Data"
    cbar_label = ""
    
    if 'brightness_temperature' in ds:
        data_to_plot = ds['brightness_temperature'].squeeze()
        cbar_label = "Brightness Temperature (K)"
        plot_title = f"Calibrated Brightness Temperature\n{ds.time.dt.strftime('%Y-%m-%d %H:%M UTC').item()}"

    elif 'albedo' in ds:
        data_to_plot = ds['albedo'].squeeze()
        cbar_label = "Albedo (%)"
        plot_title = f"Calibrated Albedo\n{ds.time.dt.strftime('%Y-%m-%d %H:%M UTC').item()}"

    if data_to_plot is not None:
        print("\n--- Generating Plot ---")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(data_to_plot.compute(), cmap=cm.viridis)
        ax.set_title(plot_title)
        ax.set_xlabel("Column Number")
        ax.set_ylabel("Line Number")
        
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label(cbar_label)
        
        plt.show()
    else:
        print("No calibrated data available to plot.")

if __name__ == "__main__":
    main()
