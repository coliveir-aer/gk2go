"""
This module provides functions for converting between GK2A satellite image
coordinates (column, line) and geographical coordinates (latitude, longitude)
for Full Disk (FD) and Local Area (LA) observation areas.

It strictly implements the Geostationary (GEOS) projection for both FD and LA,
using specific parameters (CFAC, LFAC, COFF, LOFF, image dimensions)
derived from the provided 'readme_latlon.txt' and '20190920_gk2a l1b imagery information_en.pdf'.
"""

import numpy as np
from datetime import datetime

# --- Constants for WGS84 Ellipsoid and Satellite ---
# Earth's equatorial radius in km
REQ = 6378.137
# Earth's polar radius in km
RPOL = 6356.7523
# Square of eccentricity (e^2 = (a^2 - b^2) / a^2)
ECC_SQ = (REQ**2 - RPOL**2) / REQ**2

# Geostationary satellite height from Earth's center in km (Rs from readme_latlon.txt)
SAT_DISTANCE_FROM_EARTH_CENTER = 42164.000000

# Sub-satellite longitude for GEOS projection (GK2A is at 128.2E)
SUB_LON_GEOS_DEG = 128.2
SUB_LON_GEOS_RAD = np.deg2rad(SUB_LON_GEOS_DEG)

# --- GEOS Projection Parameters for Different Areas and Resolutions ---
# These parameters (CFAC, LFAC, COFF, LOFF, image_width, image_height)
# are crucial for mapping pixel coordinates to satellite view angles
# and are extracted from 'readme_latlon.txt' and '20190920_gk2a l1b imagery information_en.pdf'.

GEOS_PROJECTION_PARAMS = {
    # Full Disk (FD) parameters
    "fd": {
        "0.5km": {
            "CFAC": 81701355.0,
            "LFAC": 81701355.0,
            "COFF": 11000.5,
            "LOFF": 11000.5,
            "image_width": 22000,
            "image_height": 22000,
        },
        "1km": {
            "CFAC": 40850677.0,
            "LFAC": 40850677.0,
            "COFF": 5500.5,
            "LOFF": 5500.5,
            "image_width": 11000,
            "image_height": 11000,
        },
        "2km": {
            "CFAC": 20425338.0,
            "LFAC": 20425338.0,
            "COFF": 2750.5,
            "LOFF": 2750.5,
            "image_width": 5500,
            "image_height": 5500,
        },
    },
    # Local Area (LA) parameters from readme_latlon.txt
    # Image dimensions are inferred from ELA (Extended Local Area) as LA is a flexible domain.
    "la": {
        "0.5km": {
            "CFAC": 81701355.0,
            "LFAC": 81701355.0,
            "COFF": 1900.5,
            "LOFF": 1900.5,
            "image_width": 7600,  # From ELA 500m NCOLS/NROWS in readme_latlon.txt
            "image_height": 4800, # From ELA 500m NCOLS/NROWS in readme_latlon.txt
        },
        "1km": {
            "CFAC": 40850677.0,
            "LFAC": 40850677.0,
            "COFF": 950.5,
            "LOFF": 950.5,
            "image_width": 3800,  # From ELA 1km NCOLS/NROWS in readme_latlon.txt
            "image_height": 2400, # From ELA 1km NCOLS/NROWS in readme_latlon.txt
        },
        "2km": {
            "CFAC": 20425338.0,
            "LFAC": 20425338.0,
            "COFF": 475.5,
            "LOFF": 475.5,
            "image_width": 1900,  # From ELA 2km NCOLS/NROWS in readme_latlon.txt
            "image_height": 1200, # From ELA 2km NCOLS/NROWS in readme_latlon.txt
        },
    },
}


# --- Helper Functions (Geodetic <-> Geocentric) ---
def _geodetic_to_geocentric(latitude_deg, longitude_deg):
    """
    Converts geodetic (latitude, longitude) to geocentric Earth-Centered, Earth-Fixed (ECEF)
    (X, Y, Z) coordinates using WGS84 ellipsoid.

    Args:
        latitude_deg (float or np.ndarray): Geodetic latitude in degrees.
        longitude_deg (float or np.ndarray): Longitude in degrees.

    Returns:
        tuple: (X, Y, Z) in kilometers.
    """
    lat_rad = np.deg2rad(latitude_deg)
    lon_rad = np.deg2rad(longitude_deg)

    N = REQ / np.sqrt(1 - ECC_SQ * np.sin(lat_rad)**2)
    X = N * np.cos(lat_rad) * np.cos(lon_rad)
    Y = N * np.cos(lat_rad) * np.sin(lon_rad)
    Z = N * (1 - ECC_SQ) * np.sin(lat_rad)
    return X, Y, Z

def _geocentric_to_geodetic(X, Y, Z):
    """
    Converts geocentric ECEF (X, Y, Z) to geodetic (latitude, longitude) coordinates
    using WGS84 ellipsoid (iterative solution).

    Args:
        X (float or np.ndarray): X coordinate in kilometers.
        Y (float or np.ndarray): Y coordinate in kilometers.
        Z (float or np.ndarray): Z coordinate in kilometers.

    Returns:
        tuple: (latitude, longitude) in degrees.
    """
    p = np.sqrt(X**2 + Y**2)
    lon_rad = np.arctan2(Y, X)

    # Initial approximation for latitude
    lat_rad = np.arctan2(Z, p * (1 - ECC_SQ))

    for _ in range(5):  # Iterate to refine latitude (sufficient for good accuracy)
        N = REQ / np.sqrt(1 - ECC_SQ * np.sin(lat_rad)**2)
        lat_rad = np.arctan2(Z + N * ECC_SQ * np.sin(lat_rad), p)

    return np.rad2deg(lat_rad), np.rad2deg(lon_rad)

# --- GEOS Projection (Full Disk and Local Area) ---
def _geos_to_latlon(column, line, area, resolution):
    """
    Converts GEOS projection image coordinates (column, line) to latitude and longitude.
    This uses the inverse GOES-R ABI fixed grid projection formulas, which are generally
    applicable to similar geostationary imagers like GK2A.

    Args:
        column (float or np.ndarray): The column coordinate(s) in the image.
        line (float or np.ndarray): The line coordinate(s) in the image.
        area (str): The observation area ('fd' or 'la').
        resolution (str): The resolution of the data (e.g., '0.5km', '1km', '2km').

    Returns:
        tuple: (latitude, longitude) in degrees.
    """
    area_params = GEOS_PROJECTION_PARAMS.get(area)
    if not area_params:
        raise ValueError(f"Invalid area '{area}' for GEOS projection.")

    params = area_params.get(resolution)
    if not params:
        raise ValueError(f"Invalid resolution '{resolution}' for area '{area}' GEOS projection.")

    CFAC = params["CFAC"]
    LFAC = params["LFAC"]
    COFF = params["COFF"]
    LOFF = params["LOFF"]

    # Convert column/line to satellite view angles (x_rad, y_rad)
    # These are radians subtended from the satellite's perspective.
    x_rad = (column - COFF) / CFAC
    y_rad = (line - LOFF) / LFAC

    # Calculate intermediate values for intersection with ellipsoid
    # Rs is SAT_DISTANCE_FROM_EARTH_CENTER
    # Formulas adapted from GOES-R ABI PUG and similar satellite projection documents.
    a = (SAT_DISTANCE_FROM_EARTH_CENTER * np.cos(x_rad) * np.cos(y_rad))**2 \
        - ((REQ**2 - RPOL**2) / REQ**2) * (RPOL**2 * np.cos(x_rad)**2 * np.sin(y_rad)**2 \
                                           + REQ**2 * np.sin(x_rad)**2)

    b = -2 * SAT_DISTANCE_FROM_EARTH_CENTER * np.cos(x_rad) * np.cos(y_rad)
    c = SAT_DISTANCE_FROM_EARTH_CENTER**2 - REQ**2

    # Solve for the distance from the satellite to the Earth's surface point (rs)
    # This quadratic equation should yield two roots; we take the physically meaningful one.
    discriminant = b**2 - 4 * a * c
    # Handle cases where the view vector does not intersect the Earth (e.g., space views)
    if isinstance(discriminant, np.ndarray):
        rs = np.full_like(discriminant, np.nan, dtype=float)
        valid_mask = discriminant >= 0
        rs[valid_mask] = (-b[valid_mask] - np.sqrt(discriminant[valid_mask])) / (2 * a[valid_mask])
    else: # single value
        if discriminant < 0:
            return np.nan, np.nan # No intersection
        rs = (-b - np.sqrt(discriminant)) / (2 * a)

    # Calculate Earth-Centered, Earth-Fixed (ECEF) coordinates (X, Y, Z)
    # in the satellite's primary coordinate system (sub-satellite longitude = 0)
    sx = SAT_DISTANCE_FROM_EARTH_CENTER - rs * np.cos(x_rad) * np.cos(y_rad)
    sy = rs * np.sin(x_rad)
    sz = rs * np.cos(x_rad) * np.sin(y_rad)

    # Rotate ECEF coordinates from the satellite's primary system to the standard ECEF system
    # using the true sub-satellite longitude.
    X = sx
    Y = sy * np.cos(SUB_LON_GEOS_RAD) - sz * np.sin(SUB_LON_GEOS_RAD)
    Z = sy * np.sin(SUB_LON_GEOS_RAD) + sz * np.cos(SUB_LON_GEOS_RAD)

    # Convert ECEF to geodetic latitude and longitude
    lat_deg, lon_deg = _geocentric_to_geodetic(X, Y, Z)

    # Mask out invalid points (e.g., from discriminant < 0)
    if isinstance(discriminant, np.ndarray):
        lat_deg[~valid_mask] = np.nan
        lon_deg[~valid_mask] = np.nan

    return lat_deg, lon_deg

def _latlon_to_geos(latitude_deg, longitude_deg, area, resolution):
    """
    Converts latitude and longitude to GEOS projection image coordinates (column, line).
    This is the inverse of the _geos_to_latlon function.

    Args:
        latitude_deg (float or np.ndarray): Latitude in degrees.
        longitude_deg (float or np.ndarray): Longitude in degrees.
        area (str): The observation area ('fd' or 'la').
        resolution (str): The resolution of the data (e.g., '0.5km', '1km', '2km').

    Returns:
        tuple: (column, line) coordinates in the image.
    """
    area_params = GEOS_PROJECTION_PARAMS.get(area)
    if not area_params:
        raise ValueError(f"Invalid area '{area}' for GEOS projection.")

    params = area_params.get(resolution)
    if not params:
        raise ValueError(f"Invalid resolution '{resolution}' for area '{area}' GEOS projection.")

    CFAC = params["CFAC"]
    LFAC = params["LFAC"]
    COFF = params["COFF"]
    LOFF = params["LOFF"]

    lat_rad = np.deg2rad(latitude_deg)
    lon_rad = np.deg2rad(longitude_deg)

    # Convert geodetic (lat, lon) to geocentric latitude (phi_c) and distance from Earth center (rc)
    # This is a common step for satellite viewing geometry.
    phi_c = np.arctan2(RPOL**2 * np.sin(lat_rad), REQ**2 * np.cos(lat_rad))
    rc = RPOL / np.sqrt(1 - ECC_SQ * np.cos(phi_c)**2)

    # Calculate Earth-centered, Earth-fixed coordinates relative to the sub-satellite longitude
    # This is the "rotated" ECEF coordinate system for simpler calculations in the satellite frame.
    X_gc = rc * np.cos(phi_c) * np.cos(lon_rad - SUB_LON_GEOS_RAD)
    Y_gc = rc * np.cos(phi_c) * np.sin(lon_rad - SUB_LON_GEOS_RAD)
    Z_gc = rc * np.sin(phi_c)

    # Calculate satellite-relative coordinates (sx, sy, sz)
    # The satellite is at (Rs, 0, 0) in its own coordinate system
    sx = SAT_DISTANCE_FROM_EARTH_CENTER - X_gc
    sy = -Y_gc # Y and Z axes are inverted in the projection plane relative to satellite's perspective.
    sz = Z_gc

    # Calculate the scan angles (x_rad, y_rad) from satellite-relative coordinates
    # These are directly related to the pixel coordinates through CFAC/LFAC/COFF/LOFF
    norm_factor = np.sqrt(sx**2 + sy**2 + sz**2)
    # Check for points behind the satellite (not visible)
    # If sx is negative, the point is behind the satellite's primary viewing axis
    valid_mask = sx > 0
    if isinstance(sx, np.ndarray): # Handle numpy arrays
        x_rad = np.full_like(sx, np.nan, dtype=float)
        y_rad = np.full_like(sy, np.nan, dtype=float)
        x_rad[valid_mask] = np.arctan2(sy[valid_mask], sx[valid_mask])
        y_rad[valid_mask] = np.arcsin(sz[valid_mask] / norm_factor[valid_mask])
    else: # Single value
        if not valid_mask:
            return np.nan, np.nan
        x_rad = np.arctan2(sy, sx)
        y_rad = np.arcsin(sz / norm_factor)


    # Convert to column and line using the projection parameters
    column = x_rad * CFAC + COFF
    line = y_rad * LFAC + LOFF

    return column, line


# --- Public API for Geolocation Conversion ---
def to_latlon(column, line, area, resolution):
    """
    Converts image coordinates (column, line) to latitude and longitude.

    Args:
        column (float or np.ndarray): The column coordinate(s) in the image.
        line (float or np.ndarray): The line coordinate(s) in the image.
        area (str): The observation area (e.g., 'fd', 'la').
        resolution (str): The resolution of the data (e.g., '0.5km', '1km', '2km').

    Returns:
        tuple: (latitude, longitude) in degrees.
    """
    if area.lower() not in GEOS_PROJECTION_PARAMS:
        raise ValueError(f"Unsupported area for geolocation: '{area}'. Must be 'fd' or 'la'.")
    return _geos_to_latlon(column, line, area.lower(), resolution)

def to_pixel(latitude, longitude, area, resolution):
    """
    Converts latitude and longitude to image coordinates (column, line).

    Args:
        latitude (float or np.ndarray): The latitude(s) in degrees.
        longitude (float or np.ndarray): The longitude(s) in degrees.
        area (str): The observation area (e.g., 'fd', 'la').
        resolution (str): The resolution of the data (e.g., '0.5km', '1km', '2km').

    Returns:
        tuple: (column, line) coordinates in the image.
    """
    if area.lower() not in GEOS_PROJECTION_PARAMS:
        raise ValueError(f"Unsupported area for geolocation: '{area}'. Must be 'fd' or 'la'.")
    return _latlon_to_geos(latitude, longitude, area.lower(), resolution)

if __name__ == "__main__":
    # --- Example Usage (for testing purposes) ---
    print("--- Testing GEOS projection (Full Disk) ---")
    # Test 1: Center pixel for FD 0.5km should be (approx 0.0, 128.2)
    fd_05km_params = GEOS_PROJECTION_PARAMS['fd']['0.5km']
    center_column_fd = fd_05km_params["image_width"] / 2.0 - 0.5 # Center pixel for 0-indexed data
    center_line_fd = fd_05km_params["image_height"] / 2.0 - 0.5 # Adjust for 0-indexing
    lat_fd_center, lon_fd_center = to_latlon(center_column_fd, center_line_fd, 'fd', '0.5km')
    print(f"FD 0.5km center ({center_column_fd:.1f}, {center_line_fd:.1f}) -> Lat: {lat_fd_center:.4f}, Lon: {lon_fd_center:.4f}")

    # Test inverse for center pixel
    col_fd_re, line_fd_re = to_pixel(lat_fd_center, lon_fd_center, 'fd', '0.5km')
    print(f"Inverse FD 0.5km ({lat_fd_center:.4f}, {lon_fd_center:.4f}) -> Col: {col_fd_re:.1f}, Line: {line_fd_re:.1f}")

    # Test 2: A known point (e.g., Seoul) for FD
    seoul_lat, seoul_lon = 37.5665, 126.9780
    col_seoul, line_seoul = to_pixel(seoul_lat, seoul_lon, 'fd', '0.5km')
    print(f"\nSeoul ({seoul_lat}, {seoul_lon}) -> FD 0.5km Col: {col_seoul:.1f}, Line: {line_seoul:.1f}")
    lat_re, lon_re = to_latlon(col_seoul, line_seoul, 'fd', '0.5km')
    print(f"Inverse FD 0.5km ({col_seoul:.1f}, {line_seoul:.1f}) -> Lat: {lat_re:.4f}, Lon: {lon_re:.4f}")


    print("\n--- Testing GEOS projection (Local Area - LA) ---")
    # Test 3: LA 1km - Center point
    la_1km_params = GEOS_PROJECTION_PARAMS['la']['1km']
    center_column_la = la_1km_params["image_width"] / 2.0 - 0.5
    center_line_la = la_1km_params["image_height"] / 2.0 - 0.5
    lat_la_center, lon_la_center = to_latlon(center_column_la, center_line_la, 'la', '1km')
    print(f"LA 1km center ({center_column_la:.1f}, {center_line_la:.1f}) -> Lat: {lat_la_center:.4f}, Lon: {lon_la_center:.4f}")
    col_la_re, line_la_re = to_pixel(lat_la_center, lon_la_center, 'la', '1km')
    print(f"Inverse LA 1km -> Col: {col_la_re:.1f}, Line: {line_la_re:.1f}")

    # Test 4: LA 0.5km - A corner point (0,0)
    lat_la_ul, lon_la_ul = to_latlon(0, 0, 'la', '0.5km')
    print(f"\nLA 0.5km (0, 0) -> Lat: {lat_la_ul:.4f}, Lon: {lon_la_ul:.4f}")
    col_la_ul_re, line_la_ul_re = to_pixel(lat_la_ul, lon_la_ul, 'la', '0.5km')
    print(f"Inverse LA 0.5km -> Col: {col_la_ul_re:.1f}, Line: {line_la_ul_re:.1f}")

    # Test with arrays
    print("\n--- Testing with Arrays (FD 0.5km) ---")
    cols_array = np.array([0, 1000, fd_05km_params["image_width"] - 1])
    lines_array = np.array([0, 1000, fd_05km_params["image_height"] - 1])
    lats_array, lons_array = to_latlon(cols_array, lines_array, 'fd', '0.5km')
    print(f"FD 0.5km (array cols, array lines) -> Lats: {lats_array}, Lons: {lons_array}")
    cols_re_array, lines_re_array = to_pixel(lats_array, lons_array, 'fd', '0.5km')
    print(f"Inverse (array lats, array lons) -> Cols: {cols_re_array}, Lines: {lines_re_array}")

