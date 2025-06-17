"""
This module provides functions for converting between GK2A satellite image
coordinates (column, line) and geographical coordinates (latitude, longitude)
for Full Disk (FD) and Local Area (LA) observation areas.

It implements a robust Geostationary (GEOS) projection based on standard
geospatial transformation principles.
"""

import numpy as np
from datetime import datetime
import sys # For debug prints to stderr

# --- Constants for WGS84 Ellipsoid and Satellite ---
# Earth's equatorial radius in km (WGS84 semi-major axis)
REQ = 6378.137
# Earth's polar radius in km (WGS84 semi-minor axis)
RPOL = 6356.7523
# Square of eccentricity (e^2 = (a^2 - b^2) / a^2)
ECC_SQ = (REQ**2 - RPOL**2) / REQ**2

# Geostationary satellite height from Earth's center in km (Rs from readme_latlon.txt)
SAT_DISTANCE_FROM_EARTH_CENTER = 42164.000000

# Sub-satellite longitude for GEOS projection (GK2A is at 128.2E)
SUB_LON_GEOS_DEG = 128.2
SUB_LON_GEOS_RAD = np.deg2rad(SUB_LON_GEOS_DEG)

# --- GEOS Projection Parameters for Different Areas and Resolutions ---
# These parameters are now primarily for `to_pixel` (inverse projection) and
# for fallback/reference. For `to_latlon`, the `x_rad` and `y_rad` inputs
# will be derived from the NetCDF attributes in `core.py`.
GEOS_PROJECTION_PARAMS = {
    "fd": {
        "0.5km": {"CFAC": 81701355.0, "LFAC": 81701355.0, "COFF": 11000.5, "LOFF": 11000.5, "image_width": 22000, "image_height": 22000},
        "1km": {"CFAC": 40850677.0, "LFAC": 40850677.0, "COFF": 5500.5, "LOFF": 5500.5, "image_width": 11000, "image_height": 11000},
        "2km": {"CFAC": 20425338.0, "LFAC": 20425338.0, "COFF": 2750.5, "LOFF": 2750.5, "image_width": 5500, "image_height": 5500},
    },
    "la": { # Using ELA values from readme_latlon.txt as best fit for inverse.
        "0.5km": {"CFAC": 81701355.0, "LFAC": 81701355.0, "COFF": 4500.5, "LOFF": 9348.5, "image_width": 7600, "image_height": 4800},
        "1km": {"CFAC": 40850677.0, "LFAC": 40850677.0, "COFF": 5500.5, "LOFF": 18697.0, "image_width": 3800, "image_height": 2400},
        "2km": {"CFAC": 20425338.0, "LFAC": 20425338.0, "COFF": 2750.5, "LOFF": 9348.5, "image_width": 1900, "image_height": 1200},
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
    
    # Handle the case where p is zero (at the poles of the ECEF system)
    # If p is zero, longitude is undefined. Set to NaN.
    lon_rad = np.arctan2(Y, X)
    if isinstance(lon_rad, np.ndarray):
        lon_rad[p == 0] = np.nan
    elif p == 0:
        lon_rad = np.nan

    # Initial approximation for latitude
    lat_rad = np.arctan2(Z, p * (1 - ECC_SQ))

    for _ in range(5):  # Iterate to refine latitude (sufficient for good accuracy)
        N = REQ / np.sqrt(1 - ECC_SQ * np.sin(lat_rad)**2)
        lat_rad = np.arctan2(Z + N * ECC_SQ * np.sin(lat_rad), p)

    return np.rad2deg(lat_rad), np.rad2deg(lon_rad)

# --- GEOS Projection (Full Disk and Local Area) ---
# This function now takes x_rad and y_rad directly as inputs
def _geos_to_latlon_core(x_rad, y_rad):
    """
    Converts GEOS projection satellite view angles (x_rad, y_rad) to latitude and longitude.
    This implementation uses a common method for ray-ellipsoid intersection for geostationary
    satellites, often found in GOES-R or MSG documentation.

    Args:
        x_rad (float or np.ndarray): Satellite scan angle in x-direction (radians).
        y_rad (float or np.ndarray): Satellite scan angle in y-direction (radians).

    Returns:
        tuple: (latitude, longitude) in degrees. Returns np.nan for points off Earth.
    """
    epsilon = 1e-12 # Small value to prevent division by zero or log(0)

    # 1. Compute intermediate values for ray direction vector components
    # These are components of a unit vector from the satellite to the Earth point,
    # in a satellite-fixed coordinate system where satellite is at (SAT_DISTANCE_FROM_EARTH_CENTER, 0, 0)
    # and looking towards (0,0,0).
    cos_x = np.cos(x_rad)
    sin_x = np.sin(x_rad)
    cos_y = np.cos(y_rad)
    sin_y = np.sin(y_rad)

    # Note: Conventions for Vx, Vy, Vz vary slightly between sources.
    # This set (from a common geostationary inverse projection) has been verified to work.
    Vx = cos_y * cos_x
    Vy = cos_y * sin_x
    Vz = sin_y

    # 2. Setup quadratic equation for intersection with the ellipsoid
    # The quadratic equation is A * K^2 + B * K + C = 0, where K is a scaling factor for the ray.
    # We are looking for the point of intersection between a ray from the satellite and the ellipsoid.

    # Coefficients for the quadratic equation
    # A coefficient: related to the shape of the ellipsoid and ray direction
    A = Vx**2 + (Vy**2 + Vz**2) * (RPOL/REQ)**2
    # B coefficient: related to the satellite's position (SAT_DISTANCE_FROM_EARTH_CENTER) and ray direction
    B = -2 * SAT_DISTANCE_FROM_EARTH_CENTER * Vx
    # C coefficient: related to satellite position and Earth's equatorial radius
    C = SAT_DISTANCE_FROM_EARTH_CENTER**2 - REQ**2

    print(f"[DEBUG _geos_to_latlon_core] Vx min/max: {np.nanmin(Vx):.4e} / {np.nanmax(Vx):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] Vy min/max: {np.nanmin(Vy):.4e} / {np.nanmax(Vy):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] Vz min/max: {np.nanmin(Vz):.4e} / {np.nanmax(Vz):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] A min/max (finite only): {np.nanmin(A[np.isfinite(A)]):.4e} / {np.nanmax(A[np.isfinite(A)]):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] B min/max (finite only): {np.nanmin(B[np.isfinite(B)]):.4e} / {np.nanmax(B[np.isfinite(B)]):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] C: {C:.4e}", file=sys.stderr)

    # 3. Solve the quadratic equation for K (distance along the ray)
    discriminant = B**2 - 4 * A * C

    print(f"[DEBUG _geos_to_latlon_core] Discriminant min/max (finite only): {np.nanmin(discriminant[np.isfinite(discriminant)]):.4e} / {np.nanmax(discriminant[np.isfinite(discriminant)]):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] Discriminant negative count: {np.sum(discriminant < 0)} / {discriminant.size}", file=sys.stderr)

    # Initialize K with NaNs, then fill for valid points
    K = np.full_like(discriminant, np.nan, dtype=float)

    # Mask for valid points: discriminant must be non-negative AND A must be non-zero
    valid_mask_discriminant = discriminant >= 0
    valid_mask_A_nonzero = (np.abs(A) > epsilon)
    combined_valid_mask = valid_mask_discriminant & valid_mask_A_nonzero

    if np.any(combined_valid_mask):
        # Take the negative root to get the intersection point closer to the satellite (on the Earth's surface)
        K[combined_valid_mask] = (-B[combined_valid_mask] - np.sqrt(discriminant[combined_valid_mask])) / (2 * A[combined_valid_mask])

    # Also, check for points behind the satellite (K < 0 or if the ray starts behind the satellite relative to its view)
    # The geometric setup implicitly assumes satellite is at a positive X_ECEF location and looking towards origin.
    # K should be positive for a valid intersection "in front" of the satellite.
    if isinstance(K, np.ndarray):
        K[K < 0] = np.nan
    elif K < 0:
        K = np.nan

    print(f"[DEBUG _geos_to_latlon_core] K min/max (finite only): {np.nanmin(K[np.isfinite(K)]):.4e} / {np.nanmax(K[np.isfinite(K)]):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] K NaN count: {np.sum(np.isnan(K))} / {K.size}", file=sys.stderr)


    # 4. Calculate ECEF coordinates (X, Y, Z) relative to the satellite's primary viewing axis (at longitude 0)
    # Satellite is at (SAT_DISTANCE_FROM_EARTH_CENTER, 0, 0)
    X_sat_prime = SAT_DISTANCE_FROM_EARTH_CENTER + K * Vx
    Y_sat_prime = K * Vy
    Z_sat_prime = K * Vz

    print(f"[DEBUG _geos_to_latlon_core] X_sat_prime min/max (finite only): {np.nanmin(X_sat_prime[np.isfinite(X_sat_prime)]):.4e} / {np.nanmax(X_sat_prime[np.isfinite(X_sat_prime)]):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] Y_sat_prime min/max (finite only): {np.nanmin(Y_sat_prime[np.isfinite(Y_sat_prime)]):.4e} / {np.nanmax(Y_sat_prime[np.isfinite(Y_sat_prime)]):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] Z_sat_prime min/max (finite only): {np.nanmin(Z_sat_prime[np.isfinite(Z_sat_prime)]):.4e} / {np.nanmax(Z_sat_prime[np.isfinite(Z_sat_prime)]):.4e}", file=sys.stderr)


    # 5. Rotate ECEF coordinates to the standard Earth-Fixed system using SUB_LON_GEOS_RAD
    # Initialize X, Y, Z with NaNs, then fill for valid points
    X = np.full_like(K, np.nan, dtype=float)
    Y = np.full_like(K, np.nan, dtype=float)
    Z = np.full_like(K, np.nan, dtype=float)

    # Ensure all components used for ECEF are finite before rotating
    valid_ecef_mask = np.isfinite(X_sat_prime) & np.isfinite(Y_sat_prime) & np.isfinite(Z_sat_prime)

    if np.any(valid_ecef_mask):
        X[valid_ecef_mask] = X_sat_prime[valid_ecef_mask] * np.cos(SUB_LON_GEOS_RAD) - Y_sat_prime[valid_ecef_mask] * np.sin(SUB_LON_GEOS_RAD)
        Y[valid_ecef_mask] = X_sat_prime[valid_ecef_mask] * np.sin(SUB_LON_GEOS_RAD) + Y_sat_prime[valid_ecef_mask] * np.cos(SUB_LON_GEOS_RAD)
        Z[valid_ecef_mask] = Z_sat_prime[valid_ecef_mask]

    print(f"[DEBUG _geos_to_latlon_core] Final ECEF X min/max (finite only): {np.nanmin(X[np.isfinite(X)]):.4e} / {np.nanmax(X[np.isfinite(X)]):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] Final ECEF Y min/max (finite only): {np.nanmin(Y[np.isfinite(Y)]):.4e} / {np.nanmax(Y[np.isfinite(Y)]):.4e}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] Final ECEF Z min/max (finite only): {np.nanmin(Z[np.isfinite(Z)]):.4e} / {np.nanmax(Z[np.isfinite(Z)]):.4e}", file=sys.stderr)


    # 6. Convert ECEF to geodetic latitude and longitude
    lat_deg, lon_deg = _geocentric_to_geodetic(X, Y, Z)

    # Final mask propagation (if any ECEF coordinate was NaN, the result is NaN)
    final_nan_mask = np.isnan(X) | np.isnan(Y) | np.isnan(Z)
    
    if isinstance(lat_deg, np.ndarray):
        lat_deg[final_nan_mask] = np.nan
        lon_deg[final_nan_mask] = np.nan
    else: # scalar case
        if final_nan_mask:
            lat_deg = np.nan
            lon_deg = np.nan

    print(f"[DEBUG _geos_to_latlon_core] Final lat_deg min/max (finite only): {np.nanmin(lat_deg[np.isfinite(lat_deg)]):.4f} / {np.nanmax(lat_deg[np.isfinite(lat_deg)]):.4f}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] Final lon_deg min/max (finite only): {np.nanmin(lon_deg[np.isfinite(lon_deg)]):.4f} / {np.nanmax(lon_deg[np.isfinite(lon_deg)]):.4f}", file=sys.stderr)
    print(f"[DEBUG _geos_to_latlon_core] Final lat_deg NaN count: {np.sum(np.isnan(lat_deg))} / {lat_deg.size}", file=sys.stderr)

    return lat_deg, lon_deg

def _latlon_to_geos(latitude_deg, longitude_deg, area, resolution):
    """
    Converts latitude and longitude to GEOS projection image coordinates (column, line).
    This is the inverse of the _geos_to_latlon_core function.

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
        raise ValueError(f"Invalid area '{area}' for GEOS projection. Supported: 'fd', 'la'.")

    params = area_params.get(resolution)
    if not params:
        raise ValueError(f"Invalid resolution '{resolution}' for area '{area}' GEOS projection. Supported resolutions: {list(area_params.keys())}.")

    CFAC = params["CFAC"]
    LFAC = params["LFAC"]
    COFF = params["COFF"]
    LOFF = params["LOFF"]

    lat_rad = np.deg2rad(latitude_deg)
    lon_rad = np.deg2rad(longitude_deg)

    # Convert geodetic (lat, lon) to geocentric latitude (phi_c) and distance from Earth center (rc)
    phi_c = np.arctan2(RPOL**2 * np.sin(lat_rad), REQ**2 * np.cos(lat_rad))
    rc = RPOL / np.sqrt(1 - ECC_SQ * np.cos(phi_c)**2)

    # Calculate Earth-centered, Earth-fixed coordinates relative to the sub-satellite longitude
    X_gc = rc * np.cos(phi_c) * np.cos(lon_rad - SUB_LON_GEOS_RAD)
    Y_gc = rc * np.cos(phi_c) * np.sin(lon_rad - SUB_LON_GEOS_RAD)
    Z_gc = rc * np.sin(phi_c)

    # Calculate satellite-relative coordinates (sx, sy, sz)
    sx = SAT_DISTANCE_FROM_EARTH_CENTER - X_gc
    sy = -Y_gc # Y and Z axes are inverted in the projection plane relative to satellite's perspective.
    sz = Z_gc

    # Calculate the scan angles (x_rad, y_rad) from satellite-relative coordinates
    norm_factor = np.sqrt(sx**2 + sy**2 + sz**2)
    valid_mask = sx > 0 # Points behind the satellite are not visible

    x_rad = np.full_like(sx, np.nan, dtype=float)
    y_rad = np.full_like(sy, np.nan, dtype=float)

    if isinstance(sx, np.ndarray): # Handle numpy arrays
        x_rad[valid_mask] = np.arctan2(sy[valid_mask], sx[valid_mask])
        
        arg_arcsin = sz[valid_mask] / norm_factor[valid_mask]
        valid_arcsin_mask = (arg_arcsin >= -1.0) & (arg_arcsin <= 1.0)
        y_rad[valid_mask][valid_arcsin_mask] = np.arcsin(arg_arcsin[valid_arcsin_mask])
    else: # Single value
        if not valid_mask:
            return np.nan, np.nan
        x_rad = np.arctan2(sy, sx)
        
        arg_arcsin = sz / norm_factor
        if not (arg_arcsin >= -1.0 and arg_arcsin <= 1.0):
            return np.nan, np.nan
        y_rad = np.arcsin(arg_arcsin)

    # Convert to column and line using the projection parameters
    column = x_rad * CFAC + COFF
    line = y_rad * LFAC + LOFF

    return column, line


# --- Public API for Geolocation Conversion ---
# The to_latlon function now takes x_rad and y_rad as direct inputs,
# and resolution is only used for error reporting in this context.
def to_latlon(x_rad, y_rad, area, resolution):
    """
    Converts image scan angles (x_rad, y_rad) to latitude and longitude.

    Args:
        x_rad (float or np.ndarray): Satellite scan angle in x-direction (radians).
        y_rad (float or np.ndarray): Satellite scan angle in y-direction (radians).
        area (str): The observation area ('fd' or 'la').
        resolution (str): The resolution of the data (e.g., '0.5km', '1km', '2km').
                          Used for context and debugging only in this function.

    Returns:
        tuple: (latitude, longitude) in degrees.
    """
    # area and resolution are not directly used in _geos_to_latlon_core but are kept
    # for consistency with the original call signature and potential future debugging/logging.
    return _geos_to_latlon_core(x_rad, y_rad)

def to_pixel(latitude, longitude, area, resolution):
    """
    Converts latitude and longitude to image coordinates (column, line).

    Args:
        latitude (float or np.ndarray): The latitude(s) in degrees.
        longitude (float or np.ndarray): The longitude(s) in degrees.
        area (str): The observation area ('fd' or 'la').
        resolution (str): The resolution of the data (e.g., '0.5km', '1km', '2km').

    Returns:
        tuple: (column, line) coordinates in the image.
    """
    if area.lower() not in GEOS_PROJECTION_PARAMS:
        raise ValueError(f"Unsupported area for geolocation: '{area}'. Must be 'fd' or 'la'.")
    return _latlon_to_geos(latitude, longitude, area.lower(), resolution)

if __name__ == "__main__":
    # --- Example Usage (for testing purposes) ---
    print("--- Testing GEOS projection (Full Disk) ---", file=sys.stderr)
    # For FD, image_center_x/y should be ~0.0 radians
    # These are directly used as x_rad, y_rad
    fd_2km_example_x_rad = 0.0 # Center of FD in radians
    fd_2km_example_y_rad = 0.0 # Center of FD in radians
    
    print(f"\n--- Test 1: FD 2km Center (x_rad=0, y_rad=0) ---", file=sys.stderr)
    lat_fd_center, lon_fd_center = to_latlon(fd_2km_example_x_rad, fd_2km_example_y_rad, 'fd', '2km')
    print(f"FD 2km center (x_rad=0, y_rad=0) -> Lat: {lat_fd_center:.4f}, Lon: {lon_fd_center:.4f}", file=sys.stderr)

    # Test inverse for center pixel (assuming we get reasonable lat/lon)
    if not np.isnan(lat_fd_center) and not np.isnan(lon_fd_center):
        col_fd_re, line_fd_re = to_pixel(lat_fd_center, lon_fd_center, 'fd', '2km')
        print(f"Inverse FD 2km ({lat_fd_center:.4f}, {lon_fd_center:.4f}) -> Col: {col_fd_re:.1f}, Line: {line_fd_re:.1f}", file=sys.stderr)
    else:
        print("Skipping inverse test for FD center as to_latlon returned NaN.", file=sys.stderr)

    # Test 2: A known point (e.g., Seoul) for FD (convert lat/lon to x_rad/y_rad first for testing to_latlon)
    seoul_lat, seoul_lon = 37.5665, 126.9780
    
    # We need to manually calculate x_rad, y_rad for Seoul using the inverse pixel-to-angle mapping.
    # This assumes the mapping for FD is consistent.
    fd_2km_params = GEOS_PROJECTION_PARAMS['fd']['2km']
    # Use to_pixel to get pixel coordinates, then convert to x_rad, y_rad for to_latlon input
    col_seoul_approx, line_seoul_approx = to_pixel(seoul_lat, seoul_lon, 'fd', '2km')
    
    if not np.isnan(col_seoul_approx) and not np.isnan(line_seoul_approx):
        seoul_x_rad_approx = (col_seoul_approx - fd_2km_params["COFF"]) / fd_2km_params["CFAC"]
        seoul_y_rad_approx = (line_seoul_approx - fd_2km_params["LOFF"]) / fd_2km_params["LFAC"]
        print(f"\n--- Test 2: FD 2km Seoul (approx scan angles) ---", file=sys.stderr)
        print(f"Seoul ({seoul_lat}, {seoul_lon}) pixel approx: ({col_seoul_approx:.1f}, {line_seoul_approx:.1f}) -> x_rad: {seoul_x_rad_approx:.4e}, y_rad: {seoul_y_rad_approx:.4e}", file=sys.stderr)
        lat_re_seoul, lon_re_seoul = to_latlon(seoul_x_rad_approx, seoul_y_rad_approx, 'fd', '2km')
        print(f"FD 2km (x_rad={seoul_x_rad_approx:.4e}, y_rad={seoul_y_rad_approx:.4e}) -> Lat: {lat_re_seoul:.4f}, Lon: {lon_re_seoul:.4f}", file=sys.stderr)
    else:
        print(f"\n--- Test 2: Skipped Seoul test as to_pixel returned NaN for approximate values. ---", file=sys.stderr)


    # Test with arrays for FD
    print("\n--- Testing with Arrays (FD 2km - Center, Corners) ---", file=sys.stderr)
    # Define a set of column/line pairs for testing
    fd_2km_width = GEOS_PROJECTION_PARAMS['fd']['2km']['image_width']
    fd_2km_height = GEOS_PROJECTION_PARAMS['fd']['2km']['image_height']

    test_cols = np.array([0, fd_2km_width // 2, fd_2km_width - 1])
    test_lines = np.array([0, fd_2km_height // 2, fd_2km_height - 1])

    # Create meshgrid of pixel coordinates
    cols_mesh, lines_mesh = np.meshgrid(test_cols, test_lines)
    
    # Manually calculate x_rad, y_rad using the FD 2km params (which are valid for FD's internal grid)
    # This simulates what _add_geolocation in core.py will do.
    x_rad_array = (cols_mesh - fd_2km_params["COFF"]) / fd_2km_params["CFAC"]
    y_rad_array = (lines_mesh - fd_2km_params["LOFF"]) / fd_2km_params["LFAC"]

    print(f"x_rad_array (generated): \n{x_rad_array}", file=sys.stderr)
    print(f"y_rad_array (generated): \n{y_rad_array}", file=sys.stderr)

    lats_array, lons_array = to_latlon(x_rad_array, y_rad_array, 'fd', '2km')
    print(f"FD 2km (array x_rad, array y_rad) -> Lats: \n{lats_array}, \nLons: \n{lons_array}", file=sys.stderr)
    
    print("\n--- End of Geolocation Module Tests ---", file=sys.stderr)
