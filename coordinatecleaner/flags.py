"""
Coordinate cleaning flags for occurrence records.

Each function returns a boolean array where True = valid (clean)
and False = potentially problematic (flagged).

Reference:
    Zizka et al. (2019). CoordinateCleaner: Standardized cleaning
    of occurrence records from biological collection databases.
    Methods in Ecology and Evolution 10:744-751.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional


def cc_val(
    lon: NDArray, 
    lat: NDArray
) -> NDArray[np.bool_]:
    """
    Flag invalid coordinates.
    
    Checks for:
    - Latitude outside [-90, 90]
    - Longitude outside [-180, 180]
    - NaN values
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
        
    Returns
    -------
    valid : ndarray of bool
        True if coordinates are valid
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    valid = (
        ~np.isnan(lon) & 
        ~np.isnan(lat) &
        (lon >= -180) & (lon <= 180) &
        (lat >= -90) & (lat <= 90)
    )
    
    return valid


def cc_zero(
    lon: NDArray, 
    lat: NDArray,
    buffer: float = 0.5
) -> NDArray[np.bool_]:
    """
    Flag coordinates at (0, 0).
    
    The point (0, 0) is in the Gulf of Guinea and is a common
    default/error value.
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
    buffer : float
        Distance buffer around (0, 0) in degrees (default 0.5)
        
    Returns
    -------
    valid : ndarray of bool
        True if not at (0, 0)
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    dist = np.sqrt(lon**2 + lat**2)
    valid = dist > buffer
    
    return valid


def cc_equ(
    lon: NDArray, 
    lat: NDArray,
    tolerance: float = 0.001
) -> NDArray[np.bool_]:
    """
    Flag coordinates where lat == lon.
    
    Equal latitude and longitude often indicates a copy/paste error.
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
    tolerance : float
        Maximum difference to be considered equal (default 0.001)
        
    Returns
    -------
    valid : ndarray of bool
        True if lat != lon
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    valid = np.abs(lon - lat) > tolerance
    
    return valid


def cc_dupl(
    lon: NDArray, 
    lat: NDArray,
    tolerance: float = 0.0001
) -> NDArray[np.bool_]:
    """
    Flag duplicate coordinates.
    
    Identifies records with identical coordinates.
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
    tolerance : float
        Distance within which points are considered duplicates
        
    Returns
    -------
    valid : ndarray of bool
        True for first occurrence, False for duplicates
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    n = len(lon)
    valid = np.ones(n, dtype=bool)
    
    # Mark duplicates (keep first)
    seen = set()
    for i in range(n):
        # Round to tolerance
        key = (round(lon[i] / tolerance), round(lat[i] / tolerance))
        if key in seen:
            valid[i] = False
        else:
            seen.add(key)
    
    return valid


def cc_round(
    lon: NDArray,
    lat: NDArray,
    decimals: int = 2
) -> NDArray[np.bool_]:
    """
    Flag heavily rounded coordinates.
    
    Coordinates rounded to few decimal places indicate low precision
    (e.g., "15.0, 47.0" likely has ~100km uncertainty).
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
    decimals : int
        Maximum decimals before flagging (default 2 = flag if <=1 decimal)
        
    Returns
    -------
    valid : ndarray of bool
        True if coordinates have sufficient precision
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    # Check if coordinates are suspiciously round
    def is_rounded(x: float, dec: int) -> bool:
        if np.isnan(x):
            return False
        multiplier = 10 ** dec
        return x * multiplier == round(x * multiplier)
    
    valid = np.array([
        not (is_rounded(lo, decimals - 1) and is_rounded(la, decimals - 1))
        for lo, la in zip(lon, lat)
    ])
    
    return valid


# Known problematic locations
GBIF_HQ = (-0.079, 51.52)  # Copenhagen
NATURAL_HISTORY_MUSEUMS = [
    (-0.1766, 51.4966),    # Natural History Museum London
    (-73.9737, 40.7810),   # American Museum of Natural History NYC
    (2.2868, 48.8422),     # Muséum Paris
    (13.3936, 52.5312),    # Museum für Naturkunde Berlin
    (144.9716, -37.8030),  # Museum Victoria Melbourne
]


def cc_gbif(
    lon: NDArray,
    lat: NDArray,
    buffer: float = 0.01
) -> NDArray[np.bool_]:
    """
    Flag coordinates at GBIF headquarters.
    
    Records geocoded to GBIF HQ are likely errors.
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
    buffer : float
        Distance buffer in degrees (default 0.01 ≈ 1km)
        
    Returns
    -------
    valid : ndarray of bool
        True if not at GBIF HQ
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    gbif_lon, gbif_lat = GBIF_HQ
    dist = np.sqrt((lon - gbif_lon)**2 + (lat - gbif_lat)**2)
    
    return dist > buffer


def cc_inst(
    lon: NDArray,
    lat: NDArray,
    buffer: float = 0.001,
    institutions: list = None
) -> NDArray[np.bool_]:
    """
    Flag coordinates at biodiversity institutions.
    
    Records geocoded to museums/herbariums are often collection
    locations, not actual species occurrences.
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values  
    buffer : float
        Distance buffer in degrees (default 0.001 ≈ 100m)
    institutions : list
        List of (lon, lat) tuples. Uses default major museums if None.
        
    Returns
    -------
    valid : ndarray of bool
        True if not at an institution
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    if institutions is None:
        institutions = NATURAL_HISTORY_MUSEUMS
    
    valid = np.ones(len(lon), dtype=bool)
    
    for inst_lon, inst_lat in institutions:
        dist = np.sqrt((lon - inst_lon)**2 + (lat - inst_lat)**2)
        valid &= (dist > buffer)
    
    return valid


# Major world capitals (lon, lat)
CAPITALS = [
    (-77.0369, 38.9072),   # Washington DC
    (-0.1276, 51.5074),    # London
    (2.3522, 48.8566),     # Paris
    (13.4050, 52.5200),    # Berlin
    (139.6917, 35.6895),   # Tokyo
    (116.4074, 39.9042),   # Beijing
    (-43.1729, -22.9068),  # Rio
    (37.6173, 55.7558),    # Moscow
    (151.2093, -33.8688),  # Sydney
    (28.9784, 41.0082),    # Istanbul
]


def cc_cap(
    lon: NDArray,
    lat: NDArray,
    buffer: float = 0.01
) -> NDArray[np.bool_]:
    """
    Flag coordinates at country capitals.
    
    Capitals are often used as default geocoding locations.
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
    buffer : float
        Distance buffer in degrees (default 0.01 ≈ 1km)
        
    Returns
    -------
    valid : ndarray of bool
        True if not at a capital
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    valid = np.ones(len(lon), dtype=bool)
    
    for cap_lon, cap_lat in CAPITALS:
        dist = np.sqrt((lon - cap_lon)**2 + (lat - cap_lat)**2)
        valid &= (dist > buffer)
    
    return valid


def cc_outl(
    lon: NDArray,
    lat: NDArray,
    method: str = 'quantile',
    threshold: float = 5.0,
    min_records: int = 7
) -> NDArray[np.bool_]:
    """
    Flag geographic outliers.
    
    Identifies records that are unusually far from other records
    of the same species, which may indicate errors.
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
    method : str
        'quantile' - flag if distance > threshold * IQR above median
        'mad' - flag if distance > threshold * MAD from median
    threshold : float
        Multiplier for outlier detection (default 5.0)
    min_records : int
        Minimum records needed; returns all True if fewer
        
    Returns
    -------
    valid : ndarray of bool
        True if not an outlier
        
    Notes
    -----
    Calculates distance from each point to the centroid of all points,
    then flags points with unusually large distances.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    n = len(lon)
    
    if n < min_records:
        return np.ones(n, dtype=bool)
    
    # Calculate centroid
    centroid_lon = np.nanmean(lon)
    centroid_lat = np.nanmean(lat)
    
    # Distance to centroid (simple Euclidean in degrees)
    # For more accuracy, should use haversine, but this is fast
    distances = np.sqrt((lon - centroid_lon)**2 + (lat - centroid_lat)**2)
    
    if method == 'quantile':
        q1, q3 = np.nanpercentile(distances, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + threshold * iqr
        valid = distances <= upper_bound
        
    elif method == 'mad':
        median = np.nanmedian(distances)
        mad = np.nanmedian(np.abs(distances - median))
        if mad == 0:
            mad = 1e-10
        upper_bound = median + threshold * mad * 1.4826  # Scale factor
        valid = distances <= upper_bound
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return valid


def cc_iucn(
    lon: NDArray,
    lat: NDArray,
    range_polygon: NDArray = None
) -> NDArray[np.bool_]:
    """
    Flag points outside species range polygon.
    
    Placeholder for IUCN range map comparison.
    Requires a polygon defining the species range.
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
    range_polygon : ndarray
        Nx2 array of polygon vertices (lon, lat)
        
    Returns
    -------
    valid : ndarray of bool
        True if inside range polygon
    """
    if range_polygon is None:
        # No polygon provided, pass all
        return np.ones(len(lon), dtype=bool)
    
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    # Simple point-in-polygon using ray casting
    def point_in_polygon(x, y, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    valid = np.array([
        point_in_polygon(lo, la, range_polygon) 
        for lo, la in zip(lon, lat)
    ])
    
    return valid


def cc_sea(
    lon: NDArray,
    lat: NDArray,
    land_mask: Optional[NDArray] = None,
    land_mask_extent: Optional[Tuple[float, float, float, float]] = None,
    ref_country: str = None
) -> NDArray[np.bool_]:
    """
    Flag coordinates on land (for marine species) or in sea (for terrestrial).
    
    For marine species, returns True if coordinate is likely in water.
    Uses a simple heuristic based on reference datasets or user-provided mask.
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
    land_mask : ndarray, optional
        2D boolean array where True = land. If provided, uses this for checking.
    land_mask_extent : tuple, optional
        (lon_min, lon_max, lat_min, lat_max) for the land_mask
    ref_country : str, optional
        Not yet implemented - would use country polygons
        
    Returns
    -------
    valid : ndarray of bool
        True if coordinate is in water (for marine species)
        
    Notes
    -----
    Without a land mask, uses a simple heuristic: checks if coordinates
    are near known major landmasses based on rough bounding boxes.
    For accurate results, provide a proper land mask (e.g., from Natural Earth
    or GSHHG coastline data).
    
    For marine species distribution modeling, consider using:
    - marineregions.org for ocean boundary data
    - ETOPO1 bathymetry (elevation < 0 = ocean)
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    
    # If user provided a land mask, use it
    if land_mask is not None and land_mask_extent is not None:
        lon_min, lon_max, lat_min, lat_max = land_mask_extent
        nrows, ncols = land_mask.shape
        
        # Convert coordinates to grid indices
        col = ((lon - lon_min) / (lon_max - lon_min) * ncols).astype(int)
        row = ((lat_max - lat) / (lat_max - lat_min) * nrows).astype(int)
        
        # Clip to valid range
        col = np.clip(col, 0, ncols - 1)
        row = np.clip(row, 0, nrows - 1)
        
        # Check if on land (invert for marine = valid)
        on_land = land_mask[row, col]
        return ~on_land  # True if in water
    
    # Simple heuristic without external data:
    # For Pacific Northwest marine species, basic sanity check
    # that coordinates aren't deep inland
    
    # Very rough check: if lat > 60 and lon > -130, likely Arctic Ocean = OK
    # This is just a placeholder - real implementation needs coastline data
    
    # Default: assume valid (can't verify without land data)
    # Log a warning
    import warnings
    warnings.warn(
        "cc_sea called without land_mask - cannot verify land/sea. "
        "Provide a land mask for accurate results.",
        UserWarning
    )
    
    return np.ones(len(lon), dtype=bool)


def clean_coordinates(
    lon: NDArray,
    lat: NDArray,
    tests: list = None,
    return_details: bool = False
) -> NDArray[np.bool_]:
    """
    Run all coordinate cleaning tests.
    
    Parameters
    ----------
    lon : ndarray
        Longitude values
    lat : ndarray
        Latitude values
    tests : list
        List of test names to run. Default runs all:
        ['val', 'zero', 'equ', 'gbif', 'inst', 'cap']
    return_details : bool
        If True, return dict of individual test results
        
    Returns
    -------
    valid : ndarray of bool
        True if passes all tests
    details : dict (if return_details=True)
        Results of each individual test
    """
    if tests is None:
        tests = ['val', 'zero', 'equ', 'gbif', 'inst', 'cap']
    
    test_funcs = {
        'val': cc_val,
        'zero': cc_zero,
        'equ': cc_equ,
        'dupl': cc_dupl,
        'round': cc_round,
        'gbif': cc_gbif,
        'inst': cc_inst,
        'cap': cc_cap,
        'sea': cc_sea,
    }
    
    results = {}
    valid = np.ones(len(lon), dtype=bool)
    
    for test in tests:
        if test in test_funcs:
            results[test] = test_funcs[test](lon, lat)
            valid &= results[test]
    
    if return_details:
        return valid, results
    return valid
