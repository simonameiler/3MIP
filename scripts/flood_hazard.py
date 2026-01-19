#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in January 2026

description: Make coastal and riverine flood hazard layers for Bangladesh

@author: simonameiler
"""

import os
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

from climada.hazard import Hazard
from climada.hazard.centroids import Centroids


def load_bgd_shapefile(shapefile_path=None, admin_level=0):
    """
    Load Bangladesh shapefile for cropping hazard maps.
    
    Parameters:
    -----------
    shapefile_path : str or Path, optional
        Path to the GADM shapefile directory. If None, uses default path.
    admin_level : int, default=0
        Administrative level (0=country, 1=division, 2=district, etc.)
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with Bangladesh geometry
    """
    if shapefile_path is None:
        shapefile_path = Path('data/shapefiles/gadm41_BGD_shp')
    
    shapefile_path = Path(shapefile_path)
    shp_file = shapefile_path / f'gadm41_BGD_{admin_level}.shp'
    
    if not shp_file.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_file}")
    
    gdf = gpd.read_file(shp_file)
    return gdf


def get_coastal_flood_files(hazard_dir, return_periods=None):
    """
    Get coastal flood hazard file paths for specified return periods.
    
    Parameters:
    -----------
    hazard_dir : str or Path
        Directory containing hazard raster files
    return_periods : list of int, optional
        Return periods to include. If None, uses [2, 5, 10, 25, 50, 100, 250, 500, 1000]
    
    Returns:
    --------
    tuple: (list of file paths, list of return periods, list of frequencies)
    """
    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
    
    hazard_dir = Path(hazard_dir)
    files = []
    rps_found = []
    
    # Coastal flood files have format: inuncoast_historical_nosub_hist_rp{RP:04d}_0.tif
    # Return periods are stored with 4 digits and decimal part (e.g., rp0002_0.tif for 2-year)
    for rp in return_periods:
        pattern = f"inuncoast_historical_nosub_hist_rp{rp:04d}_0.tif"
        file_path = hazard_dir / pattern
        
        if file_path.exists():
            files.append(str(file_path))
            rps_found.append(rp)

    if not files:
        raise FileNotFoundError(f"No coastal flood files found in {hazard_dir}")
    
    # Calculate frequencies (1/return period)
    frequencies = [1.0 / rp for rp in rps_found]
    
    return files, rps_found, frequencies


def get_riverine_flood_files(hazard_dir, return_periods=None):
    """
    Get riverine flood hazard file paths for specified return periods.
    
    Parameters:
    -----------
    hazard_dir : str or Path
        Directory containing hazard raster files
    return_periods : list of int, optional
        Return periods to include. If None, uses [2, 5, 10, 25, 50, 100, 250, 500, 1000]
    
    Returns:
    --------
    tuple: (list of file paths, list of return periods, list of frequencies)
    """
    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
    
    hazard_dir = Path(hazard_dir)
    files = []
    rps_found = []
    
    # Riverine flood files have format: inunriver_historical_000000000WATCH_1980_rp{RP:05d}.tif
    # Note: River flood carries an extra 0 (5 digits total)
    for rp in return_periods:
        pattern = f"inunriver_historical_000000000WATCH_1980_rp{rp:05d}.tif"
        file_path = hazard_dir / pattern
        
        if file_path.exists():
            files.append(str(file_path))
            rps_found.append(rp)
    
    if not files:
        raise FileNotFoundError(f"No riverine flood files found in {hazard_dir}")
    
    # Calculate frequencies (1/return period)
    frequencies = [1.0 / rp for rp in rps_found]
    
    return files, rps_found, frequencies


def generate_coastal_flood_hazard(hazard_dir, return_periods=None, shapefile_path=None, 
                                   crop_to_country=True):
    """
    Generate coastal flood hazard object for Bangladesh.
    
    Parameters:
    -----------
    hazard_dir : str or Path
        Directory containing hazard raster files
    return_periods : list of int, optional
        Return periods to include. If None, uses [2, 5, 10, 25, 50, 100, 250, 1000]
    shapefile_path : str or Path, optional
        Path to shapefile for cropping. If None, uses default BGD shapefile
    crop_to_country : bool, default=True
        If True, crop hazard to country extent
    
    Returns:
    --------
    climada.hazard.Hazard
        Coastal flood hazard object
    """
    # Get file paths
    files, rps, frequencies = get_coastal_flood_files(hazard_dir, return_periods)
    
    print(f"Loading coastal flood hazard with {len(files)} return periods: {rps}")
    
    # Create Hazard object from raster files
    haz = Hazard.from_raster(
        haz_type='FL',
        files_intensity=files,
        src_crs='EPSG:4326',
        attrs={
            'unit': 'm',
            'event_id': np.arange(len(files)),
            'frequency': np.array(frequencies)
        }
    )
    
    # Set to default CRS
    haz.centroids.to_default_crs()
    
    # Crop to Bangladesh extent if requested
    if crop_to_country:
        gdf_bgd = load_bgd_shapefile(shapefile_path)
        bounds = gdf_bgd.total_bounds  # (minx, miny, maxx, maxy)
        
        # Get centroids within bounds
        lon = haz.centroids.lon
        lat = haz.centroids.lat
        mask = (
            (lon >= bounds[0]) & (lon <= bounds[2]) &
            (lat >= bounds[1]) & (lat <= bounds[3])
        )
        
        # Select centroids within bounds
        centroid_indices = np.where(mask)[0]
        haz = haz.select(cent=centroid_indices)
        
        print(f"Cropped to Bangladesh extent: {len(centroid_indices)} centroids")
    
    return haz


def generate_riverine_flood_hazard(hazard_dir, return_periods=None, shapefile_path=None,
                                    crop_to_country=True):
    """
    Generate riverine flood hazard object for Bangladesh.
    
    Parameters:
    -----------
    hazard_dir : str or Path
        Directory containing hazard raster files
    return_periods : list of int, optional
        Return periods to include. If None, uses [2, 5, 10, 25, 50, 100, 250, 1000]
    shapefile_path : str or Path, optional
        Path to shapefile for cropping. If None, uses default BGD shapefile
    crop_to_country : bool, default=True
        If True, crop hazard to country extent
    
    Returns:
    --------
    climada.hazard.Hazard
        Riverine flood hazard object
    """
    # Get file paths
    files, rps, frequencies = get_riverine_flood_files(hazard_dir, return_periods)
    
    print(f"Loading riverine flood hazard with {len(files)} return periods: {rps}")
    
    # Create Hazard object from raster files
    haz = Hazard.from_raster(
        haz_type='FL',
        files_intensity=files,
        src_crs='EPSG:4326',
        attrs={
            'unit': 'm',
            'event_id': np.arange(len(files)),
            'frequency': np.array(frequencies)
        }
    )
    
    # Set to default CRS
    haz.centroids.to_default_crs()
    
    # Crop to Bangladesh extent if requested
    if crop_to_country:
        gdf_bgd = load_bgd_shapefile(shapefile_path)
        bounds = gdf_bgd.total_bounds  # (minx, miny, maxx, maxy)
        
        # Get centroids within bounds
        lon = haz.centroids.lon
        lat = haz.centroids.lat
        mask = (
            (lon >= bounds[0]) & (lon <= bounds[2]) &
            (lat >= bounds[1]) & (lat <= bounds[3])
        )
        
        # Select centroids within bounds
        centroid_indices = np.where(mask)[0]
        haz = haz.select(cent=centroid_indices)
        
        print(f"Cropped to Bangladesh extent: {len(centroid_indices)} centroids")
    
    return haz


def crop_raster_to_extent(raster_path, bounds, output_path=None):
    """
    Crop a raster file to specified bounds using rasterio for memory efficiency.
    
    Parameters:
    -----------
    raster_path : str or Path
        Path to input raster file
    bounds : tuple
        Bounds (minx, miny, maxx, maxy) to crop to
    output_path : str or Path, optional
        Path to save cropped raster. If None, returns array and metadata
    
    Returns:
    --------
    tuple: (cropped_array, cropped_transform, cropped_meta) if output_path is None
           Otherwise saves to file and returns output_path
    """
    from shapely.geometry import box
    
    # Create geometry from bounds
    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    
    with rasterio.open(raster_path) as src:
        # Crop the raster
        out_image, out_transform = mask(src, [bbox], crop=True, all_touched=True)
        out_meta = src.meta.copy()
        
        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        if output_path is not None:
            # Save to file
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
            return output_path
        else:
            return out_image, out_transform, out_meta


def preprocess_and_save_hazard(flood_type='coastal', hazard_dir=None, 
                                 shapefile_path=None, output_dir=None,
                                 return_periods=None):
    """
    Pre-crop flood hazard rasters to Bangladesh extent and save as CLIMADA HDF5.
    
    This function crops global raster files to Bangladesh extent using efficient
    windowed reading, creates a CLIMADA Hazard object, and saves it to HDF5 format
    for fast loading in subsequent analyses.
    
    Parameters:
    -----------
    flood_type : str, default='coastal'
        Type of flood: 'coastal' or 'riverine'
    hazard_dir : str or Path, optional
        Directory containing hazard raster files
    shapefile_path : str or Path, optional
        Path to shapefile for getting bounds
    output_dir : str or Path, optional
        Directory to save preprocessed hazard. If None, uses 'data/hazard/BGD'
    return_periods : list of int, optional
        Return periods to include
    
    Returns:
    --------
    str: Path to saved HDF5 file
    """
    # Set default paths
    if hazard_dir is None:
        hazard_dir = Path('data/hazard')
    if output_dir is None:
        output_dir = Path('data/hazard/BGD')
    if shapefile_path is None:
        shapefile_path = Path('data/shapefiles/gadm41_BGD_shp')
    
    hazard_dir = Path(hazard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load shapefile to get bounds
    print(f"Loading shapefile to get Bangladesh bounds...")
    gdf_bgd = load_bgd_shapefile(shapefile_path)
    bounds = gdf_bgd.total_bounds
    
    # Add buffer (0.1 degrees ~ 11 km at equator)
    buffer = 0.1
    bounds = (bounds[0] - buffer, bounds[1] - buffer, 
              bounds[2] + buffer, bounds[3] + buffer)
    
    print(f"Cropping bounds: {bounds}")
    
    # Get file paths based on flood type
    if flood_type == 'coastal':
        files, rps, frequencies = get_coastal_flood_files(hazard_dir, return_periods)
        output_file = output_dir / 'flood_coastal_bgd.hdf5'
    elif flood_type == 'riverine':
        files, rps, frequencies = get_riverine_flood_files(hazard_dir, return_periods)
        output_file = output_dir / 'flood_riverine_bgd.hdf5'
    else:
        raise ValueError(f"Unknown flood_type: {flood_type}. Use 'coastal' or 'riverine'")
    
    print(f"\nProcessing {flood_type} flood with {len(files)} return periods: {rps}")
    
    # Process first file to get coordinate grid
    print(f"\nProcessing return period: {rps[0]} years...")
    out_image, out_transform, out_meta = crop_raster_to_extent(files[0], bounds)
    
    # Create coordinate arrays from the cropped raster
    height, width = out_image.shape[1], out_image.shape[2]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(out_transform, rows, cols)
    lons = np.array(xs).flatten()
    lats = np.array(ys).flatten()
    
    # Initialize intensity matrix
    n_events = len(files)
    n_centroids = len(lons)
    intensity = np.zeros((n_events, n_centroids))
    
    # Store first event intensity
    intensity[0, :] = out_image[0].flatten()
    
    # Process remaining files
    for i, (file, rp) in enumerate(zip(files[1:], rps[1:]), start=1):
        print(f"Processing return period: {rp} years...")
        out_image, _, _ = crop_raster_to_extent(file, bounds)
        intensity[i, :] = out_image[0].flatten()
    
    # Create Centroids object
    print("\nCreating CLIMADA Hazard object...")
    centroids = Centroids.from_lat_lon(lats, lons)
    
    # Create Hazard object
    haz = Hazard(
        haz_type='FL',
        centroids=centroids,
        event_id=np.arange(n_events),
        frequency=np.array(frequencies),
        intensity=intensity,
        fraction=np.ones_like(intensity)
    )
    
    # Set additional attributes
    haz.unit = 'm'
    haz.event_name = [f'RP_{rp}' for rp in rps]
    
    # Check the hazard
    haz.check()
    
    # Save to HDF5
    print(f"\nSaving to {output_file}...")
    haz.write_hdf5(output_file)
    
    print(f"\n✓ Successfully saved {flood_type} flood hazard:")
    print(f"  Events: {haz.size}")
    print(f"  Centroids: {len(haz.centroids.lat)}")
    print(f"  Return periods: {rps}")
    print(f"  File: {output_file}")
    
    return str(output_file)


def load_preprocessed_hazard(flood_type='coastal', hazard_dir=None):
    """
    Load pre-cropped hazard from HDF5 file.
    
    Parameters:
    -----------
    flood_type : str, default='coastal'
        Type of flood: 'coastal' or 'riverine'
    hazard_dir : str or Path, optional
        Directory containing preprocessed hazard files. If None, uses 'data/hazard/BGD'
    
    Returns:
    --------
    climada.hazard.Hazard
        Loaded hazard object
    """
    if hazard_dir is None:
        hazard_dir = Path('data/hazard/BGD')
    
    hazard_dir = Path(hazard_dir)
    
    if flood_type == 'coastal':
        file_path = hazard_dir / 'flood_coastal_bgd.hdf5'
    elif flood_type == 'riverine':
        file_path = hazard_dir / 'flood_riverine_bgd.hdf5'
    else:
        raise ValueError(f"Unknown flood_type: {flood_type}. Use 'coastal' or 'riverine'")
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Preprocessed hazard not found: {file_path}\n"
            f"Run preprocess_and_save_hazard() first to create it."
        )
    
    print(f"Loading preprocessed {flood_type} flood hazard from {file_path}...")
    haz = Hazard.from_hdf5(file_path)
    print(f"Loaded {haz.size} events with {len(haz.centroids.lat)} centroids")
    
    return haz


# Example usage
if __name__ == '__main__':
    # Set paths
    HAZARD_DIR = Path('data/hazard')
    SHAPEFILE_PATH = Path('data/shapefiles/gadm41_BGD_shp')
    OUTPUT_DIR = Path('data/hazard/BGD')
    
    # Option 1: Preprocess and save hazards (run once on cluster)
    print("\n" + "="*70)
    print("PREPROCESSING FLOOD HAZARDS FOR BANGLADESH")
    print("="*70)
    
    # Preprocess coastal flood
    print("\n### COASTAL FLOOD ###")
    coastal_file = preprocess_and_save_hazard(
        flood_type='coastal',
        hazard_dir=HAZARD_DIR,
        shapefile_path=SHAPEFILE_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Preprocess riverine flood
    print("\n### RIVERINE FLOOD ###")
    riverine_file = preprocess_and_save_hazard(
        flood_type='riverine',
        hazard_dir=HAZARD_DIR,
        shapefile_path=SHAPEFILE_PATH,
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nCoastal flood: {coastal_file}")
    print(f"Riverine flood: {riverine_file}")
    
    # Option 2: Load preprocessed hazards (fast!)
    print("\n" + "="*70)
    print("LOADING PREPROCESSED HAZARDS")
    print("="*70)
    
    haz_coastal = load_preprocessed_hazard('coastal', OUTPUT_DIR)
    haz_riverine = load_preprocessed_hazard('riverine', OUTPUT_DIR)
