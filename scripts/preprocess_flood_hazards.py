#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess flood hazards for Bangladesh

This script crops global flood hazard rasters to Bangladesh extent and saves
them as CLIMADA HDF5 files for efficient loading in subsequent analyses.

Run on cluster with: sbatch sbatch_preprocess_flood_hazards.sh
"""

from pathlib import Path
from flood_hazard import preprocess_and_save_hazard

# Set paths for cluster
HAZARD_DIR = Path('/home/groups/bakerjw/smeiler/climada_data/data/hazard/floods')
SHAPEFILE_PATH = Path('/home/groups/bakerjw/smeiler/climada_data/data/exposure/shapefiles/gadm41_BGD_shp')
OUTPUT_DIR = Path('/home/groups/bakerjw/smeiler/climada_data/data/hazard/BGD')

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
