#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:03:08 2024

@author: evelynm

------

Formatting exposure layers for the displacement risk computations
"""

import fiona
from climada.entity.exposures import Exposures
from climada.util import coordinates as u_coords
import shapely
from shapely.geometry import box
import pyproj
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from climada.util.constants import SYSTEM_DIR

# =============================================================================
# CONSTANTS
# =============================================================================

DATA_DIR = Path.cwd() / "data"

# # TODO: replace by paths on cluster

# # global high resolution settlement layer
#path_ghsl = '/Users/evelynm/Documents/UNU_IDMC/data/exposure/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif'
path_ghsl = DATA_DIR/"exposure"/"GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0"/"GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif"

# BEM values residential
#path_bem_res = '/Users/evelynm/Documents/UNU_IDMC/data/exposure/bem_global_raster/bem_1x1_valfis_res.tif'
path_bem_res = DATA_DIR/"exposure"/"bem_global_raster"/"bem_1x1_valfis_res.tif"

# # grid for BEM
# Local path (outside project)
path_grid_tif = Path('/Users/simonameiler/climada/data/centroids/grid_1x1_gid.tif')

# # folder for country bem files with sub-components
path_cntry_bem = str(DATA_DIR/"exposure"/"bem_cntry_files") + '/'

# Local path (outside project)
path_admin1_attrs = Path('/Users/simonameiler/climada/data/centroids/grid_1x1_admin.csv')

# # source and destination projections
proj_54009 = pyproj.crs.CRS.from_string('esri:54009')
proj_4326 = pyproj.crs.CRS(4326)

# =============================================================================
# =============================================================================

def exp_from_bem_res(cntry_name):
    """
    country exposure on residential building values from the BEM

    Parameters
    ----------
    cntry_name : str
        ISO3 or long name of country

    Returns
    -------
    exp_ghsl : climada.entity.Exposures()

    """
    cntry_iso = u_coords.country_to_iso(cntry_name)
    geom_cntry = shapely.ops.unary_union(
        [geom for geom in
         u_coords.get_country_geometries([cntry_iso]).geometry])

    # Load BEM Exposure (res)
    exp_bem_res = Exposures.from_raster(
        path_bem_res, geometry=[geom_cntry])
    exp_bem_res.gdf = gpd.GeoDataFrame(
        exp_bem_res.gdf,
        geometry=gpd.points_from_xy(
            exp_bem_res.gdf.longitude, exp_bem_res.gdf.latitude),
        crs="EPSG:4326")
    exp_bem_res.value_unit = 'Residential building value (USD)'

    return exp_bem_res


def exp_from_ghsl(cntry_name):
    """
    country exposure on population counts from the global human settlement layer

    Parameters
    ----------
    cntry_name : str
        ISO3 or long name of country

    Returns
    -------
    exp_ghsl : climada.entity.Exposures()
    """
    cntry_iso = u_coords.country_to_iso(cntry_name)
    geom_cntry = shapely.ops.unary_union(
        [geom for geom in
         u_coords.get_country_geometries([cntry_iso]).geometry])

    exp_ghsl = Exposures.from_raster(
        path_ghsl, src_crs=proj_54009, dst_crs=proj_4326, geometry=[geom_cntry])

    exp_ghsl.gdf = gpd.GeoDataFrame(
        exp_ghsl.gdf,
        geometry=gpd.points_from_xy(
            exp_ghsl.gdf.longitude, exp_ghsl.gdf.latitude),
        crs="EPSG:4326")
    exp_ghsl.value_unit = 'Population count'

    return exp_ghsl


def gdf_from_bem_subcomps(cntry_name, opt='full'):
    """
    country gdfs on various variables from BEM sub-components, with matching
    centroids.

    Options:
        - 'pop_per_btype' : population count per seismic building type and gridpoint.
        - 'sec_per_btype' : economic sector per seismic building type and gridpoint.
        - 'full' : full gdf, no regrouping or deletion
        - 'per_grid' : cpx class per gridpoint, pop count sum per gridpoint,
            valfis sum per gridpoint, avg. share of 1, 2, 3 floors,
            mode of building code, mode of sector


    Parameters
    ----------
    cntry_name : str
    opt : str
    """
    cntry_iso = u_coords.country_to_iso(cntry_name)
    path_cntry_bem_csv = f'{path_cntry_bem}{cntry_iso.lower()}_bem_1x1_valfis.csv'

    geom_cntry = shapely.ops.unary_union(
        [geom for geom in
         u_coords.get_country_geometries([cntry_iso]).geometry])

    # load country csv as df
    df_bem_subcomps = pd.read_csv(path_cntry_bem_csv)

    # delete unnecessary columns (based on UNEP-GRID feedback)
    if 'bs_value_nr' in df_bem_subcomps.columns:
        df_bem_subcomps.pop('bs_value_nr')
        df_bem_subcomps.pop('bs_value_r')

    # delete columns with no economic and human value
    if 'valhum' in df_bem_subcomps.columns:
        df_bem_subcomps = df_bem_subcomps[(
            (df_bem_subcomps['valhum'] > 0) & (df_bem_subcomps['valfis'] > 0))]

    # load centroids
    gdf_grid = _centr_from_raster(cntry_name)

    # assign centroids to df
    df_bem_subcomps = _assign_centr2df(
        df_bem_subcomps, gdf_grid)

    if opt == 'full':
        return df_bem_subcomps

    if opt == 'per_grid':
        return _agg_gdf_per_gridpoint(df_bem_subcomps)

    return df_bem_subcomps


def _centr_from_raster(cntry_name):
    """
    load the 1x1km centroids on which the BEM is defined into a gdf
    gid column is labelled id_1x

    Returns
    -------
    gdf_centr : gpd.GeoDataFrame
    """
    cntry_iso = u_coords.country_to_iso(cntry_name)
    geom_cntry = shapely.ops.unary_union(
        [geom for geom in
         u_coords.get_country_geometries([cntry_iso]).geometry])

    meta, value = u_coords.read_raster(path_grid_tif,
                                       geometry=[box(*geom_cntry.bounds)])

    ulx, xres, _, uly, _, yres = meta['transform'].to_gdal()
    lrx = ulx + meta['width'] * xres
    lry = uly + meta['height'] * yres
    x_grid, y_grid = np.meshgrid(np.arange(ulx + xres / 2, lrx, xres),
                                 np.arange(uly + yres / 2, lry, yres))
    gdf_centr = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x_grid.flatten(), y_grid.flatten()),
        crs='epsg:4326')

    gdf_centr['id_1x'] = value.reshape(-1)

    return gdf_centr


def _assign_centr2df(df, gdf_grid):
    """
    combine bem-subcomponent df with centroids by id_1x (gid)

    Returns
    -------
    gpd.GeoDataFrame
    """
    return gpd.GeoDataFrame(df.merge(gdf_grid, on='id_1x', how='left'))


def _agg_gdf_per_gridpoint(gdf):
    return gpd.GeoDataFrame(gdf.groupby('id_1x').agg(
        {'valfis': 'sum',
         'valhum': 'sum',
         'cpx': 'mean',
         'bd_1_floor': 'mean',
         'bd_2_floor': 'mean',
         'bd_3_floor': 'mean',
         'geometry': 'first',
         'se_seismo': pd.Series.mode,
         'sector': pd.Series.mode
         }), crs=gdf.crs)


def assign_admin1_attr(gdf_bem, path_admin1_attrs, source):
    """
    Parameters
    ----------
    path_admin1_attrs : str
        path to grid_1x1_admin.csv
    gdf_bem : gpd.GeoDataFrame
        gdf loaded from country BEM data
    source : str
        'unmap' or 'gadm'. Which admin1 categorization source to use.
    """

    df_admin1_sel = _read_country_chunks(
        path_admin1_attrs, (gdf_bem['id_1x'].min(), gdf_bem['id_1x'].max()))
    gdf_bem = _assign_admin1_attr(gdf_bem, df_admin1_sel, source)
    return gdf_bem


def _read_country_chunks(path_admin1_attrs, gid_bounds):
    """
    Parameters
    ----------
    path_admin1_attrs : str
        path to grid_1x1_admin.csv
    gid_bounds : (int, int)
        min and max of id_1x column of BEM-gdf
    """
    gid_min, gid_max = gid_bounds
    chunk_list = []
    for chunk in pd.read_csv(path_admin1_attrs, sep=',', chunksize=50000):
        chunk_list.append(chunk[((chunk.gid >= gid_min) &
                                (chunk.gid <= gid_max))])

    return pd.concat(chunk_list)


def _assign_admin1_attr(gdf_bem, df_admin1_sel, source):
    """
    Parameters
    ----------
    gdf_bem : gpd.GeoDataFrame
        gdf loaded from country BEM data
    df_admin1_sel : pd.DataFrame
        dataframe with admin 1 categorizations per gid (id_1x) attribute of centroids
    source : str
        'unmap' or 'gadm'. Which admin1 categorization source to use.
    """

    val_col = 'unmap_2020_adm1_gid' if source == 'unmap' else 'gadm_410_adm1_fid'

    gdf_bem = pd.merge(
        gdf_bem, df_admin1_sel[['gid', val_col]], left_on='id_1x', right_on='gid')
    gdf_bem.pop('gid')
    gdf_bem.rename({val_col: 'admin1'}, axis=1, inplace=True)
    return gdf_bem
