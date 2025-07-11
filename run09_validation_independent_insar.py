#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 11:55:15 2025

@author: daniellelindsay
"""

from NC_ALOS2_filepaths import (common_paths, paths_068, paths_169, paths_170, paths_170_5_28, paths_115, paths_gps)
import insar_utils as utils
import pygmt
import pandas as pd
import numpy as np

unit = 100

# set your bin size in cm/yr
#bin_size = 0.01

#ref_station = common_paths["ref_station"]

ref_station = "P208"


asc_des_min_lat = 38.9 - 0.1
asc_des_max_lat = 39.1 + 0.1
asc_des_min_lon = -122.2 - 0.07
asc_des_max_lon = -122.0 + 0.13
region_asc = f"{asc_des_min_lon}/{asc_des_max_lon}/{asc_des_min_lat}/{asc_des_max_lat}"
vel_region_asc = [-10, 1, -10, 1]  # for scatter plot axes

# For Descending comparison
map_size_des = "M2.6c"
scatter_size_des = "X5.1/5.1c"

des_min_lat = 37.0
des_max_lat = 42.2
des_min_lon = -124.4
des_max_lon = -121.0
region_des = f"{des_min_lon}/{des_max_lon}/{des_min_lat}/{des_max_lat}"
vel_region_des = [-3.5, 3.5, -3.5, 3.5]



def prepare_heatmap_xyz(df, x_col, y_col, bin_size=0.1):
    min_val = df[[x_col, y_col]].min().min(); max_val = df[[x_col, y_col]].max().max()
    edges = np.arange(np.floor(min_val/bin_size)*bin_size,
                      np.ceil(max_val/bin_size)*bin_size + bin_size,
                      bin_size)
    counts, xedges, yedges = np.histogram2d(df[x_col], df[y_col], bins=[edges, edges])
    x_centers = (xedges[:-1] + xedges[1:]) / 2; y_centers = (yedges[:-1] + yedges[1:]) / 2
    heatmap_df = pd.DataFrame(counts,
                              index=pd.Index(x_centers, name=f"{x_col}_bin"),
                              columns=pd.Index(y_centers, name=f"{y_col}_bin"))
    return (
        heatmap_df
        .stack()
        .reset_index(name='count')
        .loc[lambda d: d['count'] > 0]
        .assign(count=lambda d: d['count'] / d['count'].max())
        .sort_values('count')
    )


gps_df = utils.load_UNR_gps(paths_gps["170_enu"], ref_station)
# Set lat and lon for plotting from the gps file. 
ref_lat = gps_df.loc[gps_df["StaID"] == ref_station, "Lat"].values
ref_lon = gps_df.loc[gps_df["StaID"] == ref_station, "Lon"].values

##############################
# Prepare asc and des comparison 
##############################

# Load Ascending (track 068) data
asc_lon, asc_lat, asc_vel, asc_azi, asc_inc = utils.load_insar_vel_data_as_2Darrays(
    paths_068["P208"]["geo_geometryRadar"],
    paths_068["P208"]["geo_velocity_msk"]
)

# Load Descending (track 170) data
des_lon, des_lat, des_vel, des_azi, des_inc = utils.load_insar_vel_data_as_2Darrays(
    paths_170["P208"]["geo_geometryRadar"],
    paths_170["P208"]["geo_velocity_msk"]
)

# Convert velocities
asc_vel = asc_vel * unit
des_vel = des_vel * unit

# Build DataFrame and filter for valid points within the region
df_asc_des = pd.DataFrame({
    'asc_lon': asc_lon.ravel(),
    'asc_lat': asc_lat.ravel(),
    'des_lon': des_lon.ravel(),
    'des_lat': des_lat.ravel(),
    'asc_inc': asc_inc.ravel(),
    'des_inc': des_inc.ravel(),
    'asc_azi': asc_azi.ravel(),
    'des_azi': des_azi.ravel(),
    'asc_vel': asc_vel.ravel(),
    'des_vel': des_vel.ravel(),
})
df_asc_des = df_asc_des.dropna(subset=['asc_vel', 'des_vel'])
df_asc_des = df_asc_des[
    (df_asc_des['asc_lat'] >= asc_des_min_lat) & (df_asc_des['asc_lat'] <= asc_des_max_lat) &
    (df_asc_des['asc_lon'] >= asc_des_min_lon) & (df_asc_des['asc_lon'] <= asc_des_max_lon) &
    (df_asc_des['des_lat'] >= asc_des_min_lat) & (df_asc_des['des_lat'] <= asc_des_max_lat) &
    (df_asc_des['des_lon'] >= asc_des_min_lon) & (df_asc_des['des_lon'] <= asc_des_max_lon)
]

# Project velocities to vertical (using the incidence angles)
df_asc_des['asc_v'] = df_asc_des['asc_vel'] * np.cos(np.deg2rad(df_asc_des['asc_inc']))
df_asc_des['des_v'] = df_asc_des['des_vel'] * np.cos(np.deg2rad(df_asc_des['des_inc']))

# compute medians
med_des = np.nanmedian(df_asc_des.loc[
    df_asc_des['des_lat'].between(38.85, 38.90) &
    df_asc_des['des_lon'].between(-122.25, -122.20),
    'des_v'
])

med_asc = np.nanmedian(df_asc_des.loc[
    df_asc_des['asc_lat'].between(38.85, 38.90) &
    df_asc_des['asc_lon'].between(-122.25, -122.20),
    'asc_v'
])

# Compute statistics (e.g., RMSE, R², and slope)
rmse_asc, r2_asc, slope_asc, intercept_asc = utils.calculate_rmse_r2_and_linear_fit(
    df_asc_des['asc_v'], df_asc_des['des_v']
)
print(f"Asc/Desc: RMSE {rmse_asc:.2f}, R² {r2_asc:.2f}, slope {slope_asc:.2f}")

# Generate 2D histogram for plotting
xyz_df_asc_des = prepare_heatmap_xyz(df_asc_des, 'des_v', 'asc_v', bin_size=0.01)

##############################
# Prepare 170, 169 comparison
##############################

# Load deramped descending data for tracks 169 and 170
des169_lon, des169_lat, des169_vel, des169_azi, des169_inc = utils.load_insar_vel_data_as_2Darrays(
    paths_169["geo"]["geo_geometryRadar"],
    paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]
)
des170_lon, des170_lat, des170_vel, des170_azi, des170_inc = utils.load_insar_vel_data_as_2Darrays(
    paths_170["geo"]["geo_geometryRadar"],
    paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]
)

# Convert velocities
des169_vel = des169_vel * unit
des170_vel = des170_vel * unit

# Build DataFrame for descending comparison and filter valid data
df_des_diff = pd.DataFrame({
    'des169_vel': des169_vel.ravel(),
    'des170_vel': des170_vel.ravel(),
    })
df_des_diff = df_des_diff.dropna(subset=['des169_vel', 'des170_vel'])

rmse_des, r2_des, slope_des, intercept_des = utils.calculate_rmse_r2_and_linear_fit(
    df_des_diff['des169_vel'], df_des_diff['des170_vel']
)
print(f"Des 169/170: RMSE {rmse_des:.2f}, R² {r2_des:.2f}, slope {slope_des:.2f}")

# Generate 2D histogram for plotting
xyz_df_des_des = prepare_heatmap_xyz(df_des_diff, 'des169_vel', 'des170_vel', bin_size=0.01)


############################
# Start Plotting 
#############################3
# Define plotting parameters for each block
# For Ascending/Descending comparison
map_size_asc = "M4c"
scatter_size_asc = "X5.1/5.1c"



fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=9, FONT_TITLE=10, MAP_TITLE_OFFSET="-7p")

# --- First subplot block: 3 panels for Asc/Desc (Tracks 068 vs. 170) ---
with fig.subplot(nrows=1, ncols=3, figsize=("12.5c", "5.1c"), autolabel="a)", sharex="l", frame=["WSrt"], margins=["0.3c", "0.3c"]):
    
    
    vmin, vmax = -0.085, 0.01
    
    # Panel 1: Ascending map (Track 068)
    fig.basemap(frame=["WSrt", "xa", "ya"], region=region_asc, projection=map_size_asc, panel=True)
    pygmt.makecpt(cmap="magma", series=[vmin+med_asc/unit, vmax+med_asc/unit])
    fig.grdimage(grid=paths_068["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], cmap=True, region=region_asc, projection=map_size_asc)
    fig.coast(shorelines=True, region=[region_asc], projection=map_size_asc, frame=["+tAscending Up 068"])
    fig.plot(x=ref_lon, y=ref_lat, style="s.2c", fill="black", pen="1p")
    fig.text(text="117 Interferograms", position="BL", offset="0.2c/0.2c",
             font="9p,Helvetica,black", region=region_asc, projection=map_size_asc)
    fig.text(text="27 Acquisitions", position="BL", offset="0.2c/0.7c",
             font="9p,Helvetica,black", region=region_asc, projection=map_size_asc)
    df = pd.DataFrame(
        data={
            "x": [-121.94,-121.94 ],
            "y": [38.825,38.825],
            "east_velocity": [-0.173*0.8, (0.9848/2.5)*0.8],
            "north_velocity": [0.9848*0.8, (0.173/2.5)*0.8],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.5p,black", line=True, spec="e1.2/1/1", vector="0.3c+p2p+e+gblack",region=region_asc, projection=map_size_asc)

    fig.basemap(frame=["WSrt", "xa", "ya"], map_scale="jTR+w5k+o0.2c/0.2c",
                region=region_asc, projection=map_size_asc)
    
    # Panel 2: Descending map (Track 170)
    fig.basemap(frame=["wSrt", "xa", "ya"], region=region_asc, projection=map_size_asc, panel=True)
    pygmt.makecpt(cmap="magma", series=[vmin+med_des/unit, vmax+med_des/unit])
    fig.grdimage(grid=paths_170["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], cmap=True, region=region_asc, projection=map_size_asc)
    fig.coast(shorelines=True, region=[region_asc], projection=map_size_asc, frame=["+tDescending Up 170"])
    fig.plot(x=ref_lon, y=ref_lat, style="s.2c", fill="black", pen="1p")
    fig.text(text="561 Interferograms", position="BL", offset="0.2c/0.2c",
             font="9p,Helvetica,black", region=region_asc, projection=map_size_asc)
    fig.text(text="91 Acquisitions", position="BL", offset="0.2c/0.7c",
             font="9p,Helvetica,black", region=region_asc, projection=map_size_asc)
    df = pd.DataFrame(
        data={
            "x": [-121.905,-121.905 ],
            "y": [38.89,38.89],
            "east_velocity": [-0.173*0.8,(-0.9848/2.5)*0.8],
            "north_velocity": [-0.9848*0.8, (0.173/2.5)*0.8],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.5p,black", line=True, spec="e1.2/1/1", vector="0.3c+p2p+e+gblack",region=region_asc, projection=map_size_asc)

    fig.basemap(frame=["wSrt", "xa", "ya"], map_scale="jTR+w5k+o0.2c/0.2c",
                region=region_asc, projection=map_size_asc)
    
    # Panel 3: Scatter plot (Vertical components: Ascending vs. Descending)
    fig.basemap(frame=["lSEt", "xaf+lDescending Up (cm/yr)", "yaf+lAscending Up (cm/yr)"],
                region=vel_region_asc, projection=scatter_size_asc, panel=True)
    # fig.plot(x=df_asc_des['asc_v'], y=df_asc_des['des_v'], style="c0.05c",
    #          fill="black", transparency=80, region=vel_region_asc, projection=scatter_size_asc)
    
    pygmt.makecpt(cmap="gray", series=[-0.1, 0.3, 0.05], reverse=True)
    fig.plot(x=xyz_df_asc_des['des_v_bin'], y=xyz_df_asc_des['asc_v_bin'], style="s0.05c",
             fill=xyz_df_asc_des["count"], cmap=True, region=vel_region_asc, projection=scatter_size_asc)
             
    fig.plot(x=[-10,1], y=[-10,1], pen="1p,black", region=vel_region_asc, projection=scatter_size_asc)
    fig.text(text="RMSE {} cm/yr".format(np.round(rmse_asc,2)), position="BR",
             offset="-0.2c/0.65c", font="9p,Helvetica,black", region=vel_region_asc, projection=scatter_size_asc)
    fig.text(text="R² {}, Slope {}".format(np.round(r2_asc,2), np.round(slope_asc,2)), position="BR",
             offset="-0.2c/0.2c", font="9p,Helvetica,black", region=vel_region_asc, projection=scatter_size_asc)
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black"):
        pygmt.makecpt(cmap="magma", series=[vmin*unit, vmax*unit])
        fig.colorbar(position="jCL+o0.25c/0.5c+w2.7c/0.3c",
                     frame=["xaf", "y+lcm/yr"], region=region_asc, projection=map_size_asc)

# Shift the origin to place the second block below the first
fig.shift_origin(yshift="-6.5c", xshift="0c")

# --- Second subplot block: 4 panels for Descending Comparison (Tracks 169 vs. 170) ---
with fig.subplot(nrows=1, ncols=4, figsize=("11c", "5.1c"), autolabel="d)", sharex="l", frame=["WSrt"], margins=["0.3c", "0.3c"]):
    
    vmin, vmax = -0.02, 0.02
    
    # Panel 1: Descending 169 map
    fig.basemap(frame=["WSrt", "xa", "ya"], region=region_des, projection=map_size_des, panel=True)
    pygmt.makecpt(cmap="vik", series=[vmin, vmax])
    fig.grdimage(grid=paths_169["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], cmap=True, region=[region_des], projection=map_size_des)
    fig.coast(shorelines=True, region=[region_des], projection=map_size_des, frame=["+t169: 2015-2021"])
    
    fig.plot(y=ref_lat, x=ref_lon, style="s.1c", fill="black", pen="0.8p,black", region=[region_des], projection=map_size_des,)
    
    df = pd.DataFrame(
        data={
            "x": [-123.5,-123.5 ],
            "y": [38.2,38.2],
            "east_velocity": [-0.173*0.8,(-0.9848/2.5)*0.8],
            "north_velocity": [-0.9848*0.8, (0.173/2.5)*0.8],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.5p,black", line=True, spec="e1.2/1/1", vector="0.3c+p2p+e+gblack",region=region_des, projection=map_size_des)

    fig.basemap(frame=["WSrt", "xa", "ya"], region=region_des, projection=map_size_des)
    
    # Panel 2: Descending 170 map
    fig.basemap(frame=["wSrt", "xa", "ya"], region=region_des, projection=map_size_des, panel=True)
    pygmt.makecpt(cmap="vik", series=[vmin, vmax])
    fig.grdimage(grid=paths_170["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], cmap=True, region=[region_des], projection=map_size_des)
    fig.coast(shorelines=True, region=[region_des], projection=map_size_des, frame=["+t170: 2015-2024"])
    fig.plot(y=ref_lat, x=ref_lon, style="s.1c", fill="black", pen="0.8p,black", region=[region_des], projection=map_size_des,)
    
    df = pd.DataFrame(
        data={
            "x": [-123.5,-123.5 ],
            "y": [38.2,38.2],
            "east_velocity": [-0.173*0.8,(-0.9848/2.5)*0.8],
            "north_velocity": [-0.9848*0.8, (0.173/2.5)*0.8],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.5p,black", line=True, spec="e1.2/1/1", vector="0.3c+p2p+e+gblack",region=region_des, projection=map_size_des)
    
    fig.basemap(frame=["wSrt", "xa", "ya"],
                region=region_des, projection=map_size_des)
    
    # Panel 3: Difference map (Des 169 minus Des 170)
    fig.basemap(frame=["wSrt", "xa", "ya"], region=region_des, projection=map_size_des, panel=True)
    pygmt.makecpt(cmap="vik", series=[vmin, vmax])
    fig.grdimage(grid=paths_169["grd"]["diff_169_170"], cmap=True, region=[region_des], projection=map_size_des)
    fig.coast(shorelines=True, region=[region_des], projection=map_size_des, frame=["+tDifference"])

    fig.plot(y=ref_lat, x=ref_lon, style="s.1c", fill="black", pen="0.8p,black", region=[region_des], projection=map_size_des,)
    
    fig.plot(x = [asc_des_min_lon, asc_des_min_lon, asc_des_max_lon, asc_des_max_lon, asc_des_min_lon], 
             y = [asc_des_min_lat, asc_des_max_lat, asc_des_max_lat, asc_des_min_lat, asc_des_min_lat], 
             pen="0.5p,black", transparency=0, region=[region_des], projection=map_size_des,)
    
    # Inset map of frames
    # Define region of interest 
    min_lon=-126.5
    max_lon=-119.0
    min_lat=35.5
    max_lat=43

    sub_region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
    
    fig.coast(shorelines=True, area_thresh=5000, frame="+t", projection="M1.1c", region=[sub_region])
    fig.plot(data=common_paths['frames']['170_2800'] , pen="0.5p,black", transparency=30, projection="M1.1c", region=[sub_region])
    fig.plot(data=common_paths['frames']['170_2850'] , pen="0.5p,black", transparency=30, projection="M1.1c", region=[sub_region])
    fig.plot(data=common_paths['frames']['169_2800'] , pen="0.5p,black", transparency=30, projection="M1.1c", region=[sub_region])
    fig.plot(data=common_paths['frames']['169_2850'] , pen="0.5p,black", transparency=30, projection="M1.1c", region=[sub_region])
    fig.plot(data=common_paths['frames']['169_170_overlap'] , pen="0.5p,black", transparency=30, fill="grey", projection="M1.1c", region=[sub_region])
    fig.coast(shorelines=True, area_thresh=5000, frame="+t", projection="M1.1c", region=[sub_region])
    fig.basemap(frame=["wSrt", "xa", "ya"], region=region_des, projection=map_size_des)
    
    # # Panel 4: Scatter plot (Des 169 vs. Des 170)
    fig.basemap(frame=["lSEt", "xaf+lDescending 169 (cm/yr)", "yaf+lDescending 170 (cm/yr)"],
                region=vel_region_des, projection=scatter_size_des, panel=True)
    
    pygmt.makecpt(cmap="gray", series=[-0.1, 0.3, 0.05], reverse=True)
    fig.plot(x=xyz_df_des_des['des169_vel_bin'], y=xyz_df_des_des['des170_vel_bin'], style="s0.05c",
              fill=xyz_df_des_des["count"], cmap=True, region=vel_region_des, projection=scatter_size_des)
    fig.plot(x=[-3.5,3.5], y=[-3.5,3.5], pen="0.5p,black", region=vel_region_des, projection=scatter_size_des)
    
    fig.text(text="RMSE {} cm/yr".format(np.round(rmse_des,2)), position="BR",
             offset="-0.2c/0.65c", font="9p,Helvetica,black", region=vel_region_des, projection=scatter_size_des)
    fig.text(text="R² {}, Slope {}".format(np.round(r2_des,2), np.round(slope_des,2)), position="BR",
             offset="-0.2c/0.2c", font="9p,Helvetica,black", region=vel_region_des, projection=scatter_size_des)
    
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black"):
        pygmt.makecpt(cmap="vik", series=[-2, 2])
        fig.colorbar(position="jCL+o0.25c/0.5c+w2.7c/0.3c",
                     frame=["xaf", "y+lcm/yr"], region=region_des, projection=map_size_des)
        
    fig.basemap(frame=["lSEt", "xaf+lDescending 169 (cm/yr)", "yaf+lDescending 170 (cm/yr)"],
                region=vel_region_des, projection=scatter_size_des)

# Save and display the final figure
fig.savefig(common_paths["fig_dir"]+f'Fig_7_insar_insar_validation_{ref_station}.png', transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+f'Fig_7_insar_insar_validation_{ref_station}.pdf', transparent=False, crop=True, anti_alias=True, show=False)
fig.show()