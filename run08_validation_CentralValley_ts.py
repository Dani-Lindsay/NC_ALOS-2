#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:26:20 2025

@author: daniellelindsay
"""

import insar_utils as utils
from NC_ALOS2_filepaths import (paths_170_5_28, paths_115, common_paths)
import numpy as np
import pygmt 
import pandas as pd 



dic_170 = {"ORBIT_DIRECTION" : "descending", 
           "PATH" : "170", 
           "Platform" : "ALOS-2", 
           "Sensor" : "a2",
           "geo_file" : paths_170_5_28["CentralValley"]["geo_geometryRadar"], 
           "vel_file" : paths_170_5_28["CentralValley"]["geo_velocity_msk"],
           "ts_file" : paths_170_5_28["CentralValley"]["geo_timeseries_msk"],
           "vel_grd" : paths_170_5_28["CentralValley"]["geo_velocity_msk_grd"],
           }

dic_115 = {"ORBIT_DIRECTION" : "descending", 
           "PATH" : "115", 
           "Platform" : "Sentinel-1", 
           "Sensor" : "s1", 
           "geo_file" : paths_115["CentralValley"]["geo_geometryRadar"], 
           "vel_file" : paths_115["CentralValley"]["geo_velocity_msk"],
           "ts_file" : paths_115["CentralValley"]["geo_timeseries_msk"],
           "vel_grd" : paths_115["CentralValley"]["geo_velocity_msk_grd"],
           }

# Define region of interest 
min_lon=-122.35
max_lon=-121.8
min_lat=38.85
max_lat=39.26
region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
fig_size = "M6c"

### GPS variables 
ref_station = "P208"
tar_station = "P270"

ref_lat = 39.109
ref_lon = -122.304

tar_lat = 39.244
tar_lon = -122.055

## Profile Centre Location
center_lo = -122.092379
center_la =  39.032585

## Profile azimuth
azi_EW = 70
azi_NS = 160

# Profile distance
p_dist = 15
width = 0.5

# Start and end in decimal years
t0 = 2021.5  
te = 2024.3  

# Distance in degrees to average for InSAR LOS velocities
dist = common_paths["dist"] #~1km in each direction 0.008 = 1km. 

##################################
# Load InSAR data
##################################

insar_170 = utils.load_insar_vel_ts_as_dictionary(dic_170)
insar_170_df = pd.DataFrame({
    'Lon': insar_170['lons'].flatten(),
    'Lat': insar_170['lats'].flatten(),
    'Inc': insar_170['inc'].flatten(),
    'Az': insar_170['azi'].flatten(),
    'Vel': insar_170['vel'].flatten(),  # Uncomment if needed
})
insar_170_df.dropna(inplace=True)
    
insar_115 = utils.load_insar_vel_ts_as_dictionary(dic_115)
insar_115_df = pd.DataFrame({
    'Lon': insar_115['lons'].flatten(),
    'Lat': insar_115['lats'].flatten(),
    'Inc': insar_115['inc'].flatten(),
    'Az': insar_115['azi'].flatten(),
    'Vel': insar_115['vel'].flatten(),  # Uncomment if needed
})
insar_115_df.dropna(inplace=True)

########################################
# Load GPS data
########################################
##### Files paths to gnss data and InSAR for Central Valley
ref_file = f"{common_paths["gps_dir"]}/{ref_station}.IGS14.tenv3.csv"
tar_file = f"{common_paths["gps_dir"]}/{tar_station}.IGS14.tenv3.csv"

ref_gps_ts = pd.read_csv(ref_file, sep=",", header=1, names = ['StaID', "YYMMMDD", "yyyy", "MJD", "week", "d", "reflon", "e0", "east", "n0", "north", "u0", "up", "ant", "sig_e", "sig_n", "sig_u", "corr_en", "corr_eu", "corr_nu", "Lat", "Lon", "height"])
tar_gps_ts = pd.read_csv(tar_file, sep=",", header=1, names = ['StaID', "YYMMMDD", "yyyy", "MJD", "week", "d", "reflon", "e0", "east", "n0", "north", "u0", "up", "ant", "sig_e", "sig_n", "sig_u", "corr_en", "corr_eu", "corr_nu", "Lat", "Lon", "height"])

# Cut GPS to observation period
ref_gps_ts = ref_gps_ts [ (ref_gps_ts.yyyy > t0) & (ref_gps_ts.yyyy < te)]
tar_gps_ts = tar_gps_ts [ (tar_gps_ts.yyyy > t0) & (tar_gps_ts.yyyy < te)]

ref_gps_ts.reset_index(drop=True, inplace=True)
tar_gps_ts.reset_index(drop=True, inplace=True)

# Subtract median values. 
ref_gps_ts['east'] = ref_gps_ts['east']- ref_gps_ts['east'].median()
ref_gps_ts['north'] = ref_gps_ts['north']- ref_gps_ts['north'].median()
ref_gps_ts['up'] = ref_gps_ts['up']-ref_gps_ts['up'].median()

tar_gps_ts['east'] = tar_gps_ts['east'] - tar_gps_ts['east'].median()
tar_gps_ts['north'] = tar_gps_ts['north'] - tar_gps_ts['north'].median()
tar_gps_ts['up'] = tar_gps_ts['up'] - tar_gps_ts['up'].median()

#####################
# decimate GNSS and project UNR enu --> los
#####################

# Project ENU to LOS for daily GPS
ref_gps_ts_daily_170 = utils.calculate_gps_timeseries_los(ref_gps_ts, insar_170_df, dic_170["PATH"])
tar_gps_ts_daily_170 = utils.calculate_gps_timeseries_los(tar_gps_ts, insar_170_df, dic_170["PATH"])
ref_gps_ts_daily_115 = utils.calculate_gps_timeseries_los(ref_gps_ts, insar_115_df, dic_115["PATH"])
tar_gps_ts_daily_115 = utils.calculate_gps_timeseries_los(tar_gps_ts, insar_115_df, dic_115["PATH"])

# Resample to InSAR ts dates, returns mean with +/- window. 
ref_gps_ts_170 = utils.resample_gps_to_insar_dates(ref_gps_ts_daily_170, insar_170["ts_dates"], window_days=6)
tar_gps_ts_170 = utils.resample_gps_to_insar_dates(tar_gps_ts_daily_170, insar_170["ts_dates"], window_days=6)
ref_gps_ts_115 = utils.resample_gps_to_insar_dates(ref_gps_ts_daily_115, insar_115["ts_dates"], window_days=6)
tar_gps_ts_115 = utils.resample_gps_to_insar_dates(tar_gps_ts_daily_115, insar_115["ts_dates"], window_days=6)

# Merge the reference and target GPS time series on the 'yyyy' column.
gps_ts_170_daily = pd.merge(ref_gps_ts_daily_170[['yyyy', 'LOS_170', 'up']], tar_gps_ts_daily_170[['yyyy', 'LOS_170', 'up']], on='yyyy', suffixes=('_ref', '_tar'))
gps_ts_115_daily = pd.merge(ref_gps_ts_daily_115[['yyyy', 'LOS_115', 'up']], tar_gps_ts_daily_115[['yyyy', 'LOS_115', 'up']], on='yyyy', suffixes=('_ref', '_tar'))

# Merge the reference and target GPS time series on the 'ts_date' column.
gps_ts_170 = pd.merge(ref_gps_ts_170[['ts_date', 'LOS_170', 'up']], tar_gps_ts_170[['ts_date', 'LOS_170', 'up']], on='ts_date', suffixes=('_ref', '_tar'))
gps_ts_115 = pd.merge(ref_gps_ts_115[['ts_date', 'LOS_115', 'up']], tar_gps_ts_115[['ts_date', 'LOS_115', 'up']], on='ts_date', suffixes=('_ref', '_tar'))

# Compute the baseline change as target minus reference for daily
gps_ts_170_daily['gps_LOS_baseline_170'] = gps_ts_170_daily['LOS_170_tar'] - gps_ts_170_daily['LOS_170_ref']
gps_ts_115_daily['gps_LOS_baseline_115'] = gps_ts_115_daily['LOS_115_tar'] - gps_ts_115_daily['LOS_115_ref']
gps_ts_170_daily['gps_Up_baseline_170'] = gps_ts_170_daily['up_tar'] - gps_ts_170_daily['up_ref']
gps_ts_115_daily['gps_Up_baseline_115'] = gps_ts_115_daily['up_tar'] - gps_ts_115_daily['up_ref']

# Compute the baseline change as target minus reference.
gps_ts_170['gps_LOS_baseline_170'] = gps_ts_170['LOS_170_tar'] - gps_ts_170['LOS_170_ref']
gps_ts_115['gps_LOS_baseline_115'] = gps_ts_115['LOS_115_tar'] - gps_ts_115['LOS_115_ref']
gps_ts_170['gps_Up_baseline_170'] = gps_ts_170['up_tar'] - gps_ts_170['up_ref']
gps_ts_115['gps_Up_baseline_115'] = gps_ts_115['up_tar'] - gps_ts_115['up_ref']

#####################
# Get InSAR time series at specific lat, lon, dist. 
#####################
# Get target time series at lat, lon, dist
ref_gps_ts_170["LOS_170_ref_insar"], _ = utils.get_ts_lat_lon_dist(insar_170, ref_lat, ref_lon, dist)
tar_gps_ts_170["LOS_170_tar_insar"], _ = utils.get_ts_lat_lon_dist(insar_170, tar_lat, tar_lon, dist)
ref_gps_ts_115["LOS_115_ref_insar"], _ = utils.get_ts_lat_lon_dist(insar_115, ref_lat, ref_lon, dist)
tar_gps_ts_115["LOS_115_tar_insar"], _ = utils.get_ts_lat_lon_dist(insar_115, tar_lat, tar_lon, dist)


for df, col in [
    (ref_gps_ts_170, "Inc_170"),
    (ref_gps_ts_115, "Inc_115"),
    (tar_gps_ts_170, "Inc_170"),
    (tar_gps_ts_115, "Inc_115"),
]:
    med = df[col].median(skipna=True)
    df[col].fillna(med, inplace=True)
    
# Projecting LOS to Up
ref_gps_ts_170["Up_170_ref_insar"] = utils.proj_los_into_vertical_no_horiz(ref_gps_ts_170["LOS_170_ref_insar"], ref_gps_ts_170["Inc_170"])
ref_gps_ts_115["Up_115_ref_insar"] = utils.proj_los_into_vertical_no_horiz(ref_gps_ts_115["LOS_115_ref_insar"], ref_gps_ts_115["Inc_115"])
tar_gps_ts_170["Up_170_tar_insar"] = utils.proj_los_into_vertical_no_horiz(tar_gps_ts_170["LOS_170_tar_insar"], tar_gps_ts_170["Inc_170"])
tar_gps_ts_115["Up_115_tar_insar"] = utils.proj_los_into_vertical_no_horiz(tar_gps_ts_115["LOS_115_tar_insar"], tar_gps_ts_115["Inc_115"])


# Calculate baseline changes 
gps_ts_170['insar_LOS_baseline_170'] = tar_gps_ts_170['LOS_170_tar_insar'] - ref_gps_ts_170['LOS_170_ref_insar']
gps_ts_115['insar_LOS_baseline_115'] = tar_gps_ts_115['LOS_115_tar_insar'] - ref_gps_ts_115['LOS_115_ref_insar']
gps_ts_170['insar_Up_baseline_170']  = tar_gps_ts_170['Up_170_tar_insar']  - ref_gps_ts_170['Up_170_ref_insar']
gps_ts_115['insar_Up_baseline_115']  = tar_gps_ts_115['Up_115_tar_insar']  - ref_gps_ts_115['Up_115_ref_insar']

#####################
# Get RMSE values for each time series 
#####################

# Example usage for track 170 LOS:
rmse_170_LOS = utils.calculate_rmse_nans(gps_ts_170['gps_LOS_baseline_170'].values, gps_ts_170['insar_LOS_baseline_170'].values)
rmse_170_Up  = utils.calculate_rmse_nans(gps_ts_170['gps_Up_baseline_170'].values,  gps_ts_170['insar_Up_baseline_170'].values)
rmse_115_LOS = utils.calculate_rmse_nans(gps_ts_115['gps_LOS_baseline_115'].values, gps_ts_115['insar_LOS_baseline_115'].values)
rmse_115_Up  = utils.calculate_rmse_nans(gps_ts_115['gps_Up_baseline_115'].values,  gps_ts_115['insar_Up_baseline_115'].values)

rmse_metrics = {'track170': {'LOS': rmse_170_LOS, 'Up': rmse_170_Up},
                'track115': {'LOS': rmse_115_LOS, 'Up': rmse_115_Up}}
print(rmse_metrics)

##################################
# Extract Profiles
##################################
# Load grid files into xyz for profile extraction
xyz_dataframe_A2 = pygmt.grd2xyz(grid=dic_170["vel_grd"], output_type="pandas", region=region)
xyz_dataframe_A2['z'] = xyz_dataframe_A2['z'].replace(0, np.nan)
xyz_dataframe_A2 = xyz_dataframe_A2.dropna()

xyz_dataframe_S1 = pygmt.grd2xyz(grid=dic_115["vel_grd"], output_type="pandas", region=region)
xyz_dataframe_S1['z'] = xyz_dataframe_S1['z'].replace(0, np.nan)
xyz_dataframe_S1 = xyz_dataframe_S1.dropna()

### Define x distance bin spacing to closer near to zero profiles
dist_bins = np.arange(-p_dist, p_dist+1, 0.25)

### Get start and end points
A_start_lon, A_start_lat, A_end_lon, A_end_lat = utils.get_start_end_points(center_lo, center_la, azi_EW, p_dist)
B_start_lon, B_start_lat, B_end_lon, B_end_lat = utils.get_start_end_points(center_lo, center_la, azi_NS, p_dist)

# ### Extract profiles
A2_a_points, A2_a_mean, A2_a_offset = utils.extract_profiles(xyz_dataframe_A2, center_lo, center_la, azi_EW, p_dist, width, dist_bins)
A2_b_points, A2_b_mean, A2_b_offset = utils.extract_profiles(xyz_dataframe_A2, center_lo, center_la, azi_NS, p_dist, width, dist_bins)

### Extract profiles
S1_a_points, S1_a_mean, S1_a_offset = utils.extract_profiles(xyz_dataframe_S1, center_lo, center_la, azi_EW, p_dist, width, dist_bins)
S1_b_points, S1_b_mean, S1_b_offset = utils.extract_profiles(xyz_dataframe_S1, center_lo, center_la, azi_NS, p_dist, width, dist_bins)

##################################
#### Plot results
##################################

fig = pygmt.Figure()
pygmt.config(FONT=9, FONT_TITLE=10, PS_MEDIA="A3", FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",)

with fig.subplot(nrows=2, ncols=1, figsize=("7c", "12c"), autolabel="a)", sharex="l", frame=["Wsrt"], margins=["0.3c", "0.3c"],):
     
    fig.basemap(frame=["Wsrt", "xa0.2", "ya0.1"], region=region, projection=fig_size, panel=True)
    
    pygmt.makecpt(cmap="magma", series=[-0.160, 0.020])
    fig.grdimage(grid=dic_170["vel_grd"], cmap=True, region=region, projection=fig_size )
    fig.coast(shorelines=True, region=region, projection=fig_size)
    
    fig.plot(x=ref_lon, y=ref_lat, style="s.2c", fill="black", pen="1p")
    fig.text(text=ref_station, x=ref_lon , y=ref_lat, justify="None", offset="0c/-0.6c", font="10p,Helvetica-Bold,black" , region=region, projection=fig_size)
    
    fig.plot(x=tar_lon, y=tar_lat, style="t.25c", fill="black", pen="0.5p")
    #fig.text(text="e,f)", x=tar_lon, y=tar_lat, justify="LM", offset="-0.1c/0c", font="9p,Helvetica,black" , region=region, projection=fig_size)
    fig.text(text=tar_station, x=tar_lon , y=tar_lat, justify="None", offset="0c/-0.6c", font="10p,Helvetica-Bold,black" , region=region, projection=fig_size)
    
    fig.plot(x=[A_start_lon, A_end_lon], y=[A_start_lat, A_end_lat], pen="1.2p,orangered", region=region, projection=fig_size, transparency=40)
    fig.text(text="X", x=A_start_lon, y=A_start_lat, justify="RM", offset="-0.2c/0c", font="10p,Helvetica,black" , region=region, projection=fig_size)
    fig.text(text="X'", x=A_end_lon, y=A_end_lat, justify="LM", offset="0.2c/0c", font="10p,Helvetica,black" , region=region, projection=fig_size)
    
    fig.plot(x=[B_start_lon, B_end_lon], y=[B_start_lat, B_end_lat], pen="1.2p,orangered", region=region, projection=fig_size, transparency=40)
    fig.text(text="Y", x=B_start_lon, y=B_start_lat, justify="RM", offset="0.0c/0.2c", font="10p,Helvetica,black", region=region, projection=fig_size )
    fig.text(text="Y'", x=B_end_lon, y=B_end_lat, justify="LM", offset="0.0c/-0.2c", font="10p,Helvetica,black", region=region, projection=fig_size )
    
    fig.text(text="ALOS-2", position="BL", offset="0.2c/0.2c", pen="black", font="10p,Helvetica-Bold,black" , region=region, projection=fig_size,)
    fig.basemap(frame=["wsrt", "xa0.2", "ya0.1"], map_scale="jTR+w5k+o0.4c/0.4c", region=region, projection=fig_size)
    
    df = pd.DataFrame(
        data={
            "x": [-121.835,-121.835 ],
            "y": [38.95,38.95],
            "east_velocity": [-0.173,-0.9848/2.5],
            "north_velocity": [-0.9848, 0.173/2.5],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.5p,black", line=True, spec="e1.2/1/1", vector="0.3c+p2p+e+gblack",region=region, projection=fig_size,)
    
    pygmt.makecpt(cmap="magma", series=[-16, 2])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="JMR+o0.0c/0c+w4c/0.4c", frame=["xa+lLOS (cm/yr)"])
    
    
    ## ******** top map ********* 
    fig.basemap(frame=["WSrt", "xa0.2", "ya0.1"], region=region, projection=fig_size, panel=True)
    pygmt.makecpt(cmap="magma", series=[-0.16, 0.020])
    
    fig.grdimage(grid=dic_115["vel_grd"], region=region, projection=fig_size, cmap=True)
    fig.coast(shorelines=True, region=region, projection=fig_size)

    fig.plot(x=ref_lon, y=ref_lat, style="s.2c", fill="black", pen="1p")
    fig.text(text=ref_station, x=ref_lon , y=ref_lat, justify="None", offset="0c/-0.6c", font="10p,Helvetica-Bold,black" , region=region, projection=fig_size)
    fig.plot(x=tar_lon, y=tar_lat, style="t.25c", fill="black", pen="0.5p")
    #fig.text(text="e,f)", x=tar_lon, y=tar_lat, justify="LM", offset="-0.1c/0c", font="9p,Helvetica,black" , region=region, projection=fig_size)
    fig.text(text=tar_station, x=tar_lon , y=tar_lat, justify="None", offset="0c/-0.6c", font="10p,Helvetica-Bold,black" , region=region, projection=fig_size)
    
    fig.plot(x=[A_start_lon, A_end_lon], y=[A_start_lat, A_end_lat], pen="1.2p,navy", region=region, projection=fig_size, transparency=40)
    fig.text(text="X", x=A_start_lon, y=A_start_lat, justify="RM", offset="-0.2c/0c", font="10p,Helvetica,black" , region=region, projection=fig_size)
    fig.text(text="X'", x=A_end_lon, y=A_end_lat, justify="LM", offset="0.2c/0c", font="10p,Helvetica,black" , region=region, projection=fig_size)
    
    fig.plot(x=[B_start_lon, B_end_lon], y=[B_start_lat, B_end_lat], pen="1.2p,navy", region=region, projection=fig_size, transparency=40)
    fig.text(text="Y", x=B_start_lon, y=B_start_lat, justify="RM", offset="0.0c/0.2c", font="10p,Helvetica,black", region=region, projection=fig_size )
    fig.text(text="Y'", x=B_end_lon, y=B_end_lat, justify="LM", offset="0.0c/-0.2c", font="10p,Helvetica,black", region=region, projection=fig_size )
    
    fig.text(text="Sentinel-1", position="BL", offset="0.2c/0.2c", pen="black", font="10p,Helvetica-Bold,black" , region=region, projection=fig_size,)
    fig.basemap(frame=["WSrt", "xa0.2", "ya0.1"], map_scale="jTR+w5k+o0.4c/0.4c", region=region, projection=fig_size)

    df = pd.DataFrame(
        data={
            "x": [-121.835,-121.835 ],
            "y": [38.95,38.95],
            "east_velocity": [-0.173,-0.9848/2.5],
            "north_velocity": [-0.9848, 0.173/2.5],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.5p,black", line=True, spec="e1.2/1/1", vector="0.3c+p2p+e+gblack",region=region, projection=fig_size,)
    
    pygmt.makecpt(cmap="magma", series=[-16, 2])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="JMR+o0.0c/0c+w4c/0.4c", frame=["xa+lLOS (cm/yr)"])
    
fig.shift_origin(xshift="w+1.5c")
fig.shift_origin(yshift="6.705")

## ******** Time series Panel ********* 

unit = 100 # 100 = cm 
prof_region =[-p_dist-0.5, p_dist+0.5, (np.nanmin(S1_a_mean.z)-0.02)*unit, (np.nanmax(S1_a_mean.z)+0.03)*unit]
region_ts = [t0-0.1, te+0.1, -0.1*unit, 0.1*unit]

with fig.subplot(nrows=2, ncols=1, figsize=("8c", "5.4c"), autolabel="c)", sharex="b", frame=["lSEt" , "xaf+l Profile Distance (km)", "ya+l LOS (cm/yr)"], 
                 margins=["0.05c", "0.05c"],
):
    fig.basemap(region=prof_region, projection="X?", panel=True)
    fig.text(text="X → X'", position="TC", offset="0c/-0.2c", font="9p,Helvetica,black" )
    fig.plot(x=S1_a_points.p, y=S1_a_points.z*unit, style="c0.1c", fill="navy", transparency=95)
    fig.plot(x=S1_a_mean.p,   y=S1_a_mean.z*unit, pen="1.2p,navy")
    fig.plot(x=A2_a_points.p, y=A2_a_points.z*unit, style="c0.1c", fill="orangered", transparency=95)
    fig.plot(x=A2_a_mean.p,   y=A2_a_mean.z*unit, pen="1.2p,orangered") 

    
    fig.basemap(region=prof_region, projection="X?", panel=True)
    fig.text(text="Y → Y'", position="TC", offset="0c/-0.2c", font="9p,Helvetica,black" )
    fig.plot(x=S1_b_points.p, y=S1_b_points.z*unit, style="c0.1c", fill="navy", transparency=95, label="Sentinel-1")
    fig.plot(x=S1_b_mean.p,   y=S1_b_mean.z*unit, pen="1.2p,navy")
    fig.plot(x=A2_b_points.p, y=A2_b_points.z*unit, style="c0.1c", fill="orangered", transparency=95, label="ALOS-2")
    fig.plot(x=A2_b_mean.p,   y=A2_b_mean.z*unit, pen="1.2p,orangered")
    fig.legend(box=False, position="JBL+jBL+o0.1/0.1c")
    
fig.shift_origin(yshift="-6.705c")

with fig.subplot(nrows=2, ncols=1, figsize=("8c", "5.4c"), autolabel="e)", sharex="b", frame=["lSEt" , "xaf+l Profile Distance (km)", "ya+l LOS (mm/yr)"],
                 margins=["0.05c", "0.05c"],
):
    
    # LOS comparison           
    fig.basemap(frame=["lstE", "xaf+lDate", "ya+lDisp. LOS (cm)"], region=region_ts, projection="X?", panel=True)
    fig.plot(x=[2021.9167, 2021.9167,  2022.1667,  2022.1667, 2021.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2022.9167, 2022.9167,  2023.1667,  2023.1667, 2022.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2023.9167, 2023.9167,  2024.1667,  2024.1667, 2023.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)    
    fig.plot(x=gps_ts_115_daily["yyyy"], y=gps_ts_115_daily["gps_LOS_baseline_115"]*unit, style="c0.08c",  fill="darkgrey", transparency=50) 
    fig.plot(x=gps_ts_170_daily["yyyy"], y=gps_ts_170_daily["gps_LOS_baseline_170"]*unit, style="c0.08c",  fill="darkgrey", transparency=50) 
    fig.plot(x=gps_ts_115["ts_date"], y=gps_ts_115["insar_LOS_baseline_115"]*unit, style="c0.1c", fill="navy", label=f"S1 RMSE {np.round(rmse_metrics['track115']['LOS']*unit, 1)} cm")
    fig.plot(x=gps_ts_170["ts_date"], y=gps_ts_170["insar_LOS_baseline_170"]*unit, style="c0.1c", fill="orangered", label=f"A2 RMSE {np.round(rmse_metrics['track170']['LOS']*unit, 1)} cm") 
    fig.text(text=f"{tar_station}-{ref_station} LOS", position="TC", offset="0c/-0.2c", font="9p,Helvetica,black")
    fig.legend(box=False, position="JBL+jBL+o0.1/0.1c")
    
    # Up comparison 
    fig.basemap(frame=["lStE", "xaf+lDate", "ya+lDisp. Up (cm)"], region=region_ts, projection="X?", panel=True)
    fig.plot(x=[2021.9167, 2021.9167,  2022.1667,  2022.1667, 2021.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2022.9167, 2022.9167,  2023.1667,  2023.1667, 2022.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2023.9167, 2023.9167,  2024.1667,  2024.1667, 2023.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)    
    fig.plot(x=gps_ts_115_daily["yyyy"], y=gps_ts_115_daily["gps_Up_baseline_115"]*unit, style="c0.08c",  fill="darkgrey", transparency=50)    
    fig.plot(x=gps_ts_170_daily["yyyy"], y=gps_ts_170_daily["gps_Up_baseline_170"]*unit, style="c0.08c",  fill="darkgrey", transparency=50) 
    fig.plot(x=gps_ts_115["ts_date"], y=gps_ts_115["insar_Up_baseline_115"]*unit, style="c0.1c", fill="navy", label=f"S1 RMSE {np.round(rmse_metrics['track115']['Up']*unit, 1)} cm")
    fig.plot(x=gps_ts_170["ts_date"], y=gps_ts_170["insar_Up_baseline_170"]*unit, style="c0.1c", fill="orangered", label=f"A2 RMSE {np.round(rmse_metrics['track170']['Up']*unit, 1)} cm")  
    fig.text(text=f"{tar_station}-{ref_station} Up", position="TC", offset="0c/-0.2c", font="9p,Helvetica,black")
    fig.legend(box=False, position="JBL+jBL+o0.1/0.1c")
    
fig.savefig(common_paths["fig_dir"]+'Fig_6_Validation_CentrallValley_ts.png', transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+'Fig_6_Validation_CentrallValley_ts.pdf', transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+'Fig_6_Validation_CentrallValley_ts.jpg', transparent=False, crop=True, anti_alias=True, show=False)

fig.show()
