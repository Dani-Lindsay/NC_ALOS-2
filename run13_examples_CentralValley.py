#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:40:43 2025

gmt grdtrack Tehama-Colusa_Canal.txt -G../170_5_28/CentralValley/geo/geo_velocity_msk.grd -S > canal_subsidence.txt
gmt grdtrack Tehama-Colusa_Canal.txt -G../170_5_28/CentralValley/geo/geo_cumdisp_20221226_20220110.grd -S > Tehama-Colusa_canal_cumdisp22.txt

save_gmt.py geo_timeseries_SET_ERA5_ramp_demErr_msk.h5 20220110 
save_gmt.py geo_timeseries_SET_ERA5_ramp_demErr_msk.h5 20221226 
gmt grdmath 20220124_20221226_SET_ERA5_ramp_demErr_msk.grd 20220124_20220110_SET_ERA5_ramp_demErr_msk.grd SUB = 20220110_20221226_cumdisp.grd

@author: daniellelindsay
"""

from NC_ALOS2_filepaths import (common_paths, paths_gps, paths_170_5_28)
import insar_utils as utils
import numpy as np                # Matrix calculations
import pandas as pd               # Pandas for data
import pygmt
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod


dist = common_paths["dist"]

lat_step = common_paths["lat_step"]
lon_step = common_paths["lon_step"]

# Start and end in decimal years
t0 = 2021.5  
te = 2024.4  

unit = 100    
    
dist = 0.005
canal_dist = 13

#38.8994° N, 120.9147° W

ref_station, ref_lon, ref_lat = "P208", -122.3039,  39.1093

Art_la, Art_lo = 39.6242, -122.1954
Art_la, Art_lo = 39.697254, -122.195851
Arb_la, Arb_lo = 39.0174, -122.0577

# Given center point (latitude, longitude)
canal1_lat, canal1_lon = 39.689877, -122.205009
canal2_lat, canal2_lon = 39.003290, -122.077080

# Define the radius within which you want to average the data (in degrees)
radius = dist  # Approximately 1 km if near the equator

def find_indices_within_radius(lons, lats, point_lon, point_lat, radius):
    """Find indices of grid points within a specified radius of a given lon/lat."""
    # Calculate the squared distance from every point in the grid
    dist_sq = (lons - point_lon)**2 + (lats - point_lat)**2
    # Find indices where distance is within the specified radius squared
    within_radius = dist_sq <= radius**2
    return np.where(within_radius)

def extract_averaged_time_series(data, lons, lats, points, radius):
    """Extract and average time series data for a list of points within a specified radius."""
    ts_list = []
    std_list = []
    for lon, lat in points:
        idx_y, idx_x = find_indices_within_radius(lons, lats, lon, lat, radius)
        # Extract the time series data and compute the mean across all points within the radius
        if idx_y.size > 0:  # Check if there are any points within radius
            mean_ts = np.nanmean(data[:, idx_y, idx_x], axis=1)
            std_ts = np.std(data[:, idx_y, idx_x], axis=1)
            mean_ts = mean_ts - mean_ts[0]
        else:
            mean_ts = np.full(data.shape[0], np.nan)  # Return NaN series if no points within radius
        ts_list.append(mean_ts)
        std_list.append(std_ts)
    return ts_list, std_list

import numpy as np
from scipy.stats import linregress

def compute_cumulative_displacement(dates, values, start, stop, return_error=False):
    """
    Compute cumulative displacement between start and stop (decimal years).
    
    Parameters
    ----------
    dates : array-like of float
        Decimal-year timestamps, must be same length as values.
    values : array-like of float
        Displacement time series (e.g., LOS displacement), may contain NaNs.
    start, stop : float
        Decimal-year interval over which to compute cumulative displacement.
    return_error : bool, default False
        If True, also compute uncertainty on cumulative displacement based on
        linear-fit slope stderr: stderr * (stop - start).
    
    Returns
    -------
    cum_disp : float
        Cumulative displacement = value_at_stop - value_at_start.
    cum_err : float, optional
        If return_error=True: estimate of uncertainty = stderr * (stop - start).
    """
    # Convert to numpy arrays
    t = np.asarray(dates, dtype=float)
    d = np.asarray(values, dtype=float)
    # Mask window and non-NaN
    mask = (t >= start) & (t <= stop) & ~np.isnan(d)
    if np.sum(mask) < 2:
        raise ValueError(f"Not enough valid points between {start} and {stop}")
    # Extract subarrays
    t_win = t[mask]
    d_win = d[mask]
    # Ensure sorted by time
    idx_sort = np.argsort(t_win)
    t_win = t_win[idx_sort]
    d_win = d_win[idx_sort]
    # Cumulative displacement: last minus first
    cum_disp = d_win[-1] - d_win[0]
    if not return_error:
        return cum_disp
    # Otherwise compute slope stderr and propagate: slope_err * duration
    res = linregress(t_win, d_win)
    duration = stop - start
    cum_err = res.stderr * duration
    return cum_disp, cum_err


###########################
# Load GNSS
###########################
# cv GPS
columns = ['Lon', 'Lat', 'Ve', 'Vn', 'Vu', 'Std_e', 'Std_n', 'Std_u', 'StaID']
gps_df = pd.read_csv(paths_gps['visr']['gps_enu'] ,delim_whitespace=True, comment='#', names=columns)
Vu_ref, Ve_ref, Vn_ref = gps_df.loc[gps_df['StaID'] == 'P208', ['Vu', 'Ve', 'Vn']].values[0]
gps_df[['Vu', 'Ve', 'Vn']] = gps_df[['Vu', 'Ve', 'Vn']] - [Vu_ref, Ve_ref, Vn_ref]
cv_gps_df = gps_df

###########################
# Load Canal
###########################
# Load canal data
columns = ['Lon', 'Lat', 'Vu']
artois_df = pd.read_csv(common_paths["Artois_file"], delim_whitespace=True, comment='>', names=columns)
arbuckle_df = pd.read_csv(common_paths["Arbuckle_file"], delim_whitespace=True, comment='>', names=columns)

g = Geod(ellps="WGS84")

def process_canal(canal_df, center_lat, center_lon, crop_km=13.0,
                  dist_col_name="dist_from_center"):
    """
    Given a canal DataFrame with ['Lon','Lat','Vu'], compute:
      - cumulative distance along canal as 'dist_km'
      - distance from the point closest to (center_lon, center_lat), in km,
        stored as dist_col_name + "_km"
      - a cropped DataFrame within ±crop_km around that center.
    Returns (full_df, cropped_df).
    """
    df = canal_df.copy().reset_index(drop=True)
    # 1) cumulative distance along canal
    cumdist = [0.0]
    for i in range(1, len(df)):
        lon1, lat1 = df.loc[i-1, ["Lon", "Lat"]]
        lon2, lat2 = df.loc[i,   ["Lon", "Lat"]]
        _, _, dist_m = g.inv(lon1, lat1, lon2, lat2)
        cumdist.append(cumdist[-1] + dist_m / 1000.0)
    df["dist_km"] = cumdist

    # 2) find index of node closest to center
    dists_m = df.apply(
        lambda row: g.inv(row["Lon"], row["Lat"], center_lon, center_lat)[2],
        axis=1
    )
    center_idx = dists_m.idxmin()

    # 3) compute dist_from_center in km
    zero = df.loc[center_idx, "dist_km"]
    col_name = f"{dist_col_name}_km"
    df[col_name] = df["dist_km"] - zero

    # 4) crop to ±crop_km
    mask = df[col_name].abs() <= crop_km
    df_crop = df.loc[mask, ["Lon", "Lat", "Vu", col_name]].reset_index(drop=True)
    return df, df_crop

# Process Artois => canal_cropped1
_, canal_cropped1 = process_canal(
    artois_df, center_lat=canal1_lat, center_lon=canal1_lon,
    crop_km=canal_dist, dist_col_name="dist_from_center1"
)

# Process Arbuckle => canal_cropped2
_, canal_cropped2 = process_canal(
    arbuckle_df, center_lat=canal2_lat, center_lon=canal2_lon,
    crop_km=canal_dist, dist_col_name="dist_from_center2"
)

# Final DataFrames for plotting:
canal_cropped1 = canal_cropped1[["Lon", "Lat", "Vu", "dist_from_center1_km"]]
canal_cropped2 = canal_cropped2[["Lon", "Lat", "Vu", "dist_from_center2_km"]]

# Example: inspect
print("Cropped around Artois center:")
print(canal_cropped1.head())
print("\nCropped around Arbuckle center:")
print(canal_cropped2.head())


###########################
# Load InSAR
###########################

#cv_center_la, cv_center_lo,  =  41.578394, -121.591573
ref_station, ref_lat, ref_lon = "P208", 39.109, -122.304

# Coordinates of points
points_CV = [
    (-122.1996, 39.6806), # Central Valley North
    (-122.0723, 39.0036), # Central Valley South
    ]

radius_CV = 0.004

cv_min_lon=-122.4
cv_max_lon=-121.8
cv_min_lat=38.8
cv_max_lat=39.8


cv_region = "%s/%s/%s/%s" % (cv_min_lon, cv_max_lon, cv_min_lat, cv_max_lat)

dic_CV = {"ORBIT_DIRECTION" : "descending", 
           "PATH" : "170", 
           "Platform" : "ALOS-2", 
           "geo_file" : paths_170_5_28["CentralValley"]["geo_geometryRadar"], 
           "vel_file" : paths_170_5_28["CentralValley"]["geo_velocity_msk"], 
           "ts_file" : paths_170_5_28["CentralValley"]["geo_timeseries_msk"], 
           "vel_grd" : paths_170_5_28["CentralValley"]["geo_velocity_msk_grd"], 
           }

insar_CV = utils.load_insar_vel_ts_as_dictionary(dic_CV)
time_series_CV, time_series_std_CV = extract_averaged_time_series(insar_CV["ts"], insar_CV["lons"], insar_CV["lats"], points_CV, radius_CV)
time_series_CV = [ts - ts[1] for ts in time_series_CV]

# Project velocities to vertical (using the incidence angles)
#df_asc_des['asc_v'] = df_asc_des['asc_vel'] * np.cos(np.deg2rad(df_asc_des['asc_inc']))
#df_asc_des['des_v'] = df_asc_des['des_vel'] * np.cos(np.deg2rad(df_asc_des['des_inc']))

vel_t1 = utils.date_to_decimal_year('20220101')
vel_t2 = utils.date_to_decimal_year('20221231')
vel_t3 = utils.date_to_decimal_year('20230101')
vel_t4 = utils.date_to_decimal_year('20231231')

vel_Art_22, err_Art_22 = utils.compute_velocity(insar_CV["ts_dates"], time_series_CV[0]*unit, start=vel_t1, stop=vel_t2)
vel_Art_23, err_Art_23 = utils.compute_velocity(insar_CV["ts_dates"], time_series_CV[0]*unit, start=vel_t3, stop=vel_t4)

vel_Arb_22, err_Arb_22 = utils.compute_velocity(insar_CV["ts_dates"], time_series_CV[1]*unit, start=vel_t1, stop=vel_t2)
vel_Arb_23, err_Arb_23 = utils.compute_velocity(insar_CV["ts_dates"], time_series_CV[1]*unit, start=vel_t3, stop=vel_t4)

cum_Art_22, cum_err_Art_22 = utils.compute_velocity(insar_CV["ts_dates"], time_series_CV[0]*unit, start=vel_t1, stop=vel_t2)
cum_Art_23, cum_err_Art_23 = utils.compute_velocity(insar_CV["ts_dates"], time_series_CV[0]*unit, start=vel_t3, stop=vel_t4)

cum_Arb_22, cum_err_Arb_22 = utils.compute_velocity(insar_CV["ts_dates"], time_series_CV[1]*unit, start=vel_t1, stop=vel_t2)
cum_Arb_23, cum_err_Arb_23 = utils.compute_velocity(insar_CV["ts_dates"], time_series_CV[1]*unit, start=vel_t3, stop=vel_t4)


# Download DEM
grid = pygmt.datasets.load_earth_relief(region=cv_region, resolution="03s")
dgrid = pygmt.grdgradient(grid=grid, radiance=[315, 30], region=cv_region)


###########################
# Compute subsidence 
###########################

# 1) Compute the mean “stable” velocity in the specified distance ranges:
mean_X = canal_cropped1.loc[
    (canal_cropped1['dist_from_center1_km'] >= -10) & 
    (canal_cropped1['dist_from_center1_km'] <= -5), 
    'Vu'
].mean()

mean_Y = canal_cropped2.loc[
    (canal_cropped2['dist_from_center2_km'] >= 10) & 
    (canal_cropped2['dist_from_center2_km'] <= 13), 
    'Vu'
].mean()

# 2) Find the minimum (most negative) Vu value for each profile:
min_X = np.nanmin(canal_cropped1['Vu'])
min_Y = np.nanmin(canal_cropped2['Vu'])

# 3) Calculate subsidence relative to the “stable” mean:
subsidence_X = min_X - mean_X
subsidence_Y = min_Y - mean_Y

print("Subsidence X:", subsidence_X)
print("Subsidence Y:", subsidence_Y)


###########################
# Plot 
###########################

style="c.03c"
res_style="c.1c"
size = "M5.01c"

vmin = 20

sub_map_size = "M4.48c"
sub_proj = "X6/2c"
#NC_grid = pygmt.datasets.load_earth_relief(region=NC_fig_region, resolution="15s")
#NC_dgrid = pygmt.grdgradient(grid=NC_grid, radiance=[315, 30], region=NC_fig_region)

fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=10, FONT_TITLE=10, FONT_SUBTITLE = 10, MAP_TITLE_OFFSET= "-7p")

with fig.subplot(nrows=1, ncols=2, figsize=("9.2c", "9.6c"), autolabel="a)",sharex="b", sharey="l",
                  frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    
    # ## ******** cv DEM Map ********* 
    fig.basemap(region=cv_region, projection=sub_map_size, frame=["WStr", "xa", "ya"], panel=True,)
    fig.grdimage(grid=grid, projection=sub_map_size, frame=["lbrt", "xa", "ya"], cmap="grey", shading=dgrid, region=cv_region)
    
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.5p,black", transparency=50, region=cv_region, projection= sub_map_size)
    
    fig.plot(data=common_paths["roads_primary"], pen="0.3p,black", region=cv_region, transparency=80, projection= sub_map_size) #, label="Canals")  
    fig.plot(data=common_paths["roads_tertiary"], pen="0.3p,black", region=cv_region, transparency=80, projection= sub_map_size) #, label="Canals")     
    fig.plot(data=common_paths["aquifer_file"], pen="0.5p,navy", region=cv_region, projection= sub_map_size) #, label="Aquifers")
    fig.plot(data=common_paths["canals_file"], pen="0.8p,dodgerblue2", region=cv_region, projection= sub_map_size) #, label="Canals")  
    fig.plot(data=common_paths["Tehama-Colusa_file"], pen="1p,dodgerblue2", region=cv_region, projection= sub_map_size) #, label="Tehama-Colusa")
    fig.plot(data=common_paths["roads_major"], pen="1p,forestgreen", region=cv_region, projection= sub_map_size) #, label="Canals") 
    fig.plot(x=Art_lo, y=Art_la, style="t.1c", fill="black", pen="1p", projection=sub_map_size)
    fig.plot(x=Arb_lo, y=Arb_la, style="t.1c", fill="black", pen="1p", projection=sub_map_size)
    fig.text(text="Greenwood", x=Art_lo, y=Art_la, justify="LB", offset="0.1c/0.1c", font="10p,Helvetica,black", projection=sub_map_size, fill="white", transparency=50)
    fig.text(text="Arbuckle", x=Arb_lo, y=Arb_la, justify="LB", offset="0.1c/0.1c", font="10p,Helvetica,black", projection=sub_map_size, fill="white", transparency=50)
    fig.text(text="Greenwood", x=Art_lo, y=Art_la, justify="LB", offset="0.1c/0.1c", font="10p,Helvetica,black", projection=sub_map_size)
    fig.text(text="Arbuckle", x=Arb_lo, y=Arb_la, justify="LB", offset="0.1c/0.1c", font="10p,Helvetica,black", projection=sub_map_size)
    fig.plot(x=points_CV[0][0], y=points_CV[0][1], style="+.2c", pen="0.5p,black", projection=sub_map_size)
    fig.plot(x=points_CV[1][0], y=points_CV[1][1], style="+.2c", pen="0.5p,black", projection=sub_map_size)
    fig.text(text="c)", x=points_CV[0][0], y=points_CV[0][1], justify="BR", offset="-0.2c/-0.1c", font="10p,Helvetica,black", projection=sub_map_size, fill="white", transparency=50)
    fig.text(text="d)", x=points_CV[1][0], y=points_CV[1][1], justify="BR", offset="-0.2c/-0.1c", font="10p,Helvetica,black", projection=sub_map_size, fill="white", transparency=50)
    fig.text(text="c)", x=points_CV[0][0], y=points_CV[0][1], justify="BR", offset="-0.2c/-0.1c", font="10p,Helvetica,black", projection=sub_map_size)
    fig.text(text="d)", x=points_CV[1][0], y=points_CV[1][1], justify="BR", offset="-0.2c/-0.1c", font="10p,Helvetica,black", projection=sub_map_size)
    
    fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", projection=sub_map_size)
    fig.text(text=ref_station, y=ref_lat, x=ref_lon, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size, fill="white", transparency=50)
    fig.text(text=ref_station, y=ref_lat, x=ref_lon, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size)
    #fig.legend(position="JBL+jBL+o0.2c", box="+gwhite+p1p")
    
    
    # ## ******** cv Map ********* 
    fig.basemap(region=cv_region, projection=sub_map_size, frame=["wStr", "xa", "ya"], panel=True,)
    fig.grdimage(grid=grid, projection=sub_map_size, frame=["lbrt", "xa", "ya"], cmap="grey", shading=dgrid, region=cv_region)
    pygmt.makecpt(cmap="magma", series=[-0.25, 0.05])
    fig.grdimage(grid=paths_170_5_28["CentralValley"]['geo_cumdisp_22'], cmap=True, nan_transparent=True, region=cv_region, projection= sub_map_size)
    
    fig.plot(x=canal_cropped1['Lon'].iloc[0],  y=canal_cropped1['Lat'].iloc[0], style="s.1c", fill="black", pen="1p", projection=sub_map_size)
    fig.plot(x=canal_cropped1['Lon'].iloc[-1], y=canal_cropped1['Lat'].iloc[-1], style="s.1c", fill="black", pen="1p", projection=sub_map_size)
    fig.plot(x=canal_cropped2['Lon'].iloc[0],  y=canal_cropped2['Lat'].iloc[0], style="s.1c", fill="black", pen="1p", projection=sub_map_size)
    fig.plot(x=canal_cropped2['Lon'].iloc[-1], y=canal_cropped2['Lat'].iloc[-1], style="s.1c", fill="black", pen="1p", projection=sub_map_size)

    fig.text(text="X",  x=canal_cropped1['Lon'].iloc[0],  y=canal_cropped1['Lat'].iloc[0], justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=cv_region, projection= sub_map_size, fill="white", transparency=50)
    fig.text(text="X'", x=canal_cropped1['Lon'].iloc[-1], y=canal_cropped1['Lat'].iloc[-1],justify="TL", offset="0.1c/-0.1c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size, fill="white", transparency=50)
    fig.text(text="Y",  x=canal_cropped2['Lon'].iloc[0],  y=canal_cropped2['Lat'].iloc[0], justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size, fill="white", transparency=50)
    fig.text(text="Y'", x=canal_cropped2['Lon'].iloc[-1], y=canal_cropped2['Lat'].iloc[-1],justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size, fill="white", transparency=50)
    
    fig.text(text="X",  x=canal_cropped1['Lon'].iloc[0],  y=canal_cropped1['Lat'].iloc[0], justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=cv_region, projection= sub_map_size)
    fig.text(text="X'", x=canal_cropped1['Lon'].iloc[-1], y=canal_cropped1['Lat'].iloc[-1],justify="TL", offset="0.1c/-0.1c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size)
    fig.text(text="Y",  x=canal_cropped2['Lon'].iloc[0],  y=canal_cropped2['Lat'].iloc[0], justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size)
    fig.text(text="Y'", x=canal_cropped2['Lon'].iloc[-1], y=canal_cropped2['Lat'].iloc[-1],justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size)
    
    fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", projection=sub_map_size)
    
    fig.plot(data=common_paths["Tehama-Colusa_file"], pen="1p,dodgerblue2", region=cv_region, projection= sub_map_size) #, label="Tehama-Colusa")
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.5p,black", transparency=50, region=cv_region, projection= sub_map_size)
    
    fig.text(text="Tehama-Colusa Canal", position="CM", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size, fill="white", transparency=50)
    fig.text(text="Tehama-Colusa Canal", position="CM", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size)
    
    fig.text(text="01-01-22 to 12-31-22", position="BC", offset="0.0c/0.2c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size, fill="white", transparency=50)
    fig.text(text="01-01-22 to 12-31-22", position="BC", offset="0.0c/0.2c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size)
    
    with pygmt.config(
            FONT_ANNOT_PRIMARY="18p,black", 
            FONT_ANNOT_SECONDARY="18p,black",
            FONT_LABEL="18p,black",
            ):
        pygmt.makecpt(cmap="magma", series=[-25, 5])
        fig.colorbar(position="jBL+o0.4c/0.5c+w2c/0.3c", frame=["xaf", "y+lLOS cm"],)
        
    fig.basemap(region=cv_region, projection=sub_map_size, frame=["lbtr", "xa", "ya"], map_scale="jTR+w10k+o0.2c/0.2c")

   
fig.shift_origin(xshift="w+0.5c", yshift="5.08c")

with fig.subplot(nrows=2, ncols=1, figsize=("6c", "4.5c"), autolabel="c)",sharex="b", sharey="l",
                   frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    

    sym_size = "c0.08c"
    text_y1 = -30
    text_y2 = -17
    region_ts = [t0-0.1, te+0.1, np.nanmin(time_series_CV)*unit-0.05*unit-5, 0.02*unit]
    fig.basemap(    region=region_ts, projection=sub_proj, frame=["lstE", "xaf", "ya+lLOS (cm)"], panel=True)
    fig.plot(region=region_ts, projection=sub_proj, x=[2021.9167, 2021.9167,  2022.1667,  2022.1667, 2021.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(region=region_ts, projection=sub_proj,x=[2022.9167, 2022.9167,  2023.1667,  2023.1667, 2022.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(region=region_ts, projection=sub_proj,x=[2023.9167, 2023.9167,  2024.1667,  2024.1667, 2023.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)        
    fig.plot(       region=region_ts, projection=sub_proj, x=insar_CV["ts_dates"], y=time_series_CV[0]*unit, style=sym_size, fill="dodgerblue4", )
    fig.text(       region=region_ts, projection=sub_proj, text="Greenwood", position="TC",offset ="0.0/-0.1c", font="8p,black") 
    
    fig.plot(x=[vel_t1, vel_t2], y=[text_y1, text_y1],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.plot(x=[vel_t3, vel_t4], y=[text_y2, text_y2],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.text(text=f"{cum_Art_22:.1f} ± {cum_err_Art_22:.1f} cm", x=(vel_t1+vel_t2)/2, y=text_y1, font="8p,Helvetica,black", offset="0c/-0.2c")
    fig.text(text=f"{cum_Art_23:.1f} ± {cum_err_Art_23:.1f} cm", x=(vel_t3+vel_t4)/2, y=text_y2, font="8p,Helvetica,black", offset="0c/0.2c")
    
    text_y1 = -35
    text_y2 = -20
    region_ts = [t0-0.1, te+0.1, np.nanmin(time_series_CV)*unit-0.05*unit-10, 0.02*unit]
    fig.basemap(    region=region_ts, projection=sub_proj, frame=["lStE", "xaf", "ya+lLOS (cm)"], panel=True)
    fig.plot(region=region_ts, projection=sub_proj,x=[2021.9167, 2021.9167,  2022.1667,  2022.1667, 2021.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(region=region_ts, projection=sub_proj,x=[2022.9167, 2022.9167,  2023.1667,  2023.1667, 2022.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(region=region_ts, projection=sub_proj,x=[2023.9167, 2023.9167,  2024.1667,  2024.1667, 2023.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)    
    
    fig.plot(x=[vel_t1, vel_t2], y=[text_y1, text_y1],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.plot(x=[vel_t3, vel_t4], y=[text_y2, text_y2],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.text(text=f"{cum_Arb_22:.1f} ± {cum_err_Arb_22:.1f} cm", x=(vel_t1+vel_t2)/2, y=text_y1, font="8p,Helvetica,black", offset="0c/-0.2c")
    fig.text(text=f"{cum_Arb_23:.1f} ± {cum_err_Arb_23:.1f} cm", x=(vel_t3+vel_t4)/2, y=text_y2, font="8p,Helvetica,black", offset="0c/0.2c")
    
    for date, value, sigma in zip(insar_CV["ts_dates"], time_series_CV[1], time_series_std_CV[1]):
        if np.isnan(value) or np.isnan(sigma):
            continue  # skip NaNs
        y0 = value - sigma
        y1 = value + sigma
        fig.plot(
            x=[date, date],
            y=[y0*unit,   y1*unit],
            pen="0.8p,gray",
            region=region_ts,
            projection=sub_proj
        )


    fig.plot(region=region_ts, projection=sub_proj, x=insar_CV["ts_dates"], y=time_series_CV[1]*unit, style=sym_size, fill="dodgerblue4", )
    fig.text(region=region_ts, projection=sub_proj, text="Arbuckle", position="TC",offset ="0.0/-0.1c", font="8p,black") 
    
fig.shift_origin(yshift="-5.08c")  

with fig.subplot(nrows=2, ncols=1, figsize=("6c", "4.5c"), autolabel="e)", sharex="b", sharey="l",
                 frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):

    region_profile = [-canal_dist-0.5, canal_dist+0.5, np.nanmin(canal_cropped1['Vu'])*unit-0.05*unit, 0.12*unit]
    region_profile = [-canal_dist-0.5, canal_dist+0.5, -23, 12]

    fig.basemap(region=region_profile, projection=sub_proj, panel=True, frame=[
                "lsEt", "xaf+l Canal Distance (km)", "yaf+l LOS (cm)"])
    pygmt.makecpt(cmap="magma", series=[-0.25, 0.02])
    fig.plot(x=canal_cropped1['dist_from_center1_km'], y=(canal_cropped1['Vu']-mean_X)*unit, style="c0.1c", fill=canal_cropped1['Vu'],
             cmap=True, transparency=50, projection=sub_proj, region=region_profile)
    fig.plot(x=canal_cropped1['dist_from_center1_km'], y=(canal_cropped1['Vu']-mean_X)*unit, pen="0.8p,black", 
             projection=sub_proj, region=region_profile)
    fig.text(text="X → X' Greenwood", position="TC", offset="0c/-0.2c", font="8p,Helvetica,black",
             region=region_profile, projection=sub_proj)
    
    fig.plot(x=[0, 5], y=[0, 0],  pen="0.8p,black,--", region=region_profile, projection=sub_proj,)
    fig.plot(x=[0, 5], y=[(min_X-mean_X)*unit,(min_X-mean_X)*unit],  pen="0.8p,black,--" , region=region_profile, projection=sub_proj,)
    fig.plot(x=4.5, y=subsidence_X*unit, style="v0.3c+b+e+h0.15", direction=([90], [np.abs(subsidence_X*unit)/18]), 
             pen="0.8p,black", fill="black", region=region_profile, projection=sub_proj,)
    fig.text(text="%s cm" % np.round(subsidence_X*unit,1), x=5, y=subsidence_X*unit+2, justify="BL", 
             font="8p,Helvetica,black",region=region_profile, projection=sub_proj,)

    region_profile = [-canal_dist-0.5, canal_dist+0.5, np.nanmin(canal_cropped2['Vu'])*unit-0.03*unit, 0.12*unit]
    region_profile = [-canal_dist-0.5, canal_dist+0.5, -23, 12]
    fig.basemap(region=region_profile, projection=sub_proj, panel=True, frame=[
        "lSEt", "xaf+l Canal Distance (km)", "yaf+l LOS (cm)"])
    pygmt.makecpt(cmap="magma", series=[-0.25, 0.02])
    fig.plot(x=canal_cropped2['dist_from_center2_km'], y=(canal_cropped2['Vu']-mean_Y)*unit, style="c0.1c", fill=canal_cropped2['Vu'],
             cmap=True, transparency=50, projection=sub_proj, region=region_profile)
    fig.plot(x=canal_cropped2['dist_from_center2_km'], y=(canal_cropped2['Vu']-mean_Y)*unit, pen="0.8p,black", projection=sub_proj,
             region=region_profile)
    fig.text(text="Y → Y' Arbuckle", position="TC", offset="0c/-0.2c", font="8p,Helvetica,black",
             region=region_profile, projection=sub_proj)
    
    fig.plot(x=[0, 5], y=[0, 0],  pen="0.8p,black,--", region=region_profile, projection=sub_proj,)
    fig.plot(x=[0, 5], y=[(min_Y-mean_Y)*unit, (min_Y-mean_Y)*unit],  pen="0.8p,black,--" , region=region_profile, projection=sub_proj,)
    fig.plot(x=4.5, y=(min_Y-mean_Y)*unit, style="v0.3c+b+e+h0.15", direction=([90], [np.abs((min_Y-mean_Y)*unit)/18]), 
             pen="0.8p,black", fill="black", region=region_profile, projection=sub_proj,)
    fig.text(text="%s cm" % np.round(subsidence_Y*unit,1), x=5, y=subsidence_Y*unit+2, justify="BL",
             font="8p,Helvetica,black",region=region_profile, projection=sub_proj,)
    
fig.savefig(common_paths["fig_dir"]+"Fig_11_example_CentralValley.png", transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+"Fig_11_example_CentralValley.pdf", transparent=False, crop=True, anti_alias=True, show=False)
fig.show()  