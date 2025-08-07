#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 00:06:31 2025

@author: daniellelindsay
"""
from NC_ALOS2_filepaths import (common_paths, paths_170_5_28,)
import insar_utils as utils
import numpy as np                
import pygmt
import pandas as pd

dist = common_paths["dist"]

#lat_step = common_paths["lat_step"]
#lon_step = common_paths["lon_step"]

# Start and end in decimal years
t0 = 2021.5  
te = 2024.3  

landslide_poly_file = common_paths["wcSlides"]

eq1 = 2021.9685
eq2 = 2022.9685

dist = 0.004

unit = 100

# Define the radius within which you want to average the data (in degrees)
radius = dist  # Approximately 1 km if near the equator

#fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", projection=sub_map_size)
#fig.text(text=ref_station, y=ref_lat, x=ref_lon, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size, fill="white", transparency=50)
#fig.text(text=ref_station, y=ref_lat, x=ref_lon, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black",region=cv_region, projection= sub_map_size)

eel_lat, eel_lon = 40.08125, -123.4565
gra_lat, gra_lon = 40.72454, -123.81898

########################################
### InSAR Timeseries
########################################

dic_Ee = {"ORBIT_DIRECTION" : "descending", 
           "PATH" : "170", 
           "Platform" : "ALOS-2", 
           "geo_file" : paths_170_5_28["EelRiver"]["geo_geometryRadar"], 
           "vel_file" : paths_170_5_28["EelRiver"]["geo_velocity_msk"], 
           "ts_file" :  paths_170_5_28["EelRiver"]["geo_timeseries_msk"], 
           "vel_grd" :  paths_170_5_28["EelRiver"]["geo_velocity_msk_grd"], 
           }


dic_Gr = {"ORBIT_DIRECTION" : "descending", 
           "PATH" : "170", 
           "Platform" : "ALOS-2", 
           "geo_file" : paths_170_5_28["Graham"]["geo_geometryRadar"], 
           "vel_file" : paths_170_5_28["Graham"]["geo_velocity_msk"], 
           "ts_file" :  paths_170_5_28["Graham"]["geo_timeseries_msk"], 
           "vel_grd" :  paths_170_5_28["Graham"]["geo_velocity_msk_grd"], 
           }


insar_Ee = utils.load_insar_vel_ts_as_dictionary(dic_Ee)
insar_Gr = utils.load_insar_vel_ts_as_dictionary(dic_Gr)

points_Ee = [(-123.47859, 40.06566), (-123.46672, 40.06880)] # -123.48453, 40.06314, -123.47859, 40.06566, -123.47774, 40.06503
points_Ee = [(-123.485614,  40.063447), (-123.466443, 40.070359)]

points_Gr = [(-123.839620, 40.701837), (-123.861337, 40.694509)] # (-123.852107, 40.723637),
 
#points_Gr = [( 40.69284, -123.82379), (-123.861337, 40.694509)]
# Assuming 'des2_lons' and 'des2_lats' are 2D arrays of longitude and latitude values
# and 'des2_ts' is the 3D time series data array

radius_LS = 0.0015

time_series_Ee = utils.extract_averaged_time_series(insar_Ee["ts"], insar_Ee["lons"], insar_Ee["lats"], points_Ee, radius_LS)
time_series_Gr = utils.extract_averaged_time_series(insar_Gr["ts"], insar_Gr["lons"], insar_Gr["lats"], points_Gr, radius_LS)
time_series_Ee = [ts - ts[1] for ts in time_series_Ee]
time_series_Gr = [ts - ts[1] for ts in time_series_Gr]
 
geometry_Ee  = utils.extract_geometry_at_points(dic_Ee["geo_file"], points_Ee, radius_LS)
geometry_Gr  = utils.extract_geometry_at_points(dic_Gr["geo_file"], points_Gr, radius_LS)

# Calculate velocity per year 

vel_t1 = 2022.1667
vel_t2 = 2022.9167
vel_t3 = 2023.1667
vel_t4 = 2023.9167

vel_Gr_c_22, err_Gr_c_22 = utils.compute_velocity(insar_Gr["ts_dates"], time_series_Gr[0]*unit, start=vel_t1, stop=vel_t2)
vel_Gr_d_22, err_Gr_d_22 = utils.compute_velocity(insar_Gr["ts_dates"], time_series_Gr[1]*unit, start=vel_t1, stop=vel_t2)
vel_Gr_c_23, err_Gr_c_23 = utils.compute_velocity(insar_Gr["ts_dates"], time_series_Gr[0]*unit, start=vel_t3, stop=vel_t4)
vel_Gr_d_23, err_Gr_d_23 = utils.compute_velocity(insar_Gr["ts_dates"], time_series_Gr[1]*unit, start=vel_t3, stop=vel_t4)

vel_Ee_g_22, err_Ee_g_22 = utils.compute_velocity(insar_Ee["ts_dates"], time_series_Ee[0]*unit, start=vel_t1, stop=vel_t2)
vel_Ee_h_22, err_Ee_h_22 = utils.compute_velocity(insar_Ee["ts_dates"], time_series_Ee[1]*unit, start=vel_t1, stop=vel_t2)
vel_Ee_g_23, err_Ee_g_23 = utils.compute_velocity(insar_Ee["ts_dates"], time_series_Ee[0]*unit, start=vel_t3, stop=vel_t4)
vel_Ee_h_23, err_Ee_h_23 = utils.compute_velocity(insar_Ee["ts_dates"], time_series_Ee[1]*unit, start=vel_t3, stop=vel_t4)


DS_vel_Gr_c_22 = utils.project_los2vector(vel_Gr_c_22, geometry_Gr["incidenceAngle"][0], geometry_Gr["azimuthAngle"][0], 90 * geometry_Gr["slope"][0], geometry_Gr["aspect"][0])
DS_vel_Gr_d_22 = utils.project_los2vector(vel_Gr_d_22, geometry_Gr["incidenceAngle"][1], geometry_Gr["azimuthAngle"][1], 90 * geometry_Gr["slope"][1], geometry_Gr["aspect"][1])
DS_vel_Gr_c_23 = utils.project_los2vector(vel_Gr_c_23, geometry_Gr["incidenceAngle"][0], geometry_Gr["azimuthAngle"][0], 90 * geometry_Gr["slope"][0], geometry_Gr["aspect"][0])
DS_vel_Gr_d_23 = utils.project_los2vector(vel_Gr_d_23, geometry_Gr["incidenceAngle"][1], geometry_Gr["azimuthAngle"][1], 90 * geometry_Gr["slope"][1], geometry_Gr["aspect"][1])

    
DS_vel_Ee_g_22 = utils.project_los2vector(vel_Ee_g_22, geometry_Ee["incidenceAngle"][0], geometry_Ee["azimuthAngle"][0], 90*geometry_Ee["slope"][0], geometry_Ee["aspect"][0])
DS_vel_Ee_h_22 = utils.project_los2vector(vel_Ee_g_22, geometry_Ee["incidenceAngle"][1], geometry_Ee["azimuthAngle"][1], 90*geometry_Ee["slope"][1], geometry_Ee["aspect"][1])
DS_vel_Ee_g_23 = utils.project_los2vector(vel_Ee_g_23, geometry_Ee["incidenceAngle"][0], geometry_Ee["azimuthAngle"][0], 90*geometry_Ee["slope"][0], geometry_Ee["aspect"][0])
DS_vel_Ee_h_23 = utils.project_los2vector(vel_Ee_g_23, geometry_Ee["incidenceAngle"][1], geometry_Ee["azimuthAngle"][1], 90*geometry_Ee["slope"][1], geometry_Ee["aspect"][1])


##################################
#### Plot results
##################################



fig_size = "M6.1c"




fig = pygmt.Figure()
pygmt.config(FONT=10, FONT_TITLE=10, PS_MEDIA="A3", FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain",)

fig.shift_origin(yshift="8.5c")
########3 Map Graham
with fig.subplot(nrows=1, ncols=2, figsize=("12.0c", "6.4c"), autolabel="a)", sharex="l", frame=["WSrt"], margins=["0.3c", "0.3c"],):
     
    # Define region of interest 
    min_lon=-123.95-0.02
    max_lon=-123.7-0.02
    min_lat=40.6
    max_lat=40.8
    region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
    
    # Download DEM
    grid = pygmt.datasets.load_earth_relief(region=region, resolution="03s")
    dgrid = pygmt.grdgradient(grid=grid, radiance=[315, 30], region=region)
    
    fig.basemap(frame=["WSrt", "xa0.1", "ya0.1"], region=region, projection=fig_size, panel=True)
    fig.grdimage(grid=grid, projection=fig_size, frame=["lbrt", "xa", "ya"], cmap="grey", shading=dgrid, region=region)
    fig.plot(data=landslide_poly_file, pen="1.0p,dodgerblue4", projection=fig_size)
        
    labels = ["c)", "d)"]
    
    for (lon, lat), lbl in zip(points_Gr, labels):
        # Plot marker at (lon, lat)
        fig.plot(x=lon, y=lat, style="c.2c", pen="0.8p", projection=fig_size)
    
        # Label the marker with offset 0.5 cm to the right
        fig.text(text=lbl, x=lon, y=lat, justify="RM", offset="0.5c/0c", font="10p,Helvetica,black", 
                 region=region, projection=fig_size)
    
    fig.plot(y=gra_lat, x=gra_lon, style="s.15c", fill="black", pen="0.8p,black", region=region, projection=fig_size)
    fig.basemap(frame=["WSrt", "xa0.1", "ya0.1"], map_scale="jBR+w5k+o0.3/0.5c", projection=fig_size)
    
    
    # Define region of interest 
    min_lon=-123.95-0.02
    max_lon=-123.7-0.02
    min_lat=40.6
    max_lat=40.8
    region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
    fig.basemap(frame=["wSrt", "xa0.1", "ya0.1"], region=region, projection=fig_size, panel=True)
    pygmt.makecpt(cmap="vik", series=[-0.1+0.035, 0.1+0.035])
    fig.grdimage(grid=paths_170_5_28['Graham']['geo_velocity_msk_grd'], cmap=True, region=region, projection=fig_size)
    fig.plot(data=landslide_poly_file, pen="1.0p,white", projection=fig_size)
    
    fig.text(region=region, projection=fig_size, text="Graham Complex", position="TC",offset ="0.0/-0.2c") 
    
    labels = ["c)", "d)"]
    
    for (lon, lat), lbl in zip(points_Gr, labels):
        # Plot marker at (lon, lat)
        fig.plot(x=lon, y=lat, style="c.2c", pen="0.5p", projection=fig_size)
    
        # Label the marker with offset 0.5 cm to the right
        fig.text(text=lbl, x=lon, y=lat, justify="RM", offset="0.5c/0c", font="10p,Helvetica,black", 
                 region=region, projection=fig_size)
    
    df = pd.DataFrame(
        data={
            "x": [-123.825 + 0.125*0.74,-123.825 + 0.125*0.74],
            "y": [40.79, 40.79],
            "east_velocity": [-0.173*0.8,(-0.9848/2.5)*0.8],
            "north_velocity": [-0.9848*0.8, (0.173/2.5)*0.8],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.5p,black", line=True, spec="e1.2/1/1", vector="0.3c+p2p+e+gblack",region=region, projection=fig_size)

    
    with pygmt.config(
             FONT_ANNOT_PRIMARY="18p,black", 
             FONT_ANNOT_SECONDARY="18p,black",
             FONT_LABEL="18p,black",
             ):
        pygmt.makecpt(cmap="vik", series=[-10, 10])
        fig.colorbar(position="jBL+o0.5c/0.5c+w3c/0.4c", frame=["xa", "y+lcm/yr"], projection=fig_size)
            
    fig.plot(y=gra_lat, x=gra_lon, style="s.15c", fill="black", pen="0.8p,black", region=region, projection=fig_size)
    fig.basemap(frame=["wSrt", "xa0.1", "ya0.1"], map_scale="jBR+w5k+o0.3/0.5c", projection=fig_size)
    
fig.shift_origin(xshift="w+1c")
fig.shift_origin(yshift="0c")

## ******** Profiles Panel ********* 
region_ts = [t0-0.1, te+0.1, -0.5*100, 0.5*100]

########3 Time series Graham
with fig.subplot(nrows=2, ncols=1, figsize=("8.5c", "6.4c"), autolabel="c)", sharex="b", frame=["lSEt" , "xaf", "ya+l LOS (mm/yr)"],
                 margins=["0.05c", "0.05c"],
):
 
    y_text = -24
    region_ts = [t0-0.1, te+0.1, -0.27*100, 0.05*100]
    fig.basemap(frame=["lstE", "xaf+lDate", "ya+lLOS (cm)"], region=region_ts, projection="X?", panel=True)
    fig.plot(x=[2021.9167, 2021.9167,  2022.1667,  2022.1667, 2021.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2022.9167, 2022.9167,  2023.1667,  2023.1667, 2022.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2023.9167, 2023.9167,  2024.1667,  2024.1667, 2023.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)    
    fig.plot(x=[eq1, eq1], y=[-0.25*100, 0.05*100],  pen="1p,black,--",) 
    fig.plot(x=[eq2, eq2], y=[-0.25*100, 0.05*100],  pen="1p,black,--",)
    
    fig.plot(x=[vel_t1, vel_t2], y=[y_text, y_text],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.plot(x=[vel_t3, vel_t4], y=[y_text, y_text],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.text(text=f"{vel_Gr_c_22:.1f} ± {err_Gr_c_22:.1f} cm/yr", x=(vel_t1+vel_t2)/2, y=y_text, font="8p,Helvetica,black", offset="0c/0.2c")
    fig.text(text=f"{vel_Gr_c_23:.1f} ± {err_Gr_c_23:.1f} cm/yr", x=(vel_t3+vel_t4)/2, y=y_text, font="8p,Helvetica,black", offset="0c/0.2c")
    
    fig.plot(x=insar_Gr["ts_dates"], y=time_series_Gr[0]*unit, style="c0.1c", fill="dodgerblue4") #, label = "A2") #, label="%s A2 InSAR baseline" % tar_station)
    fig.text(text="M6.4", x=eq1, y=-15, justify="LM", offset="0.1c/0.0c",)
    fig.text(text="M6.4", x=eq2, y=-15, justify="LM", offset="0.1c/0.0c",)
    

    region_ts = [t0-0.1, te+0.1, -0.06*100, 0.26*100]
    fig.basemap(frame=["lStE", "xaf", "ya+lLOS (cm)"], region=region_ts, projection="X?", panel=True)
    fig.plot(x=[2021.9167, 2021.9167,  2022.1667,  2022.1667, 2021.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2022.9167, 2022.9167,  2023.1667,  2023.1667, 2022.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2023.9167, 2023.9167,  2024.1667,  2024.1667, 2023.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)    
    fig.plot(x=[eq1, eq1], y=[-0.05*100, 0.3*100],  pen="1p,black,--", region=region_ts) 
    fig.plot(x=[eq2, eq2], y=[-0.05*100, 0.3*1005],  pen="1p,black,--", region=region_ts)
    
    fig.plot(x=[vel_t1, vel_t2], y=[15, 15],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.plot(x=[vel_t3, vel_t4], y=[0, 0],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.text(text=f"{vel_Gr_d_22:.1f} ± {err_Gr_d_22:.1f} cm/yr", x=(vel_t1+vel_t2)/2, y=15, font="8p,Helvetica,black", offset="0c/0.2c")
    fig.text(text=f"{vel_Gr_d_23:.1f} ± {err_Gr_d_23:.1f} cm/yr", x=(vel_t3+vel_t4)/2, y=0, font="8p,Helvetica,black", offset="0c/0.2c")
    
    fig.plot(x=insar_Gr["ts_dates"], y=time_series_Gr[1]*unit, style="c0.1c", fill="dodgerblue4") #, label = "A2") #, label="%s A2 InSAR baseline" % tar_station)
    
    
########3 Map Eel
# Download DEM


fig.shift_origin(xshift="-13c")
fig.shift_origin(yshift="-7.5c")

with fig.subplot(nrows=1, ncols=2, figsize=("12.0c", "6.4c"), autolabel="e)", sharex="l", frame=["WSrt"], margins=["0.3c", "0.3c"],):
    
    lon_step = 0.05
    lat_step = 0.04
    
    region = "%s/%s/%s/%s" % (points_Ee[0][0]-lon_step, points_Ee[0][0]+lon_step, points_Ee[0][1]-lat_step, points_Ee[0][1]+lat_step), # Landslide 2 Graham
    grid = pygmt.datasets.load_earth_relief(region=region, resolution="01s")
    dgrid = pygmt.grdgradient(grid=grid, radiance=[315, 30], region=region)
    
    fig.basemap(frame=["WSrt", "xa0.1", "ya0.1"], region=region, projection=fig_size, panel=True)
    fig.grdimage(grid=grid, projection=fig_size, frame=["lbrt", "xa", "ya"], cmap="grey", shading=dgrid, region=region)
    fig.plot(data=landslide_poly_file, pen="1.0p,dodgerblue4", projection=fig_size)
    
    labels = ["g)", "h)"]
    for (lon, lat), lbl in zip(points_Ee, labels):
        # Plot marker at (lon, lat)
        fig.plot(x=lon, y=lat, style="c.2c", pen="0.8p,black", projection=fig_size)
    
        # Label the marker with offset 0.5 cm to the right
        fig.text(text=lbl, x=lon, y=lat, justify="RM", offset="0.5c/0c", font="10p,Helvetica,black", 
                 region=region, projection=fig_size)
    
    fig.plot(y=eel_lat, x=eel_lon, style="s.15c", fill="black", pen="0.8p,black", region=region, projection=fig_size)
    fig.basemap(frame=["WSrt", "xa0.1", "ya0.1"], map_scale="jBR+w2k+o0.3/0.5c", projection=fig_size)
    
    fig.basemap(frame=["wSrt", "xa0.1", "ya0.1"], region=region, projection=fig_size, panel=True)
    pygmt.makecpt(cmap="vik", series=[-0.1, 0.1])
    fig.grdimage(grid=paths_170_5_28['EelRiver']['geo_velocity_msk_grd'], cmap=True, region=region, projection=fig_size)
    fig.plot(data=landslide_poly_file, pen="1.0p,white", projection=fig_size)
    
    
    
    fig.text(region=region, projection=fig_size, text="Boulder Creek", position="TC",offset ="0.0/-0.2c") 
    
    
    labels = ["g)", "h)"]
    for (lon, lat), lbl in zip(points_Ee, labels):
        # Plot marker at (lon, lat)
        fig.plot(x=lon, y=lat, style="c.2c", pen="0.8p,blacl", projection=fig_size)
    
        # Label the marker with offset 0.5 cm to the right
        fig.text(text=lbl, x=lon, y=lat, justify="RM", offset="0.5c/0c", font="10p,Helvetica,black", 
                 region=region, projection=fig_size)
    
    
    df = pd.DataFrame(
        data={
            "x": [-123.47859+lon_step*0.9,-123.47859+lon_step*0.9],
            "y": [40.06566+lat_step*0.9, 40.06566+lat_step*0.9],
            "east_velocity": [-0.173*0.8,(-0.9848/2.5)*0.8],
            "north_velocity": [-0.9848*0.8, (0.173/2.5)*0.8],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.5p,black", line=True, spec="e1.2/1/1", vector="0.3c+p2p+e+gblack",region=region, projection=fig_size)

    
    with pygmt.config(
             FONT_ANNOT_PRIMARY="18p,black", 
             FONT_ANNOT_SECONDARY="18p,black",
             FONT_LABEL="18p,black",
             ):
        pygmt.makecpt(cmap="vik", series=[-10, 10])
        fig.colorbar(position="jBL+o0.5c/0.5c+w3c/0.4c", frame=["xa", "y+lcm/yr"], projection=fig_size)
        
    fig.plot(y=eel_lat, x=eel_lon, style="s.15c", fill="black", pen="0.8p,black", region=region, projection=fig_size)

    fig.basemap(frame=["wSrt", "xa0.1", "ya0.1"], map_scale="jBR+w2k+o0.3/0.5c", projection=fig_size)
    
fig.shift_origin(xshift="w+1c")
fig.shift_origin(yshift="0c")

## ******** Profiles Panel ********* 


########3 Time series Eel 
with fig.subplot(nrows=2, ncols=1, figsize=("8.5c", "6.4c"), autolabel="g)", sharex="b", frame=["lSEt" , "xaf", "ya+l LOS (mm/yr)"],
                 margins=["0.05c", "0.05c"],
):
    region_ts = [t0-0.1, te+0.1, -20, 5]
    y_text = -18
    fig.basemap(frame=["lstE", "xaf+lDate", "ya+lLOS (cm)"], region=region_ts, projection="X?", panel=True)
    fig.plot(x=[2021.9167, 2021.9167,  2022.1667,  2022.1667, 2021.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2022.9167, 2022.9167,  2023.1667,  2023.1667, 2022.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2023.9167, 2023.9167,  2024.1667,  2024.1667, 2023.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)    
    fig.plot(x=[eq1, eq1], y=[-20, 5],  pen="1p,black,--",) 
    fig.plot(x=[eq2, eq2], y=[-20, 5],  pen="1p,black,--",) 
    fig.text(text="M6.4", x=eq1, y=0, justify="LM", offset="0.1c/0.0c",)
    fig.text(text="M6.4", x=eq2, y=0, justify="LM", offset="0.1c/0.0c",)
    fig.plot(x=[vel_t1, vel_t2], y=[y_text, y_text],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.plot(x=[vel_t3, vel_t4], y=[y_text, y_text],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.text(text=f"{vel_Ee_g_22:.1f} ± {err_Ee_g_22:.1f} cm/yr", x=(vel_t1+vel_t2)/2, y=y_text, font="8p,Helvetica,black", offset="0c/0.2c")
    fig.text(text=f"{vel_Ee_g_23:.1f} ± {err_Ee_g_23:.1f} cm/yr", x=(vel_t3+vel_t4)/2, y=y_text, font="8p,Helvetica,black", offset="0c/0.2c")
    
    fig.plot(x=insar_Ee["ts_dates"], y=time_series_Ee[0]*unit, style="c0.1c", fill="dodgerblue4") #, label = "A2") #, label="%s A2 InSAR baseline" % tar_station)
    
    y_text = -18
    region_ts = [t0-0.1, te+0.1, -20, 5]
    fig.basemap(frame=["lStE", "xaf", "ya+lLOS (cm)"], region=region_ts, projection="X?", panel=True)
    fig.plot(x=[2021.9167, 2021.9167,  2022.1667,  2022.1667, 2021.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2022.9167, 2022.9167,  2023.1667,  2023.1667, 2022.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2023.9167, 2023.9167,  2024.1667,  2024.1667, 2023.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)    
    fig.plot(x=[eq1, eq1], y=[-20, 5],  pen="1p,black,--",) 
    fig.plot(x=[eq2, eq2], y=[-20, 5],  pen="1p,black,--",) 

    fig.plot(x=[vel_t1, vel_t2], y=[y_text, y_text],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.plot(x=[vel_t3, vel_t4], y=[y_text, y_text],  pen="2p,darkorange", region=region_ts, transparency=50)
    fig.text(text=f"{vel_Ee_h_22:.1f} ± {err_Ee_h_22:.1f} cm/yr", x=(vel_t1+vel_t2)/2, y=y_text, font="8p,Helvetica,black", offset="0c/0.2c")
    fig.text(text=f"{vel_Ee_h_23:.1f} ± {err_Ee_h_23:.1f} cm/yr", x=(vel_t3+vel_t4)/2, y=y_text, font="8p,Helvetica,black", offset="0c/0.2c")
    
    fig.plot(x=insar_Ee["ts_dates"], y=time_series_Ee[1]*unit, style="c0.1c", fill="dodgerblue4") #, label = "A2") #, label="%s A2 InSAR baseline" % tar_station)
    
    
fig.savefig(common_paths["fig_dir"]+"Fig_12_landslide_examples.png", transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+"Fig_12_landslide_examples.pdf", transparent=False, crop=True, anti_alias=True, show=False)

fig.show()



print(f"Point c — inc {geometry_Gr["incidenceAngle"][0]}, slope {90*geometry_Gr["slope"][0]}, "
      f"Point d - inc {geometry_Gr["incidenceAngle"][1]}, slope {90*geometry_Gr["slope"][1]}, ")
      
# 2022 interval at point c
print(f"2022 Point c — LOS: {vel_Gr_c_22:.2f} cm/yr, "
      f"downslope: {DS_vel_Gr_c_22:.2f} cm/yr, "
      f"Δ = {vel_Gr_c_22 - DS_vel_Gr_c_22:.2f} cm/yr")

# 2023 interval at point c
print(f"2023 Point c — LOS: {vel_Gr_c_23:.2f} cm/yr, "
      f"downslope: {DS_vel_Gr_c_23:.2f} cm/yr, "
      f"Δ = {vel_Gr_c_23 - DS_vel_Gr_c_23:.2f} cm/yr")

# 2022 interval at point d
print(f"2022 Point d — LOS: {vel_Gr_d_22:.2f} cm/yr, "
      f"downslope: {DS_vel_Gr_d_22:.2f} cm/yr, "
      f"Δ = {vel_Gr_d_22 - DS_vel_Gr_d_22:.2f} cm/yr")

# 2023 interval at point d
print(f"2023 Point d — LOS: {vel_Gr_d_23:.2f} cm/yr, "
      f"downslope: {DS_vel_Gr_d_23:.2f} cm/yr, "
      f"Δ = {vel_Gr_d_23 - DS_vel_Gr_d_23:.2f} cm/yr")


print(
    f"Gr Point c — inc {geometry_Gr['incidenceAngle'][0]:.1f}°, "
    f"slope {90*geometry_Gr['slope'][0]:.1f}°, "
    f"aspect {geometry_Gr['aspect'][0]:.1f}°; "
    f"Point d — inc {geometry_Gr['incidenceAngle'][1]:.1f}°, "
    f"slope {90*geometry_Gr['slope'][1]:.1f}°, "
    f"aspect {geometry_Gr['aspect'][1]:.1f}°"
)

# 2022 interval at Gr point c
print(f"2022 Gr c — LOS: {vel_Gr_c_22:.2f} cm/yr, "
      f"downslope: {DS_vel_Gr_c_22:.2f} cm/yr, "
      f"Δ = {vel_Gr_c_22 - DS_vel_Gr_c_22:.2f} cm/yr")
# 2023 interval at Gr point c
print(f"2023 Gr c — LOS: {vel_Gr_c_23:.2f} cm/yr, "
      f"downslope: {DS_vel_Gr_c_23:.2f} cm/yr, "
      f"Δ = {vel_Gr_c_23 - DS_vel_Gr_c_23:.2f} cm/yr")
# 2022 interval at Gr point d
print(f"2022 Gr d — LOS: {vel_Gr_d_22:.2f} cm/yr, "
      f"downslope: {DS_vel_Gr_d_22:.2f} cm/yr, "
      f"Δ = {vel_Gr_d_22 - DS_vel_Gr_d_22:.2f} cm/yr")
# 2023 interval at Gr point d
print(f"2023 Gr d — LOS: {vel_Gr_d_23:.2f} cm/yr, "
      f"downslope: {DS_vel_Gr_d_23:.2f} cm/yr, "
      f"Δ = {vel_Gr_d_23 - DS_vel_Gr_d_23:.2f} cm/yr")


# Eel River (Ee) points g & h
print(
    f"Ee Point g — inc {geometry_Ee['incidenceAngle'][0]:.1f}°, "
    f"slope {90*geometry_Ee['slope'][0]:.1f}°, "
    f"aspect {geometry_Ee['aspect'][0]:.1f}°; "
    f"Point h — inc {geometry_Ee['incidenceAngle'][1]:.1f}°, "
    f"slope {90*geometry_Ee['slope'][1]:.1f}°, "
    f"aspect {geometry_Ee['aspect'][1]:.1f}°"
)

# 2022 interval at Ee point g
print(f"2022 Ee g — LOS: {vel_Ee_g_22:.2f} cm/yr, "
      f"downslope: {DS_vel_Ee_g_22:.2f} cm/yr, "
      f"Δ = {vel_Ee_g_22 - DS_vel_Ee_g_22:.2f} cm/yr")
# 2023 interval at Ee point g
print(f"2023 Ee g — LOS: {vel_Ee_g_23:.2f} cm/yr, "
      f"downslope: {DS_vel_Ee_g_23:.2f} cm/yr, "
      f"Δ = {vel_Ee_g_23 - DS_vel_Ee_g_23:.2f} cm/yr")
# 2022 interval at Ee point h
print(f"2022 Ee h — LOS: {vel_Ee_h_22:.2f} cm/yr, "
      f"downslope: {DS_vel_Ee_h_22:.2f} cm/yr, "
      f"Δ = {vel_Ee_h_22 - DS_vel_Ee_h_22:.2f} cm/yr")
# 2023 interval at Ee point h
print(f"2023 Ee h — LOS: {vel_Ee_h_23:.2f} cm/yr, "
      f"downslope: {DS_vel_Ee_h_23:.2f} cm/yr, "
      f"Δ = {vel_Ee_h_23 - DS_vel_Ee_h_23:.2f} cm/yr")
