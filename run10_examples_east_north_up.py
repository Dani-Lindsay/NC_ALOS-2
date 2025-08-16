#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 22:37:25 2025

@author: daniellelindsay
"""

import insar_utils as utils
from NC_ALOS2_filepaths import (paths_gps, paths_068, paths_169, paths_170, common_paths, decomp)
import numpy as np                # Matrix calculations
import pandas as pd               # Pandas for data
import pygmt

dist = common_paths["dist"]
lat_step = common_paths["lat_step"]
lon_step = common_paths["lon_step"]

ref_station = common_paths["ref_station"]

min_lon=-124.63
max_lon=-121.0
min_lat=37.0
max_lat=42.2

NC_fig_region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)

east_grd = decomp["grd"]["gps_insar_east"]
north_grd = decomp["grd"]["gps_insar_north"]
up_grd = decomp["grd"]["gps_insar_up"]

# gey GPS
gps_df = utils.load_UNR_gps(paths_gps["170_enu_IGS14"])
ref = gps_df.loc[gps_df['StaID']==ref_station]
if ref.empty:
    raise ValueError(f"Reference station '{ref_station}' not in gps_df")
ref_lon, ref_lat, ref_Ve, ref_Vn, ref_Vu = ref[['Lon','Lat', 'Ve', 'Vn', 'Vu']].iloc[0]

gps_df['Ve'] = gps_df['Ve'] - ref_Ve
gps_df['Vn'] = gps_df['Vn'] - ref_Vn
gps_df['Vu'] = gps_df['Vu'] - ref_Vu


# Set lat and lon for plotting from the gps file. 
ref_lat = gps_df.loc[gps_df["StaID"] == ref_station, "Lat"].values
ref_lon = gps_df.loc[gps_df["StaID"] == ref_station, "Lon"].values

###########################
# Plot 
###########################

style="c.03c"
res_style="c.1c"
size = "M7c"
vmin = 20

NC_grid = pygmt.datasets.load_earth_relief(region=NC_fig_region, resolution="15s")
NC_dgrid = pygmt.grdgradient(grid=NC_grid, radiance=[315, 30], region=NC_fig_region)

fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=12, FONT_TITLE=14, FONT_SUBTITLE = 12, MAP_TITLE_OFFSET= "-7p")

#Plot main map 
fig.basemap(region=NC_fig_region, projection= size,frame=["WSrt", "xa", "ya"],)
fig.grdimage(grid=NC_grid, projection=size, frame=["lbrt", "xa", "ya"], cmap='wiki-france.cpt', shading=NC_dgrid, transparency=40)
pygmt.makecpt(cmap="vik", series=[-25, 25])
fig.grdimage(grid=north_grd, cmap=True, nan_transparent=True, region=NC_fig_region, projection= size)
# Plot Faults
for fault_file in common_paths["fault_files"]:
    fig.plot(data=fault_file, pen="0.5p,black", transparency=50, region=NC_fig_region, projection= size)
    
fig.coast(shorelines=True, frame = ["+tNorth Velocity"], region=NC_fig_region, projection= size)

fig.text(text="a)", position="TL", offset="0.1c/-0.1c", justify="TL", region=NC_fig_region, projection= size)
fig.plot(x=gps_df["Lon"], y=gps_df["Lat"],  style="c.2c", fill=gps_df["Vn"], pen="0.8p,black", cmap=True, region=NC_fig_region, projection= size)

fig.basemap(frame=["WSrt", "xa", "ya"], map_scale="jBL+w100k+o1.7/0.8c", projection=size)

with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
    fig.colorbar(position="jBL+o0.4c/0.4c+w4c/0.4c", frame=["xaf", "y+lmm/yr"],)
    
fig.shift_origin(xshift="7.5c")

# #Plot main map 
fig.basemap(region=NC_fig_region, projection= size,frame=["wSrt", "xa", "ya"],)
fig.grdimage(grid=NC_grid, projection=size, frame=["lbrt", "xa", "ya"], cmap='wiki-france.cpt', shading=NC_dgrid, transparency=40)
pygmt.makecpt(cmap="vik", series=[-25, 25])
fig.grdimage(grid=east_grd, cmap=True, nan_transparent=True, region=NC_fig_region, projection= size)
# Plot Faults
for fault_file in common_paths["fault_files"]:
    fig.plot(data=fault_file, pen="0.5p,black", transparency=50, region=NC_fig_region, projection= size)

fig.text(text="b)", position="TL", offset="0.1c/-0.1c", justify="TL", region=NC_fig_region, projection= size)
fig.coast(shorelines=True, frame = ["+tEast Velocity"], region=NC_fig_region, projection= size)
fig.plot(x=gps_df["Lon"], y=gps_df["Lat"],  style="c.2c", fill=gps_df["Ve"], pen="0.8p,black", cmap=True, region=NC_fig_region, projection= size)

with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
    fig.colorbar(position="jBL+o0.4c/0.4c+w4c/0.4c", frame=["xaf", "y+lmm/yr"],)

    
fig.shift_origin(xshift="7.5c")

# #Plot main map 
fig.basemap(region=NC_fig_region, projection= size,frame=["wSrt", "xa", "ya"],)
fig.grdimage(grid=NC_grid, projection=size, frame=["lbrt", "xa", "ya"], cmap='wiki-france.cpt', shading=NC_dgrid, transparency=40)
pygmt.makecpt(cmap="vik", series=[-12.5, 12.5])
fig.grdimage(grid=up_grd, cmap=True, nan_transparent=True, region=NC_fig_region, projection= size)
# Plot Faults
for fault_file in common_paths["fault_files"]:
    fig.plot(data=fault_file, pen="0.5p,black", transparency=50, region=NC_fig_region, projection= size)

fig.coast(shorelines=True, frame = ["+tUp Velocity"], region=NC_fig_region, projection= size)
fig.plot(x=gps_df["Lon"], y=gps_df["Lat"],  style="c.2c", fill=gps_df["Vu"], pen="0.8p,black", cmap=True, region=NC_fig_region, projection= size)
#fig.plot(y=ref_lat, x=ref_lon, style="s.3c", fill="black", pen="0.8p,black", region=NC_fig_region, projection= size)

fig.text(text="c)", position="TL", offset="0.1c/-0.1c", justify="TL", region=NC_fig_region, projection= size)

sf_min_lon = -122.7
sf_max_lon = -122.0
sf_min_lat = 37.5
sf_max_lat = 38.0

fig.plot(x = [sf_min_lon, sf_min_lon, sf_max_lon, sf_max_lon, sf_min_lon], 
         y = [sf_min_lat, sf_max_lat, sf_max_lat, sf_min_lat, sf_min_lat], 
         pen="0.8p,black,--", transparency=0)

with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
    fig.colorbar(position="jBL+o0.4c/0.4c+w4c/0.4c", frame=["xaf", "y+lmm/yr"],)
    
fig.shift_origin(xshift="7.5c")

# #Plot main map 
fig.basemap(region=NC_fig_region, projection= size,frame=["wSrt", "xa", "ya"],)
fig.grdimage(grid=NC_grid, projection=size, frame=["lbrt", "xa", "ya"], cmap='wiki-france.cpt', shading=NC_dgrid, transparency=40)
pygmt.makecpt(cmap="vik", series=[-12.5, 12.5])
fig.grdimage(grid=up_grd, cmap=True, nan_transparent=True, region=NC_fig_region, projection= size)
# Plot Faults
for fault_file in common_paths["fault_files"]:
    fig.plot(data=fault_file, pen="0.5p,black", transparency=50, region=NC_fig_region, projection= size)

fig.coast(shorelines=True, frame = ["+tUp Velocity"], region=NC_fig_region, projection= size)
fig.plot(x=gps_df["Lon"], y=gps_df["Lat"],  style="c.2c", fill=gps_df["Vu"], pen="0.8p,black", cmap=True, region=NC_fig_region, projection= size)

fig.text(text="d)", position="TL", offset="0.1c/-0.1c", justify="TL", region=NC_fig_region, projection= size)



good = ['P318', 'P060', 'LRA3', 'P189',  'P264', 'P341', 'P223']
bad  = ['P786', 'TRND', 'P219', 'P343', 'P162', 'P165', 'P193', 'P270', 'P158', 'P344']


station_ids = bad # Loop over desired station IDs and add text labels
for sta in station_ids:
    row = gps_df[gps_df["StaID"] == sta]
    if row.empty:
        # Optionally warn if station not found
        print(f"Warning: station {sta} not found in gps_df")
        continue
    lon = row["Lon"].values[0]
    lat = row["Lat"].values[0]
    # Adjust offset as needed: here 0.1c to the right and 0.1c up
    fig.text(x=lon, y=lat, text="%s" % sta,  font="10p,Helvetica,black", offset="0.5c/0.5c+v", justify="LM", fill="salmon", transparency=0)
    fig.text(x=lon, y=lat, text="%s" % sta,  font="10p,Helvetica,black", offset="0.5c/0.5c+v", justify="LM")


station_ids = good
# Loop over desired station IDs and add text labels
for sta in station_ids:
    row = gps_df[gps_df["StaID"] == sta]
    if row.empty:
        # Optionally warn if station not found
        print(f"Warning: station {sta} not found in gps_df")
        continue
    lon = row["Lon"].values[0]
    lat = row["Lat"].values[0]
    # Adjust offset as needed: here 0.1c to the right and 0.1c up
    fig.text(x=lon, y=lat, text="%s" % sta,  font="10p,Helvetica,black", offset="0.5c/0.5c+v", justify="LM", fill="aquamarine3", transparency=0)
    fig.text(x=lon, y=lat, text="%s" % sta,  font="10p,Helvetica,black", offset="0.5c/0.5c+v", justify="LM")

station_ids = ['P181','CCSF',]
# Loop over desired station IDs and add text labels
for sta in station_ids:
    row = gps_df[gps_df["StaID"] == sta]
    if row.empty:
        # Optionally warn if station not found
        print(f"Warning: station {sta} not found in gps_df")
        continue
    lon = row["Lon"].values[0]
    lat = row["Lat"].values[0]
    # Adjust offset as needed: here 0.1c to the right and 0.1c up
    fig.text(x=lon, y=lat, text="%s" % sta,  font="10p,Helvetica,black", offset="-0.5c/-0.25c+v", justify="RM", fill="aquamarine3", transparency=0)
    fig.text(x=lon, y=lat, text="%s" % sta,  font="10p,Helvetica,black", offset="-0.5c/-0.25c+v", justify="RM")

station_ids = ['WIN2']
# Loop over desired station IDs and add text labels
for sta in station_ids:
    row = gps_df[gps_df["StaID"] == sta]
    if row.empty:
        # Optionally warn if station not found
        print(f"Warning: station {sta} not found in gps_df")
        continue
    lon = row["Lon"].values[0]
    lat = row["Lat"].values[0]
    # Adjust offset as needed: here 0.1c to the right and 0.1c up
    fig.text(x=lon, y=lat, text="%s" % sta,  font="10p,Helvetica,black", offset="-0.5c/-0.5c+v", justify="RM", fill="aquamarine3", transparency=0)
    fig.text(x=lon, y=lat, text="%s" % sta,  font="10p,Helvetica,black", offset="-0.5c/-0.5c+v", justify="RM")


with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
    fig.colorbar(position="jBL+o0.4c/0.4c+w4c/0.4c", frame=["xaf", "y+lmm/yr"],)
  
    
fig.savefig(common_paths["fig_dir"]+f"Fig_8_{ref_station}_East_North_Up_170_068_GNSS_INSAR.png", transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+f"Fig_8_{ref_station}_East_North_Up_170_068_GNSS_INSAR.pdf", transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+f"Fig_8_{ref_station}_East_North_Up_170_068_GNSS_INSAR.jpg", transparent=False, crop=True, anti_alias=True, show=False)
fig.show()  

