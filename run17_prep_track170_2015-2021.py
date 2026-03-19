#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import insar_utils as utils
from NC_ALOS2_filepaths import (paths_gps, paths_170, paths_169, common_paths)
import numpy as np
import os
import subprocess

# Run these before hand.... 
#timeseries2velocity.py timeseries_SET_ERA5_demErr.h5 --end-date 20200101 -o velocity_endDate20200101.h5
#/Users/daniellelindsay/miniconda3/envs/MintPy_24_2/bin/geocode.py velocity_endDate20200101.h5 -l inputs/geometryRadar.h5 --lalo 0.002 0.002 --bbox 36.17 42.41 -124.63 -119.22 --outdir geo/.
#plate_motion.py -g geo_geometryRadar.h5 -v geo_velocity_endDate20200101.h5 --plate NorthAmerica -o geo_velocity_endDate20200101_ITRF14.h5
#mask.py geo_velocity_endDate20200101_ITRF14.h5 -m geo_maskTempCoh.h5 -o geo_velocity_endDate20200101_ITRF14_msk.h5

# ------------------------
# Parameters
# ------------------------
ref_station       = "P345"

unit = 1000 # 1000 = mm 

# ------------------------
# Parameters
# ------------------------
dist = common_paths["dist"]
ref_station = common_paths["ref_station"]

# Orbit dictionaries for each track
des170_dic = {"ORBIT_DIRECTION": "descending", "PATH": "170", "Frames": "2800_2850",
              "Heading": -166.9210958373203, "Platform": "ALOS-2", "Sensor": "a2"}

# ------------------------
# Set InSAR File Paths from the dictionaries
# ------------------------
geo_file_170 = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/P345/geo/geo_geometryRadar.h5"
vel_file_170 = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/P345/geo/geo_velocity_endDate20200101_ITRF14_msk.h5"
itrf_LOS_170 = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/P345/geo/ITRF14.h5"

# ------------------------
# Load InSAR Data
# ------------------------
insar_170, shape170 = utils.load_insar_vel_as_df(geo_file_170, vel_file_170, des170_dic)
insar_170["Vel"] = insar_170["Vel"] * unit

# ------------------------
# Load GPS Data
# ------------------------
gps_170 = utils.load_UNR_gps(paths_gps["170_enu_IGS14"])
gps_170 = gps_170[gps_170['Vu'] >= -5]

# Project GPS ENU velocities to InSAR LOS
gps_170 = utils.project_gps2los(gps_170, insar_170)
gps_170 = utils.ref_los_to_station(gps_170, ref_station)
gps_170 = utils.gps_LOS_correction_plate_motion(geo_file_170, itrf_LOS_170, gps_170, ref_station, unit)

# ------------------------
# Process InSAR Data
# ------------------------
gps_170 = utils.calculate_average_insar_velocity(gps_170, insar_170, dist)
insar_170["Vel_ori"] = insar_170["Vel"]

# ------------------------
# Deramp InSAR (Quadratic Ramp Removal)
# ------------------------
insar_170 = utils.apply_quadratic_deramp_2D(gps_170, insar_170)
insar_170["Vel"] = insar_170["Vel_quadramp"]

utils.write_new_h5(insar_170["Vel"] / unit, "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/P345/geo/geo_velocity_endDate20200101_ITRF14_msk.h5", shape170, "deramp")


# ------------------------
# TRACK 169
# ------------------------


# Orbit dictionaries for each track
des169_dic = {"ORBIT_DIRECTION": "descending", "PATH": "169", "Frames": "2800_2850",
              "Heading": -166.9210958373203, "Platform": "ALOS-2", "Sensor": "a2"}
# ------------------------
# Set InSAR File Paths from the dictionaries
# ------------------------
geo_file_169 = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/P345/geo/geo_geometryRadar.h5"
vel_file_169 = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/P345/geo/geo_velocity_endDate20200101_ITRF14_msk.h5"
itrf_LOS_169 = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/P345/geo/ITRF14.h5"
# ------------------------
# Load InSAR Data
# ------------------------
insar_169, shape169 = utils.load_insar_vel_as_df(geo_file_169, vel_file_169, des169_dic)
insar_169["Vel"] = insar_169["Vel"] * unit
# ------------------------
# Load GPS Data
# ------------------------
gps_169 = utils.load_UNR_gps(paths_gps["169_enu_IGS14"])
gps_169 = gps_169[gps_169['Vu'] >= -5]
# Project GPS ENU velocities to InSAR LOS
gps_169 = utils.project_gps2los(gps_169, insar_169)
gps_169 = utils.ref_los_to_station(gps_169, ref_station)
gps_169 = utils.gps_LOS_correction_plate_motion(geo_file_169, itrf_LOS_169, gps_169, ref_station, unit)
# ------------------------
# Process InSAR Data
# ------------------------
gps_169 = utils.calculate_average_insar_velocity(gps_169, insar_169, dist)
insar_169["Vel_ori"] = insar_169["Vel"]
# ------------------------
# Deramp InSAR (Quadratic Ramp Removal)
# ------------------------
insar_169 = utils.apply_quadratic_deramp_2D(gps_169, insar_169)
insar_169["Vel"] = insar_169["Vel_quadramp"]
utils.write_new_h5(insar_169["Vel"] / unit, "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/P345/geo/geo_velocity_endDate20200101_ITRF14_msk.h5", shape169, "deramp")

###### Now run theses ones:  
#save_gmt.py geo_velocity_endDate20200101_ITRF14_msk_deramp.h5
#gmt grdmath geo_velocity_endDate20200101_ITRF14_msk_deramp.grd 1000 MUL = geo_velocity_endDate20200101_ITRF14_msk_deramp_mm.grd
#gmt grdmath /Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/P345/geo/geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp_mm.grd geo_velocity_endDate20200101_ITRF14_msk_deramp_mm.grd SUB = diff_endDate20200101_ITRF14_msk_deramp_mm.grd