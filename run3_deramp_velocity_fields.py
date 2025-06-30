#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:42:19 2025
Deramp InSAR velocities using quadratic deramp.

***** Run in pygmt conda environment ***** 

@author: daniellelindsay
"""

import insar_utils as utils
from NC_ALOS2_filepaths import (paths_gps, paths_068, paths_169, paths_170, common_paths)

unit = 1000 # 1000 = mm 

# ------------------------
# Parameters
# ------------------------
#dist = 0.004    # Averaging distance in degrees (roughly 1km)
#ref_station = "CASR"
dist = common_paths["dist"]
ref_station = common_paths["ref_station"]

# Orbit dictionaries for each track
des169_dic = {"ORBIT_DIRECTION": "descending", "PATH": "169", "Frames": "2800_2850",
              "Heading": -166.92284024662985, "Platform": "ALOS-2", "Sensor": "a2"}
des170_dic = {"ORBIT_DIRECTION": "descending", "PATH": "170", "Frames": "2800_2850",
              "Heading": -166.9210958373203, "Platform": "ALOS-2", "Sensor": "a2"}
asc068_dic  = {"ORBIT_DIRECTION": "ascending",  "PATH": "068", "Frames": "0800_0750",
              "Heading": -13.078884587112812, "Platform": "ALOS-2", "Sensor": "a2"}

# ------------------------
# Load GPS Data
# ------------------------
gps_UNR_169 = utils.load_UNR_gps(paths_gps["169_enu"], ref_station)
gps_UNR_170 = utils.load_UNR_gps(paths_gps["170_enu"], ref_station)
gps_UNR_068 = utils.load_UNR_gps(paths_gps["068_enu"], ref_station)

# ------------------------
# Set InSAR File Paths from the dictionaries
# ------------------------
# For each track we use the "geo" keys from the corresponding paths dictionary.
geo_file_169 = paths_169["geo"]["geo_geometryRadar"]
geo_file_170 = paths_170["geo"]["geo_geometryRadar"]
geo_file_068 = paths_068["geo"]["geo_geometryRadar"]

vel_file_169 = paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]
vel_file_170 = paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]
vel_file_068 = paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]

# ------------------------
# Load InSAR Data
# ------------------------
insar_169, shape169 = utils.load_insar_vel_as_df(geo_file_169, vel_file_169, des169_dic)
insar_170, shape170 = utils.load_insar_vel_as_df(geo_file_170, vel_file_170, des170_dic)
insar_068, shape068 = utils.load_insar_vel_as_df(geo_file_068, vel_file_068, asc068_dic)

# Convert to mm for GPS comparison 
insar_169["Vel"] = insar_169["Vel"] * unit 
insar_170["Vel"] = insar_170["Vel"] * unit 
insar_068["Vel"] = insar_068["Vel"] * unit 
# ------------------------
# Process InSAR Data
# ------------------------
# Calculate average InSAR velocity for each GPS point
gps_UNR_169 = utils.calculate_average_insar_velocity(gps_UNR_169, insar_169, dist)
gps_UNR_170 = utils.calculate_average_insar_velocity(gps_UNR_170, insar_170, dist)
gps_UNR_068 = utils.calculate_average_insar_velocity(gps_UNR_068, insar_068, dist)

# Project GPS ENU velocities to InSAR LOS
gps_UNR_169 = utils.calculate_gps_los(gps_UNR_169, insar_169)
gps_UNR_170 = utils.calculate_gps_los(gps_UNR_170, insar_170)
gps_UNR_068 = utils.calculate_gps_los(gps_UNR_068, insar_068)

# Save original InSAR velocities for reference
insar_169["Vel_ori"] = insar_169["Vel"]
insar_170["Vel_ori"] = insar_170["Vel"]
insar_068["Vel_ori"] = insar_068["Vel"]

# ------------------------
# Deramp InSAR (Quadratic Ramp Removal)
# ------------------------
insar_169 = utils.apply_quadratic_deramp_2D(gps_UNR_169, insar_169)
insar_170 = utils.apply_quadratic_deramp_2D(gps_UNR_170, insar_170)
insar_068 = utils.apply_quadratic_deramp_2D(gps_UNR_068, insar_068)

# Update InSAR velocities to the deramped values
insar_169["Vel"] = insar_169["Vel_quadramp"]
insar_170["Vel"] = insar_170["Vel_quadramp"]
insar_068["Vel"] = insar_068["Vel_quadramp"]

# ------------------------
# Save New H5 Files with Deramped Velocities
# ------------------------
# Velocities are scaled by 1/1000 (e.g., converting from mm/yr to m/yr for consistency).
utils.write_new_h5(insar_169["Vel"]/unit, vel_file_169, shape169, "deramp")
utils.write_new_h5(insar_170["Vel"]/unit, vel_file_170, shape170, "deramp")
utils.write_new_h5(insar_068["Vel"]/unit, vel_file_068, shape068, "deramp")


# --- Prep vel files ---    
# Calculate diff.h5
utils.run_command(["diff.py", paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14"], paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o", paths_169["geo"]["diff_deramp"]])
utils.run_command(["diff.py", paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14"], paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o", paths_170["geo"]["diff_deramp"]])
utils.run_command(["diff.py", paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14"], paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o", paths_068["geo"]["diff_deramp"]])

# Save as gmt grd for plotting 
utils.run_command(["save_gmt.py", paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o",  paths_169["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]])
utils.run_command(["save_gmt.py", paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o",  paths_170["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]])
utils.run_command(["save_gmt.py", paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o",  paths_068["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]])

# --- Convert all .grd files to mm (×1000) for each track ---
for track in (paths_169, paths_170, paths_068):
    for name, grd_path in track["grd"].items():
        if grd_path.endswith(".grd"):
            mm_path = grd_path.replace(".grd", "_mm.grd")
            utils.run_command([
                "gmt", "grdmath",
                grd_path, "1000", "MUL", "=", mm_path
            ])