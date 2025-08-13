#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:29:00 2025

@author: daniellelindsay
"""

import glob
import os
import insar_utils as utils

fig_dir = "/Volumes/WD2TB_Phd/NC_ALOS-2/Figures_08_13/"
data_dir = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo"
gps_dir = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/GPS"
inputs_dir = os.path.join(data_dir, "Inputs")


meter_step = 180  # Target pixel size in meters
lat_deg = 40.0    # Central lattitude in the frame
lat_step, lon_step = utils.meters_to_degrees(meter_step, lat_deg)

# ------------------------
# Common Paths
# ------------------------
common_paths = {
    "fig_dir": fig_dir,
    "gps_dir": gps_dir,
    "ref_station" : "CASR",
    "ref_lat" : 38.43978,
    "ref_lon" : -122.74691,
    "fault_files": glob.glob(os.path.join(inputs_dir, "qfaults", "*.txt")),
    "pb_file":  os.path.join(inputs_dir, "transform.gmt"),
    "pb2_file": os.path.join(inputs_dir, "trench.gmt"),
    "pb3_file": os.path.join(inputs_dir, "ridge.gmt"),
    "frames": {
        "170_2800": os.path.join(inputs_dir, "Frame_170_2800.txt"),
        "170_2850": os.path.join(inputs_dir, "Frame_170_2850.txt"),
        "169_2800": os.path.join(inputs_dir, "Frame_169_2800.txt"),
        "169_2850": os.path.join(inputs_dir, "Frame_169_2850.txt"),
        "068_0800": os.path.join(inputs_dir, "Frame_068_0800.txt"),
        "068_0750": os.path.join(inputs_dir, "Frame_068_0750.txt"),
        "169_170_overlap": os.path.join(inputs_dir, "Frame_169_170_overlap.txt"),
    },
    "bbox": {"w": "-124.63", "e": "-119.22", "s": "36.17", "n": "42.41"},
    "dist" : 0.004, #Region overwhich to average velocities ~1km in each direction 0.008 = 1km.
    "lat_step" : str(0.002), # needs to be 0.002 to match VISR interpolation 
    "lon_step" : str(0.002), # needs to be 0.002 to match VISR interpolation 
    "MLV_level":        os.path.join(inputs_dir, "MLV_level_path.txt"),
    "canals_file":      os.path.join(inputs_dir, "ca_canals.txt"),
    "Tehama-Colusa_file": os.path.join(inputs_dir, "Tehama-Colusa_canal_cumdisp22.txt"),
    "Artois_file":      os.path.join(inputs_dir, "Artois_canal_cumdisp22.txt"),
    "Arbuckle_file":    os.path.join(inputs_dir, "Arbuckle_canal_cumdisp22.txt"),
    "aquifer_file":     os.path.join(inputs_dir, "aquifer_boundaries.txt"),
    "roads_major":      os.path.join(inputs_dir, "Roads_Major_Highways.txt"),
    "roads_primary":    os.path.join(inputs_dir, "Roads_Primary_Secondary.txt"),
    "roads_local":      os.path.join(inputs_dir, "Roads_Residential_Local.txt"),
    "roads_tertiary":   os.path.join(inputs_dir, "Roads_Tertiary_Unclassified.txt"),
    "wcSlides":         os.path.join(inputs_dir,"wcSlides.gmt"),
}


# Create a dictionary with keys like "068_enu", "169_enu", and "170_enu"
paths_gps = {
    "Fig_Dir":      os.path.join(gps_dir, "Figures"),
    "DataHoldings": os.path.join(gps_dir, "DataHoldings.txt"),
    "Steps":        os.path.join(gps_dir, "steps.txt"),
    "UNRdaily_Dir": os.path.join(gps_dir, "ISG14"),
    "NA_Dir":       os.path.join(gps_dir, "NA"),
    "068_StaList":  os.path.join(data_dir, "068", "StationList.txt"),
    "169_StaList":  os.path.join(data_dir, "169", "StationList.txt"),
    "170_StaList":  os.path.join(data_dir, "170", "StationList.txt"),
    "068_enu_ISG14":os.path.join(data_dir, "068", "UNR_IGS14_gps.csv"),
    "169_enu_ISG14":os.path.join(data_dir, "169", "UNR_IGS14_gps.csv"),
    "170_enu_ISG14":os.path.join(data_dir, "170", "UNR_IGS14_gps.csv"),
    "068_enu_NA":   os.path.join(data_dir, "068", "UNR_NA_gps.csv"),
    "169_enu_NA":   os.path.join(data_dir, "169", "UNR_NA_gps.csv"),
    "170_enu_NA":   os.path.join(data_dir, "170", "UNR_NA_gps.csv"),
    "068_LOS_comp": os.path.join(data_dir, "068", "UNR_IGS14_gps_LOS_InSAR.csv"),
    "169_LOS_comp": os.path.join(data_dir, "169", "UNR_IGS14_gps_LOS_InSAR.csv"),
    "170_LOS_comp": os.path.join(data_dir, "170", "UNR_IGS14_gps_LOS_InSAR.csv"),
    "visr":{
        "east" :    os.path.join(gps_dir, "VISR","visr", "visr_intrp_e_NAN.grd"),
        "north" :   os.path.join(gps_dir, "VISR", "visr", "visr_intrp_n_NAN.grd"),
        "gps_enu" : os.path.join(gps_dir, "VISR", "GPS_resolved_IGS14_Sept24_CASR.txt"),
    }
}


    
# # ------------------------
# # Track 068 Paths
# # ------------------------
paths_068 = {
    "baselines" : os.path.join(data_dir, "068", "baseline_center.txt"),
    "CASR": {
        "coherence" : os.path.join(data_dir, "068", "CASR", "coherenceSpatialAvg.txt"),
        "timeseries": os.path.join(data_dir, "068", "CASR", "timeseries.h5"),
        "timeseries_SET": os.path.join(data_dir, "068", "CASR", "timeseries_SET.h5"),
        "timeseries_SET_ERA5": os.path.join(data_dir, "068", "CASR", "timeseries_SET_ERA5.h5"),
        "timeseries_SET_ERA5_demErr": os.path.join(data_dir, "068", "CASR", "timeseries_SET_ERA5_demErr.h5"),
        "velocity": os.path.join(data_dir, "068", "CASR", "velocity.h5"),
        "velocity_SET": os.path.join(data_dir, "068", "CASR", "velocity_SET.h5"),
        "velocity_SET_ERA5": os.path.join(data_dir, "068", "CASR", "velocity_SET_ERA5.h5"),
        "velocity_SET_ERA5_demErr": os.path.join(data_dir, "068", "CASR", "velocity_SET_ERA5_demErr.h5"),
        "maskTempCoh": os.path.join(data_dir, "068", "CASR", "maskTempCoh.h5"),
        "waterMask": os.path.join(data_dir, "068", "CASR", "waterMask.h5"),
        "velocityERA5": os.path.join(data_dir, "068", "CASR", "velocityERA5.h5"),
        "demErr": os.path.join(data_dir, "068", "CASR", "demErr.h5"),
    },
    "geo": {
        # Geometry files
        "geometryRadar": os.path.join(data_dir, "068", "CASR", "inputs", "geometryRadar.h5"),
        "geo_geometryRadar": os.path.join(data_dir, "068", "CASR", "geo", "geo_geometryRadar.h5"),
        # Velocity Files
        "geo_velocity": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity.h5"),
        "geo_velocity_SET": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET.h5"),
        "geo_velocity_SET_ERA5": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5.h5"),
        "geo_velocity_SET_ERA5_demErr": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr.h5"),
        "geo_velocity_SET_ERA5_demErr_ITRF14": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14.h5"),
        "geo_maskTempCoh": os.path.join(data_dir, "068", "CASR", "geo", "geo_maskTempCoh.h5"),
        "geo_waterMask": os.path.join(data_dir, "068", "CASR", "geo", "geo_waterMask.h5"),
        "geo_velocity_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_msk.h5"),
        "geo_velocity_SET_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_msk.h5"),
        "geo_velocity_SET_ERA5_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_msk.h5"),
        "geo_velocity_SET_ERA5_demErr_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_msk.h5"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk.h5"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp.h5"),
        # Correction Layers
        "diff_SET"   : os.path.join(data_dir, "068", "CASR", "geo", "diff_SET.h5"),
        "diff_ERA5"  : os.path.join(data_dir, "068", "CASR", "geo", "diff_ERA5.h5"),
        "diff_demErr": os.path.join(data_dir, "068", "CASR", "geo", "diff_demErr.h5"),
        "diff_ITRF14": os.path.join(data_dir, "068", "CASR", "geo", "diff_ITRF14.h5"),
        "diff_deramp": os.path.join(data_dir, "068", "CASR", "geo", "diff_deramp.h5"),
        "ITRF_enu"   : os.path.join(data_dir, "068", "CASR", "geo", "ITRF14enu.h5"),
        "ITRF_LOS"   : os.path.join(data_dir, "068", "CASR", "geo", "ITRF14.h5"),
#         "vel_grd": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp.grd"),
    },
    "grd":{
        "diff_SET": os.path.join(data_dir, "068", "CASR", "geo", "diff_SET.grd"),
        "diff_ERA5": os.path.join(data_dir, "068", "CASR", "geo", "diff_ERA5.grd"),
        "diff_demErr": os.path.join(data_dir, "068", "CASR", "geo", "diff_demErr.grd"),
        "diff_ITRF14": os.path.join(data_dir, "068", "CASR", "geo", "diff_ITRF14.grd"),
        "diff_deramp": os.path.join(data_dir, "068", "CASR", "geo", "diff_deramp.grd"),
        "geo_velocity_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_msk.grd"),
        "geo_velocity_SET_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_msk.grd"),
        "geo_velocity_SET_ERA5_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_msk.grd"),
        "geo_velocity_SET_ERA5_demErr_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_msk.grd"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk.grd"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp.grd"),
    },
    "grd_mm":{
        "diff_SET": os.path.join(data_dir, "068", "CASR", "geo", "diff_SET_mm.grd"),
        "diff_ERA5": os.path.join(data_dir, "068", "CASR", "geo", "diff_ERA5_mm.grd"),
        "diff_demErr": os.path.join(data_dir, "068", "CASR", "geo", "diff_demErr_mm.grd"),
        "diff_ITRF14": os.path.join(data_dir, "068", "CASR", "geo", "diff_ITRF14_mm.grd"),
        "diff_deramp": os.path.join(data_dir, "068", "CASR", "geo", "diff_deramp_mm.grd"),
        "geo_velocity_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_msk_mm.grd"),
        "geo_velocity_SET_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_msk_mm.grd"),
        "geo_velocity_SET_ERA5_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_msk_mm.grd"),
        "geo_velocity_SET_ERA5_demErr_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_msk_mm.grd"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_mm.grd"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk": os.path.join(data_dir, "068", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp_mm.grd"),
    },
    "P208": {
        "geo":              os.path.join(data_dir, "068", "P208", "geo"),
        "geometryRadar":    os.path.join(data_dir, "068", "P208", "inputs", "geometryRadar.h5"),
        "timeseries":       os.path.join(data_dir, "068", "P208", "timeseries_SET_ERA5_ramp_demErr.h5"),
        "velocity":         os.path.join(data_dir, "068", "P208", "velocity.h5"),
        "maskTempCoh":      os.path.join(data_dir, "068", "P208", "maskTempCoh.h5"),
        "geo_timeseries":   os.path.join(data_dir, "068", "P208", "geo", "geo_timeseries_SET_ERA5_ramp_demErr.h5"),
        "geo_velocity":     os.path.join(data_dir, "068", "P208", "geo", "geo_velocity.h5"),
        "geo_maskTempCoh":  os.path.join(data_dir, "068", "P208", "geo", "geo_maskTempCoh.h5"),
        "geo_velocity_msk": os.path.join(data_dir, "068", "P208", "geo", "geo_velocity_msk.h5"),
        "geo_geometryRadar":os.path.join(data_dir, "068", "P208", "geo", "geo_geometryRadar.h5"),
        "vel_grd":          os.path.join(data_dir, "068", "P208", "geo", "geo_velocity_msk.grd"),
    },
    "P784": {
        "geo":              os.path.join(data_dir, "068", "P784", "geo"),
        "geometryRadar":    os.path.join(data_dir, "068", "P784", "inputs", "geometryRadar.h5"),
        "timeseries":       os.path.join(data_dir, "068", "P784", "timeseries_SET_ERA5_ramp_demErr.h5"),
        "velocity":         os.path.join(data_dir, "068", "P784", "velocity.h5"),
        "maskTempCoh":      os.path.join(data_dir, "068", "P784", "maskTempCoh.h5"),
        "geo_timeseries":   os.path.join(data_dir, "068", "P784", "geo", "geo_timeseries_SET_ERA5_ramp_demErr.h5"),
        "geo_velocity":     os.path.join(data_dir, "068", "P784", "geo", "geo_velocity.h5"),
        "geo_maskTempCoh":  os.path.join(data_dir, "068", "P784", "geo", "geo_maskTempCoh.h5"),
        "geo_velocity_msk": os.path.join(data_dir, "068", "P784", "geo", "geo_velocity_msk.h5"),
        "geo_geometryRadar":os.path.join(data_dir, "068", "P784", "geo", "geo_geometryRadar.h5"),
        "vel_grd":          os.path.join(data_dir, "068", "P784", "geo", "geo_velocity_msk.grd"),
    },
}

# # ------------------------
# # Track 169 Paths
# # ------------------------

paths_169 = {
    "baselines" : os.path.join(data_dir, "169", "baseline_center.txt"),
    "CASR": {
        "coherence" : os.path.join(data_dir, "169", "CASR", "coherenceSpatialAvg.txt"),
        "timeseries": os.path.join(data_dir, "169", "CASR", "timeseries.h5"),
        "timeseries_SET": os.path.join(data_dir, "169", "CASR", "timeseries_SET.h5"),
        "timeseries_SET_ERA5": os.path.join(data_dir, "169", "CASR", "timeseries_SET_ERA5.h5"),
        "timeseries_SET_ERA5_demErr": os.path.join(data_dir, "169", "CASR", "timeseries_SET_ERA5_demErr.h5"),
        "velocity": os.path.join(data_dir, "169", "CASR", "velocity.h5"),
        "velocity_SET": os.path.join(data_dir, "169", "CASR", "velocity_SET.h5"),
        "velocity_SET_ERA5": os.path.join(data_dir, "169", "CASR", "velocity_SET_ERA5.h5"),
        "velocity_SET_ERA5_demErr": os.path.join(data_dir, "169", "CASR", "velocity_SET_ERA5_demErr.h5"),
        "maskTempCoh": os.path.join(data_dir, "169", "CASR", "maskTempCoh.h5"),
        "waterMask": os.path.join(data_dir, "169", "CASR", "waterMask.h5"),
        "velocityERA5": os.path.join(data_dir, "169", "CASR", "velocityERA5.h5"),
        "demErr": os.path.join(data_dir, "169", "CASR", "demErr.h5"),
    },
    "geo": {
        # Geometry Files
        "geometryRadar": os.path.join(data_dir, "169", "CASR", "inputs", "geometryRadar.h5"),
        "geo_geometryRadar": os.path.join(data_dir, "169", "CASR", "geo", "geo_geometryRadar.h5"),
        # Velocity Files
        "geo_velocity": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity.h5"),
        "geo_velocity_SET": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET.h5"),
        "geo_velocity_SET_ERA5": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5.h5"),
        "geo_velocity_SET_ERA5_demErr": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr.h5"),
        "geo_velocity_SET_ERA5_demErr_ITRF14": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14.h5"),
        "geo_maskTempCoh": os.path.join(data_dir, "169", "CASR", "geo", "geo_maskTempCoh.h5"),
        "geo_waterMask": os.path.join(data_dir, "169", "CASR", "geo", "geo_waterMask.h5"),
        "geo_velocity_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_msk.h5"),
        "geo_velocity_SET_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_msk.h5"),
        "geo_velocity_SET_ERA5_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_msk.h5"),
        "geo_velocity_SET_ERA5_demErr_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_msk.h5"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk.h5"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp.h5"),
        # Correction Layers
        "diff_SET":     os.path.join(data_dir, "169", "CASR", "geo", "diff_SET.h5"),
        "diff_ERA5":    os.path.join(data_dir, "169", "CASR", "geo", "diff_ERA5.h5"),
        "diff_demErr":  os.path.join(data_dir, "169", "CASR", "geo", "diff_demErr.h5"),
        "diff_ITRF14":  os.path.join(data_dir, "169", "CASR", "geo", "diff_ITRF14.h5"),
        "diff_deramp":  os.path.join(data_dir, "169", "CASR", "geo", "diff_deramp.h5"),
        "diff_169_170": os.path.join(data_dir, "169", "CASR", "geo", "diff_169_170.h5"),
        "ITRF_enu"   :  os.path.join(data_dir, "169", "CASR", "geo", "ITRF14enu.h5"),
        "ITRF_LOS"   :  os.path.join(data_dir, "169", "CASR", "geo", "ITRF14.h5"),
    },
    
    "grd":{
        "diff_SET": os.path.join(data_dir, "169", "CASR", "geo", "diff_SET.grd"),
        "diff_ERA5": os.path.join(data_dir, "169", "CASR", "geo", "diff_ERA5.grd"),
        "diff_demErr": os.path.join(data_dir, "169", "CASR", "geo", "diff_demErr.grd"),
        "diff_ITRF14": os.path.join(data_dir, "169", "CASR", "geo", "diff_ITRF14.grd"),
        "diff_169_170": os.path.join(data_dir, "169", "CASR", "geo", "diff_169_170.grd"),
        "diff_deramp": os.path.join(data_dir, "169", "CASR", "geo", "diff_deramp.grd"),
        "geo_velocity_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_msk.grd"),
        "geo_velocity_SET_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_msk.grd"),
        "geo_velocity_SET_ERA5_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_msk.grd"),
        "geo_velocity_SET_ERA5_demErr_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_msk.grd"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk.grd"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp.grd"),
    },
    
    "grd_mm":{
    "diff_SET": os.path.join(data_dir, "169", "CASR", "geo", "diff_SET_mm.grd"),
    "diff_ERA5": os.path.join(data_dir, "169", "CASR", "geo", "diff_ERA5_mm.grd"),
    "diff_demErr": os.path.join(data_dir, "169", "CASR", "geo", "diff_demErr_mm.grd"),
    "diff_ITRF14": os.path.join(data_dir, "169", "CASR", "geo", "diff_ITRF14_mm.grd"),
    "diff_deramp": os.path.join(data_dir, "169", "CASR", "geo", "diff_deramp_mm.grd"),
    "geo_velocity_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_msk_mm.grd"),
    "geo_velocity_SET_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_msk_mm.grd"),
    "geo_velocity_SET_ERA5_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_msk_mm.grd"),
    "geo_velocity_SET_ERA5_demErr_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_msk_mm.grd"),
    "geo_velocity_SET_ERA5_demErr_ITRF14_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_mm.grd"),
    "geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk": os.path.join(data_dir, "169", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp_mm.grd"),
    },
}

# ------------------------
# Track 170 Paths
# ------------------------
paths_170 = {
    "baselines" : os.path.join(data_dir, "170", "baseline_center.txt"),

    "CASR": {
        "coherence" : os.path.join(data_dir, "170", "CASR", "coherenceSpatialAvg.txt"),
        "timeseries": os.path.join(data_dir, "170", "CASR", "timeseries.h5"),
        "timeseries_SET": os.path.join(data_dir, "170", "CASR", "timeseries_SET.h5"),
        "timeseries_SET_ERA5": os.path.join(data_dir, "170", "CASR", "timeseries_SET_ERA5.h5"),
        "timeseries_SET_ERA5_demErr": os.path.join(data_dir, "170", "CASR", "timeseries_SET_ERA5_demErr.h5"),
        "velocity": os.path.join(data_dir, "170", "CASR", "velocity.h5"),
        "velocity_SET": os.path.join(data_dir, "170", "CASR", "velocity_SET.h5"),
        "velocity_SET_ERA5": os.path.join(data_dir, "170", "CASR", "velocity_SET_ERA5.h5"),
        "velocity_SET_ERA5_demErr": os.path.join(data_dir, "170", "CASR", "velocity_SET_ERA5_demErr.h5"),
        "maskTempCoh": os.path.join(data_dir, "170", "CASR", "maskTempCoh.h5"),
        "waterMask": os.path.join(data_dir, "170", "CASR", "waterMask.h5"),
        "velocityERA5": os.path.join(data_dir, "170", "CASR", "velocityERA5.h5"),
        "demErr": os.path.join(data_dir, "170", "CASR", "demErr.h5"),
    },    
    "geo": {
        # Geomtry Files
        "geometryRadar": os.path.join(data_dir, "170", "CASR", "inputs", "geometryRadar.h5"),
        "geo_geometryRadar": os.path.join(data_dir, "170", "CASR", "geo", "geo_geometryRadar.h5"),
        #Velocity Files
        "geo_velocity": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity.h5"),
        "geo_velocity_SET": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET.h5"),
        "geo_velocity_SET_ERA5": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5.h5"),
        "geo_velocity_SET_ERA5_demErr": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr.h5"),
        "geo_velocity_SET_ERA5_demErr_ITRF14": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14.h5"),
        "geo_maskTempCoh": os.path.join(data_dir, "170", "CASR", "geo", "geo_maskTempCoh.h5"),
        "geo_waterMask": os.path.join(data_dir, "170", "CASR", "geo", "geo_waterMask.h5"),
        "geo_velocity_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_msk.h5"),
        "geo_velocity_SET_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_msk.h5"),
        "geo_velocity_SET_ERA5_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_msk.h5"),
        "geo_velocity_SET_ERA5_demErr_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_msk.h5"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk.h5"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp.h5"),
        "geo_timeseries": os.path.join(data_dir, "170", "CASR", "geo", "geo_timeseries_SET_ERA5_demErr.h5"),
        "geo_timeseries_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_timeseries_SET_ERA5_demErr_msk.h5"),
        # Correction Layers
        "diff_SET":    os.path.join(data_dir, "170", "CASR", "geo", "diff_SET.h5"),
        "diff_ERA5":   os.path.join(data_dir, "170", "CASR", "geo", "diff_ERA5.h5"),
        "diff_demErr": os.path.join(data_dir, "170", "CASR", "geo", "diff_demErr.h5"),
        "diff_ITRF14": os.path.join(data_dir, "170", "CASR", "geo", "diff_ITRF14.h5"),
        "diff_deramp": os.path.join(data_dir, "170", "CASR", "geo", "diff_deramp.h5"),
        "ITRF_enu"   : os.path.join(data_dir, "170", "CASR", "geo", "ITRF14enu.h5"),
        "ITRF_LOS"   : os.path.join(data_dir, "170", "CASR", "geo", "ITRF14.h5"),
        "ITRF_enu_msk"   : os.path.join(data_dir, "170", "CASR", "geo", "ITRF14enu_msk.h5"),
    },
    "grd":{
        "diff_SET": os.path.join(data_dir, "170", "CASR", "geo", "diff_SET.grd"),
        "diff_ERA5": os.path.join(data_dir, "170", "CASR", "geo", "diff_ERA5.grd"),
        "diff_demErr": os.path.join(data_dir, "170", "CASR", "geo", "diff_demErr.grd"),
        "diff_ITRF14": os.path.join(data_dir, "170", "CASR", "geo", "diff_ITRF14.grd"),
        "diff_deramp": os.path.join(data_dir, "170", "CASR", "geo", "diff_deramp.grd"),
        "geo_velocity_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_msk.grd"),
        "geo_velocity_SET_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_msk.grd"),
        "geo_velocity_SET_ERA5_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_msk.grd"),
        "geo_velocity_SET_ERA5_demErr_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_msk.grd"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk.grd"),
        "geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp.grd"),
    },
    
    "grd_mm": {
    "diff_SET": os.path.join(data_dir, "170", "CASR", "geo", "diff_SET_mm.grd"),
    "diff_ERA5": os.path.join(data_dir, "170", "CASR", "geo", "diff_ERA5_mm.grd"),
    "diff_demErr": os.path.join(data_dir, "170", "CASR", "geo", "diff_demErr_mm.grd"),
    "diff_ITRF14": os.path.join(data_dir, "170", "CASR", "geo", "diff_ITRF14_mm.grd"),
    "diff_deramp": os.path.join(data_dir, "170", "CASR", "geo", "diff_deramp_mm.grd"),
    "geo_velocity_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_msk_mm.grd"),
    "geo_velocity_SET_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_msk_mm.grd"),
    "geo_velocity_SET_ERA5_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_msk_mm.grd"),
    "geo_velocity_SET_ERA5_demErr_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_msk_mm.grd"),
    "geo_velocity_SET_ERA5_demErr_ITRF14_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_mm.grd"),
    "geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp_mm.grd"),
}, 
    "P208": {
        "geo":              os.path.join(data_dir, "170", "P208", "geo"),
        "geometryRadar":    os.path.join(data_dir, "170", "P208", "inputs", "geometryRadar.h5"),
        "timeseries":       os.path.join(data_dir, "170", "P208", "timeseries_SET_ERA5_ramp_demErr.h5"),
        "velocity":         os.path.join(data_dir, "170", "P208", "velocity.h5"),
        "maskTempCoh":      os.path.join(data_dir, "170", "P208", "maskTempCoh.h5"),
        "geo_timeseries":   os.path.join(data_dir, "170", "P208", "geo", "geo_timeseries_SET_ERA5_ramp_demErr.h5"),
        "geo_velocity":     os.path.join(data_dir, "170", "P208", "geo", "geo_velocity.h5"),
        "geo_maskTempCoh":  os.path.join(data_dir, "170", "P208", "geo", "geo_maskTempCoh.h5"),
        "geo_velocity_msk": os.path.join(data_dir, "170", "P208", "geo", "geo_velocity_msk.h5"),
        "geo_geometryRadar":os.path.join(data_dir, "170", "P208", "geo", "geo_geometryRadar.h5"),
        "vel_grd":          os.path.join(data_dir, "170", "P208", "geo", "geo_velocity_msk.grd"),
    },
    
    "P784": {
        "geo":              os.path.join(data_dir, "170", "P784", "geo"),
        "geometryRadar":    os.path.join(data_dir, "170", "P784", "inputs", "geometryRadar.h5"),
        "timeseries":       os.path.join(data_dir, "170", "P784", "timeseries_SET_ERA5_ramp_demErr.h5"),
        "velocity":         os.path.join(data_dir, "170", "P784", "velocity.h5"),
        "maskTempCoh":      os.path.join(data_dir, "170", "P784", "maskTempCoh.h5"),
        "geo_timeseries":   os.path.join(data_dir, "170", "P784", "geo", "geo_timeseries_SET_ERA5_ramp_demErr.h5"),
        "geo_velocity":     os.path.join(data_dir, "170", "P784", "geo", "geo_velocity.h5"),
        "geo_maskTempCoh":  os.path.join(data_dir, "170", "P784", "geo", "geo_maskTempCoh.h5"),
        "geo_velocity_msk": os.path.join(data_dir, "170", "P784", "geo", "geo_velocity_msk.h5"),
        "geo_geometryRadar":os.path.join(data_dir, "170", "P784", "geo", "geo_geometryRadar.h5"),
        "vel_grd":          os.path.join(data_dir, "170", "P784", "geo", "geo_velocity_msk.grd"),
    },
#     "geysers": {
#         "geo_geometryRadar": os.path.join(data_dir, "170", "CASR", "geo", "geo_geometryRadar_geysers.h5"),
#         "geo_timeseries": os.path.join(data_dir, "170", "CASR","geo", "geo_timeseries_SET_ERA5_demErr_geysers.h5"),
#         "geo_velocity": os.path.join(data_dir, "170", "CASR", "geo", "geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp_geysers.h5"),
#     }
}

decomp = {
    "CASR":{
        "asc_semi": os.path.join(data_dir, "068_170_CASR", "velocity_asc_semi_up.h5"),
        "des_semi": os.path.join(data_dir, "068_170_CASR", "velocity_des_semi_up.h5"),
        "insar_only_up": os.path.join(data_dir, "068_170_CASR", "velocity_insar_only_up.h5"),
        "insar_only_east": os.path.join(data_dir, "068_170_CASR", "velocity_insar_only_east.h5"),
        "gps_insar_up": os.path.join(data_dir, "068_170_CASR", "velocity_gps_insar_up.h5"),
        "gps_insar_east": os.path.join(data_dir, "068_170_CASR", "velocity_gps_insar_east.h5"),
        "gps_insar_north": os.path.join(data_dir, "068_170_CASR", "velocity_gps_insar_north.h5"),
        },
    "grd":{ 
        "asc_semi": os.path.join(data_dir, "068_170_CASR", "velocity_asc_semi_up.grd"),
        "des_semi": os.path.join(data_dir, "068_170_CASR", "velocity_des_semi_up.grd"),
        "insar_only_up": os.path.join(data_dir, "068_170_CASR", "velocity_insar_only_up.grd"),
        "insar_only_east": os.path.join(data_dir, "068_170_CASR", "velocity_insar_only_east.grd"),
        "gps_insar_up": os.path.join(data_dir, "068_170_CASR", "velocity_gps_insar_up.grd"),
        "gps_insar_east": os.path.join(data_dir, "068_170_CASR", "velocity_gps_insar_east.grd"),
        "gps_insar_north": os.path.join(data_dir, "068_170_CASR", "velocity_gps_insar_north.grd"),
        },
    "P784": {
        "insar_only_up":        os.path.join(data_dir, "068_170_P784", "velocity_insar_only_up.h5"),
        "insar_only_east":      os.path.join(data_dir, "068_170_P784", "velocity_insar_only_east.h5"),
        "insar_only_up_grd":    os.path.join(data_dir, "068_170_P784", "velocity_insar_only_up.grd"),
        "insar_only_east_grd":  os.path.join(data_dir, "068_170_P784", "velocity_insar_only_east.grd"),
        "geo":                  os.path.join(data_dir, "170", "P784", "geo", "geo_geometryRadar.h5") 
        }
}

paths_170_5_28 = {
    "CentralValley": {
        "geo_geometryRadar":    os.path.join(data_dir, "170_5_28", "CentralValley", "geo", "geo_geometryRadar.h5"),
        "geo_TempCoh":          os.path.join(data_dir, "170_5_28", "CentralValley", "geo", "geo_temporalCoherence.h5"),
        "geo_maskTempCoh":      os.path.join(data_dir, "170_5_28", "CentralValley", "geo", "geo_maskTempCoh.h5"),
        "geo_velocity":         os.path.join(data_dir, "170_5_28", "CentralValley", "geo", "geo_velocity.h5"),
        "geo_velocity_grd":     os.path.join(data_dir, "170_5_28", "CentralValley", "geo", "geo_velocity.grd"),
        "geo_velocity_msk":     os.path.join(data_dir, "170_5_28", "CentralValley", "geo", "geo_velocity_msk.h5"),
        "geo_velocity_msk_grd": os.path.join(data_dir, "170_5_28", "CentralValley", "geo", "geo_velocity_msk.grd"),
        "geo_timeseries":       os.path.join(data_dir, "170_5_28", "CentralValley", "geo", "geo_timeseries_SET_ERA5_ramp_demErr.h5"),
        "geo_timeseries_msk":   os.path.join(data_dir, "170_5_28", "CentralValley", "geo", "geo_timeseries_SET_ERA5_ramp_demErr_msk.h5"),
        "geo_cumdisp_22": os.path.join(data_dir, "170_5_28", "CentralValley", "geo", "20220110_20221226_cumdisp.grd"),
    },
    "EelRiver": {
        "geo_geometryRadar":    os.path.join(data_dir, "170_5_28", "EelRiver", "geo", "geo_geometryRadar.h5"),
        "geo_TempCoh":      os.path.join(data_dir, "170_5_28", "EelRiver", "geo", "geo_temporalCoherence.h5"),
        "geo_velocity":         os.path.join(data_dir, "170_5_28", "EelRiver", "geo", "geo_velocity.h5"),
        "geo_velocity_msk":     os.path.join(data_dir, "170_5_28", "EelRiver", "geo", "geo_velocity_msk.h5"),
        "geo_velocity_msk_grd": os.path.join(data_dir, "170_5_28", "EelRiver", "geo", "geo_velocity_msk.grd"),
        "geo_velocity_grd":     os.path.join(data_dir, "170_5_28", "EelRiver", "geo", "geo_velocity.grd"),
        "geo_timeseries":       os.path.join(data_dir, "170_5_28", "EelRiver", "geo", "geo_timeseries_tropHgt_demErr.h5"),
        "geo_timeseries_msk":   os.path.join(data_dir, "170_5_28", "EelRiver", "geo", "geo_timeseries_tropHgt_demErr_msk.h5"),
    },

    "Graham": {
        "geo_geometryRadar":    os.path.join(data_dir, "170_5_28", "Graham", "geo", "geo_geometryRadar.h5"),
        "geo_TempCoh":      os.path.join(data_dir, "170_5_28", "Graham", "geo", "geo_temporalCoherence.h5"),
        "geo_velocity":         os.path.join(data_dir, "170_5_28", "Graham", "geo", "geo_velocity.h5"),
        "geo_velocity_msk":     os.path.join(data_dir, "170_5_28", "Graham", "geo", "geo_velocity_msk.h5"),
        "geo_velocity_msk_grd": os.path.join(data_dir, "170_5_28", "Graham", "geo", "geo_velocity_msk.grd"),
        "geo_velocity_grd":     os.path.join(data_dir, "170_5_28", "Graham", "geo", "geo_velocity.grd"),
        "geo_timeseries":       os.path.join(data_dir, "170_5_28", "Graham", "geo", "geo_timeseries_tropHgt_demErr.h5"),
        "geo_timeseries_msk":   os.path.join(data_dir, "170_5_28", "Graham", "geo", "geo_timeseries_tropHgt_demErr_msk.h5"),
    },
}

paths_115 = {
    "CentralValley": {
        "geo_geometryRadar":    os.path.join(data_dir, "115_CentralVal_21_24", "inputs", "geometryGeo.h5"),
        "geo_maskTempCoh":      os.path.join(data_dir, "115_CentralVal_21_24", "maskTempCoh.h5"),
        "geo_TempCoh":          os.path.join(data_dir, "115_CentralVal_21_24", "temporalCoherence.h5"),
        "geo_velocity":         os.path.join(data_dir, "115_CentralVal_21_24", "velocity.h5"),
        "geo_velocity_grd":     os.path.join(data_dir, "115_CentralVal_21_24", "velocity.grd"),
        "geo_velocity_msk":     os.path.join(data_dir, "115_CentralVal_21_24", "velocity_msk.h5"),
        "geo_velocity_msk_grd": os.path.join(data_dir, "115_CentralVal_21_24", "velocity_msk.grd"),
        "geo_timeseries":       os.path.join(data_dir, "115_CentralVal_21_24", "timeseries_SET_ERA5_ramp_demErr.h5"),
        "geo_timeseries_msk":   os.path.join(data_dir, "115_CentralVal_21_24", "timeseries_SET_ERA5_ramp_demErr_msk.h5"),
    }
}