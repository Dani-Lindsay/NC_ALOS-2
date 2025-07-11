#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:33:39 2025

@ Author: daniellelindsay

GPS Processing Pipeline:
1) Download and filter station list by ROI & date window
2) Download daily time series
3) For each station:
   a) Data length check
   b) Manual exclusion → plot to Manual/, skip summary
   c) Noise threshold on Z → plot to Noisy/, skip summary
   d) Auto‐correct all type‐1 steps → mark Corrected
   e) Trend fit & std; summary only Raw or Corrected
   f) Plot with vertical lines, trend lines, legend off to right
"""
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Tuple
import h5py
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

from NC_ALOS2_filepaths import (paths_gps, paths_068, paths_169, paths_170, common_paths)
import insar_utils as utils

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
tol             = 1.0     # years slack for station list
data_threshold  = 0.70    # fraction of days required
noise_threshold = 1.0     # std(Z) > threshold mm → noisy station
STEP_THRESHOLD  = 3.0     # mm jump to correct

ref_frame    = 'IGS14'
IGS14_URL    = 'https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/'
NA_URL       = 'https://geodesy.unr.edu/gps_timeseries/tenv3/plates/NA/'
steps_URL    = 'https://geodesy.unr.edu/NGLStationPages/steps.txt'
holdings_URL = 'https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt'

# ROI
minlat, maxlat = 36.1842293490991, 42.40351528258976
minlon, maxlon = 360-124.62438691953557, 360-119.29408932631216
Range = [minlon, maxlon, minlat, maxlat]

# File locations
DataHoldings_file = paths_gps['DataHoldings']
Steps_file        = paths_gps['Steps']
UNR_dir           = paths_gps['UNRdaily_Dir']
fig_dir           = paths_gps['Fig_Dir']
sta_list_068      = paths_gps['068_StaList']
sta_list_169      = paths_gps['169_StaList']
sta_list_170      = paths_gps['170_StaList']
out_068           = paths_gps['068_enu']
out_169           = paths_gps['169_enu']
out_170           = paths_gps['170_enu']
ref_station       = common_paths['ref_station']

# Manual exclusion list # check 312,
manual_exclude = [
    "ASHL", "DUBP", "EBBS", "EBMD", "FLNT", "INV1", "LRA1", "LRA2", "LRA4",
    "MCCM", "MNRC", "MONB", "OREO", "OXMT", "P136", "P141", "P144", "P144", 
    "P149", "P150", "P170", "P215", "P221", "P222", "P242", "P274", "P299",
    "P312", "P332", "P340", "P348", "P655", "P656", "P658", "P663", "P666", 
    "P668", "P671", "P673", "P674", "P794", "RAPT", "RBRU", "RDFD", "RDGM", 
    "RNO1", "SLID", "TRAN", "YBHB", "P175", "P661", "P670", "P669", "CSJB",
    "P305", "MNDS", "FARB", "P534", "P313", "P336", "P219"
    "P231", "CAMT", "P171" ] # These three on Monterey peninsulae where we have unwrapping errors. 


# Don't fix the equipment step for these stations
skip_equip = [
    "CAP1", "CCSF", "CCSF", "CROW", "CYTE", "FARB", "HCRO", "HOPB", "LKVW",
    "LUTZ", "MCCM", "MODB", "MUSB", "MUSB", "P059", "P096", "P136", "P140",
    "P143", "P155", "P155", "P159", "P164", "P172", "P195", "P195",
    "P207", "P239", "P320", "P322", "P329", "P335", "P186", 
    "P336", "P336", "P341", "P362", "P657", "P668", "P668", "P671", "P671",
    "P672", "P730", "P731", "SACR", "SAOB", "SARG", "SBRB",
]

# Missing steps from steps.txt -- Not complete and some have been fixed and now redunant 
manual_steps = [
    ["CARK", 2022.1437], ["GRDV", 2016.0301], ["GRDV", 2018.956],
    ["P096", 2023.2498], ["P160", 2023.258],  ["P161", 2023.2498], ["P161", 2023.9640],
    ["P165", 2023.2444], ["P166", 2023.4387], ["P167", 2023.4689], ["P168", 2021.3279],
    ["P169", 2023.2498], ["P183", 2023.2498], ["P185", 2023.2498], ["P187", 2023.2444],
    ["P188", 2023.6304], ["P189", 2023.5975], ["P198", 2023.2444], ["P198", 2023.6769],
    ["P198", 2023.8138], ["P198", 2023.9233], ["P199", 2023.2498], ["P199", 2024.0600],
    ["P200", 2023.2444], ["P202", 2023.2444], ["P204", 2023.269],  ["P265", 2023.3511],
    ["P265", 2023.9425], ["P268", 2023.2772], ["P314", 2023.2772], ["P315", 2023.3511],
    ["P317", 2023.2444], ["P321", 2023.2498], ["P332", 2023.2444], ["P349", 2023.3155],
    ["P670", 2023.2608], ["KFRC", 2021.5688], ["P319", 2023.8439], ["P326", 2021.3260],
    ["P329", 2023.4497], ["P335", 2023.4798], ["P336", 2023.4497], ["P322", 2021.3417],
    ["CCSF", 2016.9144], ["DIAB", 2022.5462], ["P175", 2023.2498], ["P177", 2023.2498],
    ["P178", 2023.2444], ["P212", 2023.2444], ["P214", 2023.2444], ["P215", 2023.269],
    ["P216", 2023.2444], ["P217", 2023.2444], ["P220", 2023.2772], ["P212", 2023.2444],
    ["P222", 2023.3511], ["P225", 2023.2444], ["P230", 2023.2772], ["P234", 2023.3812],
    ["P234", 2023.2444], ["P235", 2023.2444], ["P237", 2023.269],  ["P239", 2023.2444],
    ["P242", 2023.2444], ["P242", 2023.3155], ["P242", 2023.9343], ["P247", 2023.6304],
    ["P250", 2023.4579], ["P254", 2023.3046], ["P304", 2023.4196], ["P332", 2023.4935],
    ["P670", 2023.4004], ["P670", 2023.403] ]


# ---------------------------------------------------------------------------
# Helpers: download, date endpoints, station list
# ---------------------------------------------------------------------------
def dload_site_list(url: str, out_file: str) -> None:
    print(f"Downloading {url} → {out_file}")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    urlretrieve(url, out_file)


def get_start_end_dates(ts_file: str) -> Tuple[float,float,datetime,datetime]:
    with h5py.File(ts_file, 'r') as f:
        raw = f['date'][:]
    dt = [pd.to_datetime(d.decode(), utc=False) for d in raw]
    d0, d1 = dt[0], dt[-1]
    return (
        utils.date_to_decimal_year(d0),
        utils.date_to_decimal_year(d1),
        d0, d1
    )


def compute_min_data_len(d0: datetime, d1: datetime, thresh: float) -> int:
    return round((d1 - d0).days * thresh)


def get_station_list(
    holdings_file: str,
    output_file: str,
    start_dec: float,
    end_dec: float,
    geo_range: List[float],
    tol: float
) -> int:
    df = pd.read_csv(
        holdings_file,
        sep=r'\s+', comment='#',
        usecols=[0,1,2,7,8], header=0,
        names=['StaID','lat','lon','dtbeg','dtend']
    )
    df['dtbeg'] = pd.to_datetime(df['dtbeg'], errors='coerce')
    df['dtend'] = pd.to_datetime(df['dtend'], errors='coerce')
    df.dropna(subset=['dtbeg','dtend'], inplace=True)
    df['avail_start'] = df['dtbeg'].apply(utils.date_to_decimal_year)
    df['avail_end']   = df['dtend'].apply(utils.date_to_decimal_year)
    mn_lon, mx_lon, mn_lat, mx_lat = geo_range
    geo_mask = df.lon.between(mn_lon, mx_lon) & df.lat.between(mn_lat, mx_lat)
    time_mask= (
        df['avail_start'] <= start_dec + tol
    ) & (
        df['avail_end'] >= end_dec - tol
    )
    df2 = df.loc[geo_mask & time_mask,
              ['StaID','lat','lon','avail_start','avail_end']]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df2.to_csv(
        output_file, sep=' ', index=False, header=False,
        float_format='%.6f'
    )
    print(f"Wrote {len(df2)} stations → {output_file}")
    return len(df2)


def download_data_with_sta_list(
    data_dir: str,
    sta_list_file: str,
    ref_frame: str
) -> int:
    urls = {'IGS14': IGS14_URL, 'NA': NA_URL}
    ext = 'tenv3' if ref_frame=='IGS14' else f'{ref_frame}.tenv3'
    os.makedirs(data_dir, exist_ok=True)
    cnt = 0
    with open(sta_list_file) as f:
        for L in f:
            sta = L.split()[0]
            outp = os.path.join(data_dir, f"{sta}.{ref_frame}.tenv3")
            if os.path.exists(outp):
                continue
            url = f"{urls[ref_frame]}{sta}.{ext}"
            print(f"Downloading {sta} → {outp}")
            subprocess.run(['wget','-O',outp,url], check=True)
            cnt += 1
    print(f"Downloaded {cnt} new → {data_dir}")
    return cnt

# ---------------------------------------------------------------------------
# Processing & correction helpers
# ---------------------------------------------------------------------------
def parse_steps_file(
    file_path: str,
    sta_id: str
) -> Tuple[List[float], List[float]]:
    month_map = {m: i for i, m in enumerate(
        ['JAN','FEB','MAR','APR','MAY','JUN',
         'JUL','AUG','SEP','OCT','NOV','DEC'], 1
    )}
    equip, quake = [], []
    for ln in open(file_path):
        p = ln.split()
        if p[0] != sta_id:
            continue
        yy = int(p[1][:2]) + 2000
        mm = month_map[p[1][2:5]]
        dd = int(p[1][5:])
        dt = datetime(yy, mm, dd)
        decy = dt.year + (dt.timetuple().tm_yday - 1)/365.25
        if p[2] == '1': equip.append(decy)
        if p[2] == '2': quake.append(decy)
    return equip, quake


def find_step_indices(
    years: np.ndarray,
    step: float
) -> Tuple[int, int]:
    i = np.argmin(np.abs(years - step))
    return i - 1, i


def estimate_and_correct_offset(
    years: np.ndarray,
    E: np.ndarray, N: np.ndarray, Z: np.ndarray,
    window: Tuple[int, int], threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], List[float]]:

    i0, i1 = window
    offE = np.nanmean(E[i1:i1+10]) - np.nanmean(E[i0-10:i0])
    offN = np.nanmean(N[i1:i1+10]) - np.nanmean(N[i0-10:i0])
    offZ = np.nanmean(Z[i1:i1+10]) - np.nanmean(Z[i0-10:i0])
    offsets = [offE, offN, offZ]

    steps: List[float] = []
    if abs(offE) > threshold or abs(offN) > threshold or abs(offZ) > threshold:
        steps.append(years[i1])
        E[i1:] -= offE
        N[i1:] -= offN
        Z[i1:] -= offZ

    return E, N, Z, steps, offsets


def process_station_data(
    file_path: str,
    sta_id: str,
    t0: float,
    te: float,
    min_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yrs, Es, Ns, Zs = [], [], [], []
    with open(file_path) as f:
        for ln in f:
            p = ln.split()
            if p[0] != sta_id:
                continue
            y = float(p[2])
            if not (t0 <= y <= te):
                continue
            yrs.append(y)
            Es.append(float(p[8]))
            Ns.append(float(p[10]))
            Zs.append(float(p[12]))
    if len(Es) < min_len:
        return None, None, None, None
    years = np.array(yrs)
    E = (np.array(Es) - Es[0]) * 1000
    N = (np.array(Ns) - Ns[0]) * 1000
    Z = (np.array(Zs) - Zs[0]) * 1000
    return years, E, N, Z


def compute_trend(
    years: np.ndarray,
    vals: np.ndarray,
    t0: float
) -> Tuple[float, float, float]:
    s, _, r, p, st = stats.linregress(years - t0, vals)
    return s, st, r

# ---------------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------------
def plot_timeseries(
    years: np.ndarray,
    E: np.ndarray, N: np.ndarray, Z: np.ndarray,
    equip: List[float], quake: List[float], corr: List[float],
    sta_id: str, fig_dir: str, t0: float, te: float,
    track_label: str,
    E_ori: np.ndarray, N_ori: np.ndarray, Z_ori: np.ndarray,
    track: str,
):
    paths_map = {
        'Raw': os.path.join(fig_dir, 'Uncorrected'),
        'Corrected': os.path.join(fig_dir, 'Corrected'),
        'Noisy': os.path.join(fig_dir, 'Noisy'),
        'Manual': os.path.join(fig_dir, 'Manual_Exclude')
    }
    for p in paths_map.values():
        os.makedirs(p, exist_ok=True)
    # Determine folder directly from track_label
    folder = track_label if track_label in paths_map else 'Raw'
    
    
    # Re-zero series so first point is zero
    E = E - E[0]
    N = N - N[0]
    Z = Z - Z[0]
    
    E_ori = E_ori - E_ori[0]
    N_ori = N_ori - N_ori[0]
    Z_ori = Z_ori - Z_ori[0]
    # Compute trends & std for annotation
    Ve, se, Re = compute_trend(years, E, t0)
    Vn, sn, Rn = compute_trend(years, N, t0)
    Vz, sz, Rz = compute_trend(years, Z, t0)
    plt.figure(figsize=(8,4))
    # Plot equipment and quake steps
    for idx, s in enumerate(equip):
        label = '1: Instrument' if idx == 0 else None
        plt.axvline(s, color='yellow', lw=2, label=label)
    for idx, s in enumerate(quake):
        label = '2: Earthquake' if idx == 0 else None
        plt.axvline(s, color='lightblue', lw=2, label=label)
    # Plot corrected steps 
    if corr:
        for idx, s in enumerate(corr):
            label = 'Fixed step' if idx == 0 else None
            plt.axvline(s, color='red', ls='--', lw=2, label=label)
            
    # Plot displacements
    plt.plot(years, Z_ori, 'o', ms=1, alpha=0.2,  color='gray')
    plt.plot(years, N_ori, 'o', ms=1, alpha=0.2,  color='gray')
    plt.plot(years, E_ori, 'o', ms=1, alpha=0.2,  color='gray')
    
    plt.plot(years, Z, 'o', ms=1, alpha=0.5, label='Z', color='tab:blue')
    plt.plot(years, N, 'o', ms=1, alpha=0.5, label='N', color='tab:orange')
    plt.plot(years, E, 'o', ms=1, alpha=0.5, label='E', color='tab:green')
    # Trend lines
    xs = np.array([t0, te])
    plt.plot(xs, Ve*(xs - t0), '--', label=f'E {Ve:.2f}±{se:.2f} mm/yr', color='tab:green')
    plt.plot(xs, Vn*(xs - t0), '--', label=f'N {Vn:.2f}±{sn:.2f} mm/yr', color='tab:orange')
    plt.plot(xs, Vz*(xs - t0), '--', label=f'Z {Vz:.2f}±{sz:.2f} mm/yr', color='tab:blue')

    plt.xlabel('Year')
    plt.ylabel('Displacement (mm)')
    plt.title(f'Station {sta_id}, Track {track} ({folder})')
    plt.xlim(t0, te)
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    outp = os.path.join(paths_map[folder], f"{track}_{sta_id}_{folder}.png")
    plt.savefig(outp)
    plt.close()

# ---------------------------------------------------------------------------
# Orchestration per track
# ---------------------------------------------------------------------------
def run_path(
    track: str,
    sta_list_file: str,
    data_dir: str,
    fig_dir: str,
    steps_file: str,
    t0: float,
    te: float,
    min_len: int,
    exclude_ids: List[str]
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    df = pd.read_csv(
        sta_list_file,
        sep=' ',
        header=None,
        usecols=[0, 1, 2],
        names=['StaID', 'lat', 'lon']
    )
    for _, row in df.iterrows():
        # Set Station ID, lat, lon
        sid, lat, lon = row['StaID'], row['lat'], row['lon']
        # Initialize status
        status_label = 'Raw'
        # Read data
        yrs, E_raw, N_raw, Z_raw = process_station_data(
        os.path.join(data_dir, f"{sid}.{ref_frame}.tenv3"),
        sid, t0, te, min_len)
        if yrs is None:
            continue
            
        # Make copies: one set for plotting the raw/uncorrected, another for correction
        E_ori = E_raw.copy()
        N_ori = N_raw.copy()
        Z_ori = Z_raw.copy()
        
        # Work on separate arrays for correction & trend computation
        E = E_raw.copy()
        N = N_raw.copy()
        Z = Z_raw.copy()
        
        # Get steps
        equip_all, quake = parse_steps_file(steps_file, sid)
        print(f"{sid}: raw quake steps →", quake)
        # Filter steps for those between start and end dates
        equip = [x for x in equip_all if t0 < x < te]
        
        corr_steps: List[float] = []

        # Manual overrides
        for ms_id, ms_year in manual_steps:
            if sid == ms_id:
                w0, w1 = find_step_indices(yrs, ms_year)
                E, N, Z, c, offsets = estimate_and_correct_offset(
                    yrs, E, N, Z, (w0, w1), STEP_THRESHOLD
                )
                if c:
                    corr_steps.extend(c)
                    print(f"{sid}: manual correction detected at {c}, offsets={offsets}")
    
        # Auto-correct steps
        if sid not in skip_equip:
            for st in equip:
                w0, w1 = find_step_indices(yrs, st)
                E, N, Z, c, offsets = estimate_and_correct_offset(
                    yrs, E, N, Z, (w0, w1), STEP_THRESHOLD
                )
                if c:
                    corr_steps.extend(c)
                    print(f"{sid}: auto correction detected at {c}, offsets={offsets}")
    
        # Optionally dedupe & sort:
        corr_steps = sorted(set(corr_steps))
        print(f"{sid}: final corr_steps = {corr_steps}")
                        
        # If corr_steps is not empty then set label to "Corrected" 
        if corr_steps and status_label == 'Raw':
            print(f"{sid}: label switched to Corrected")
            status_label = 'Corrected'       
                    
        # Compute trends & std
        Ve, se, Re = compute_trend(yrs, E, t0)
        Vn, sn, Rn = compute_trend(yrs, N, t0)
        Vz, sz, Rz = compute_trend(yrs, Z, t0)
        # Noise exclusion
        if sz > noise_threshold:
            print(f"{sid}: noisy (Z std={sz:.2f} mm), plotting to Noisy")
            status_label = 'Noisy'
        # Manual exclusion
        if sid in manual_exclude:
            print(f"{sid}: manual exclude, plotting to Manual")
            status_label = 'Manual'
        # Summary only for Raw or Corrected
        if status_label in ['Raw', 'Corrected']:
            summary.append({
                'StaID': sid,
                'Lon': lon,
                'Lat': lat,
                'Ve': Ve,
                'Vn': Vn,
                'Vz': Vz,
                'Std_e': se,
                'Std_n': sn,
                'Std_z': sz,
                'r_e': Re,
                'r_n': Rn,
                'r_z': Rz
            })
        # Plot
        plot_timeseries(
            yrs, E, N, Z, equip, quake,
            corr_steps, sid, fig_dir, t0, te, status_label,
            E_ori, N_ori, Z_ori, track)
    return summary

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Update UNR dataholdings and steps.txt files
    dload_site_list(holdings_URL, DataHoldings_file)
    dload_site_list(steps_URL, Steps_file)
    
    # Get start and end dates directly from the InSAR tracks
    s68, e68, d068, d168 = get_start_end_dates(paths_068[ref_station]['timeseries'])
    s169, e169, d169, d269 = get_start_end_dates(paths_169[ref_station]['timeseries'])
    s170, e170, d170, d370 = get_start_end_dates(paths_170[ref_station]['timeseries'])
    
    # Calculate minimum data length based on start and end dates
    ml68  = compute_min_data_len(d068, d168, data_threshold)
    ml169 = compute_min_data_len(d169, d269, data_threshold)
    ml170 = compute_min_data_len(d170, d370, data_threshold)
    
    # Get list of stations with start/end dates within +/-1 year of InSAR tracks and in ROI
    get_station_list(DataHoldings_file, sta_list_068, s68, e68, Range, tol)
    get_station_list(DataHoldings_file, sta_list_169, s169, e169, Range, tol)
    get_station_list(DataHoldings_file, sta_list_170, s170, e170, Range, tol)
    
    # Download data in reference frames
    download_data_with_sta_list(UNR_dir, sta_list_068, ref_frame)
    download_data_with_sta_list(UNR_dir, sta_list_169, ref_frame)
    download_data_with_sta_list(UNR_dir, sta_list_170, ref_frame)
    
    # Process each track
    df68 = pd.DataFrame(run_path('068', sta_list_068, UNR_dir, fig_dir, Steps_file, s68, e68, ml68, manual_exclude))
    df169 = pd.DataFrame(run_path('169', sta_list_169, UNR_dir, fig_dir, Steps_file, s169, e169, ml169, manual_exclude))
    df170 = pd.DataFrame(run_path('170', sta_list_170, UNR_dir, fig_dir, Steps_file, s170, e170, ml170, manual_exclude))
    
    # Write out csv
    df68.to_csv(out_068, index=False)
    df169.to_csv(out_169, index=False)
    df170.to_csv(out_170, index=False)
    
    print('Processing complete!')
