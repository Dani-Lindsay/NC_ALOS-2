#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:07:48 2024

Fault arrows
https://forum.generic-mapping-tools.org/t/plotting-geology-fault-problem-pen-attribute-solid-line-style-overide-from-segment-header/3591

@author: daniellelindsay
"""

import pygmt
import pandas as pd
from NC_ALOS2_filepaths import (common_paths, paths_068, paths_169, paths_170)
import insar_utils as utils

###############################
# Read baselines and burst
###############################

offset = 0

### ******* 068 *******
# Call the function with the path to the uploaded file
ref_date = utils.yymmdd_to_decimal_year("200929")
sec_date, centered_perp_base = utils.read_baselines(paths_068["baselines"])
pairs, mean, btemp, bperp = utils.read_coherence_data(paths_068["CASR"]["coherence"])

dic_068 = {
    "ref_date": ref_date, 
    "sec_date": sec_date,
    "centered_perp_base": centered_perp_base,
    "pairs": pairs,
    "mean": mean,
    "btemp": btemp,
    "bperp": bperp,
    #"burst": burst
}
# Calculate pairs_bperp using the function
dic_068["pairs_bperp"] = utils.calculate_pairs_bperp(dic_068)

### ******* 169 *******
ref_date = utils.yymmdd_to_decimal_year("150408")
sec_date, centered_perp_base = utils.read_baselines(paths_169["baselines"])
pairs, mean, btemp, bperp = utils.read_coherence_data(paths_169["CASR"]["coherence"])

dic_169 = {
    "ref_date": ref_date, 
    "sec_date": sec_date,
    "centered_perp_base": centered_perp_base,
    "pairs": pairs,
    "mean": mean,
    "btemp": btemp,
    "bperp": bperp,
    #"burst": burst
}

# Calculate pairs_bperp using the function
dic_169["pairs_bperp"] = utils.calculate_pairs_bperp(dic_169)

### ******* 170 *******
ref_date = utils.yymmdd_to_decimal_year("150511")
sec_date, centered_perp_base = utils.read_baselines(paths_170["baselines"])
pairs, mean, btemp, bperp = utils.read_coherence_data(paths_170["CASR"]["coherence"])

dic_170 = {
    "ref_date": ref_date, 
    "sec_date": sec_date,
    "centered_perp_base": centered_perp_base,
    "pairs": pairs,
    "mean": mean,
    "btemp": btemp,
    "bperp": bperp,
    #"burst": burst
}

# Calculate pairs_bperp using the function
dic_170["pairs_bperp"] = utils.calculate_pairs_bperp(dic_170)

# Define region of interest 
min_lon=-126.5
max_lon=-119.0
min_lat=36.0
max_lat=42.5

region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
fig_size = "M12c"

grid = pygmt.datasets.load_earth_relief(region=region, resolution="15s")
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30], region=region)

### Begin plotting ###
fig = pygmt.Figure()
pygmt.config(FONT=10, FONT_TITLE=11, MAP_HEADING_OFFSET=0.1, PS_MEDIA="A3", FORMAT_GEO_MAP="ddd", MAP_FRAME_TYPE="plain",)

# Plot DEM
fig.grdimage(grid=grid, projection=fig_size, frame=["lbrt", "xa", "ya"], cmap='wiki-france.cpt', shading=dgrid, region=region, transparency=40)
fig.coast(shorelines=True,lakes=False, borders="2/thin")

# Plot Faults
# Loop through and plot each fault file
# Plot Faults
for fault_file in common_paths["fault_files"]:
    fig.plot(data=fault_file, pen="0.5p,black", transparency=50)
    
fig.plot(data=common_paths['pb_file'] , pen="1.5p,red3", style="f-1c/0.5c+r+s+p1.5p,red3,solid")
fig.plot(data=common_paths['pb2_file'] , pen="1.5p,red3", style="f0.5c/0.15c+r+t", fill="red3")
fig.plot(data=common_paths['pb3_file'] , pen="1.5p,red3", style="f-1c/0.5c+r+s+p1.5p,red3,solid")

#fig.plot(data="/Users/daniellelindsay/Figures/inputdata/wcSlides.gmt", pen="0.3p,purple", fill="purple")

# Label Faults
fig.plot(x=-125.1, y=41.7, style="e280/1.5/0.7", fill="white", transparency=30)
fig.text(text="CSZ", x=-125.1, y=41.7, justify="CM", font="12p,red3" , angle=280)

fig.plot(x=-125.47, y=40.05, style="e0/1.5/0.7", fill="white", transparency=30)
fig.text(text="MFZ", x=-125.47, y=40.05, justify="CM", font="12p,red3" )

fig.plot(x=-123.0, y=37.65, style="e310/1.5/0.7", fill="white", transparency=30)
fig.text(text="SAF", x=-123.0, y=37.65, justify="CM", font="12p,red3" , angle=310)

df = pd.DataFrame(
    data={
        "x": [-125.7, -125.0],
        "y": [41.0, 39.0],
        "east_velocity": [17.20, -28.59],
        "north_velocity": [23.98, 43.59],
        "east_sigma": [0, 0],
        "north_sigma": [0, 0],
        "correlation_EN": [0, 0],
        "SITE": ["", ""],
        })

fig.velo(data=df, pen="1p,black", uncertaintyfill="lightblue1", line=True, spec="e0.035/0.39/18", vector="0.3c+p1p+e+gblack",)
fig.text(text=["JF(NA)","PA(NA)"], x=df.x, y=df.y, justify="TC",offset ="0/-0.2c",  font="12p,black")
 
fig.text(text="Central Valley", x=-120.3, y=36.9, justify="CM", font="12p,black" , angle=310)
fig.text(text="Sierra Nevada", x=-120.2, y=38.5, justify="CM", font="12p,black" , angle=310)
fig.text(text="Coastal R.", x=-123.2, y=39.5, justify="CM", font="9p,black" , angle=300)

sha_lat, sha_lon = 41.3099, -122.3106 # Mt Shasta
las_lat, las_lon = 40.492,  -121.508 # Lassen Volano
med_lat, med_lon = 41.6108, -121.5535 # Medicine Lake Volcano
gey_lat, gey_lon = 38.84, -122.83 #Geysers

fig.plot(x=sha_lon, y=sha_lat, style="kvolcano/0.4c", pen="1p,black", fill="darkred")
fig.plot(x=las_lon, y=las_lat, style="kvolcano/0.4c", pen="1p,black", fill="darkred")
fig.plot(x=med_lon, y=med_lat, style="kvolcano/0.4c", pen="1p,black", fill="darkred")
fig.plot(x=gey_lon, y=gey_lat, style="kvolcano/0.4c", pen="1p,black", fill="darkred")

fig.text(text="Mt. Shasta",     x=sha_lon, y=sha_lat, justify="BL", offset="0.2c/-0.2c", font="9p,gray15" )
fig.text(text="Medicine Lake",  x=med_lon, y=med_lat, justify="BL", offset="0.2c/-0.2c", font="9p,gray15" )
fig.text(text="Lassen",         x=las_lon, y=las_lat, justify="BL", offset="0.2c/-0.2c", font="9p,gray15" )
fig.text(text="Geysers",         x=gey_lon, y=gey_lat, justify="BL", offset="0.2c/-0.2c", font="9p,gray15" )

hum_lat, hum_lon = 40.8021, -124.1637 # Humboldt Bay 40.8021° N, 124.1637
sar_lat, sar_lon = 38.4404, -122.7141 # Santa Rosa
sfo_lat, sfo_lon = 37.7749, -122.4194 # San Francisco
cre_lat, cre_lon = 41.7558, -124.2026 # Cresent City

fig.plot(x=hum_lon, y=hum_lat, style="kcircle/0.18c", pen="1p,black", fill="dimgray")
fig.plot(x=sar_lon, y=sar_lat, style="kcircle/0.18c", pen="1p,black", fill="dimgray")
fig.plot(x=sfo_lon, y=sfo_lat, style="kcircle/0.18c", pen="1p,black", fill="dimgray")
fig.plot(x=cre_lon, y=cre_lat, style="kcircle/0.18c", pen="1p,black", fill="dimgray")

fig.text(text="Eureka",         x=hum_lon, y=hum_lat, justify="BL", offset="0.2c/0.2c", font="9p,gray15" )
fig.text(text="Santa Rosa",     x=sar_lon, y=sar_lat, justify="BL", offset="0.2c/0c", font="9p,gray15" )
fig.text(text="San Francisco",  x=sfo_lon, y=sfo_lat, justify="BL", offset="0.2c/0c", font="9p,gray15" )
fig.text(text="Crescent City",  x=cre_lon, y=cre_lat, justify="BL", offset="0.2c/0c", font="9p,gray15" )

fig.text(text="a)", position="TL", justify="TL", offset="0.1c/-0.1c")

#Subplot of central valley validation site. 
val_min_lon=-122.35
val_max_lon=-121.8
val_min_lat=38.85
val_max_lat=39.26

fig.plot(x = [val_min_lon, val_min_lon, val_max_lon, val_max_lon, val_min_lon], 
         y = [val_min_lat, val_max_lat, val_max_lat, val_min_lat, val_min_lat], 
         pen="0.8p,black,--", transparency=0)
fig.text(text="Fig. 6&7", x=val_max_lon, y=val_min_lat, justify="BL", offset="0.1c/0.1c", font="8p,gray15" , fill="white", transparency=50)
fig.text(text="Fig. 6&7", x=val_max_lon, y=val_min_lat, justify="BL", offset="0.1c/0.1c", font="8p,gray15" )


# Subplot region for MLV
mlv_min_lon = -122.5
mlv_max_lon = -121.2
mlv_min_lat = 41.14
mlv_max_lat = 41.90
fig.plot(x = [mlv_min_lon, mlv_min_lon, mlv_max_lon, mlv_max_lon, mlv_min_lon], 
         y = [mlv_min_lat, mlv_max_lat, mlv_max_lat, mlv_min_lat, mlv_min_lat], 
         pen="0.8p,black,--", transparency=0)
fig.text(text="Fig. 8", x=mlv_min_lon, y=mlv_min_lat, justify="TL", offset="0.1c/-0.1c", font="8p,gray15" , fill="white", transparency=50 )
fig.text(text="Fig. 8", x=mlv_min_lon, y=mlv_min_lat, justify="TL", offset="0.1c/-0.1c", font="8p,gray15" )

# Subplot region Geysers
gey_min_lon = -123.15
gey_max_lon = -122.4
gey_min_lat = 38.58-0.03
gey_max_lat = 38.98+0.03
fig.plot(x = [gey_min_lon, gey_min_lon, gey_max_lon, gey_max_lon, gey_min_lon], 
         y = [gey_min_lat, gey_max_lat, gey_max_lat, gey_min_lat, gey_min_lat], 
         pen="0.8p,black,--", transparency=0)
fig.text(text="Fig. 9", x=gey_min_lon, y=gey_max_lat, justify="BL", offset="0.1c/0.1c", font="8p,gray15", fill="white", transparency=50  )
fig.text(text="Fig. 9", x=gey_min_lon, y=gey_max_lat, justify="BL", offset="0.1c/0.1c", font="8p,gray15" )

# Subplot region for Central Valley 
cv_min_lon=-122.4
cv_max_lon=-121.8
cv_min_lat=38.8
cv_max_lat=39.8
fig.plot(x = [cv_min_lon, cv_min_lon, cv_max_lon, cv_max_lon, cv_min_lon], 
         y = [cv_min_lat, cv_max_lat, cv_max_lat, cv_min_lat, cv_min_lat], 
         pen="0.8p,black,--", transparency=0)
fig.text(text="Fig. 10", x=cv_min_lon, y=cv_max_lat, justify="BL", offset="0.1c/0.1c", font="8p,gray15", fill="white", transparency=50 )
fig.text(text="Fig. 10", x=cv_min_lon, y=cv_max_lat, justify="BL", offset="0.1c/0.1c", font="8p,gray15" )


# Subplot region Landslides 
ls_min_lon=-123.95
ls_max_lon=-123.7
ls_min_lat=40.6
ls_max_lat=40.8
fig.plot(x = [ls_min_lon, ls_min_lon, ls_max_lon, ls_max_lon, ls_min_lon], 
         y = [ls_min_lat, ls_max_lat, ls_max_lat, ls_min_lat, ls_min_lat], 
         pen="0.8p,black,--", transparency=0)
fig.text(text="Fig. 11a", x=ls_min_lon, y=ls_min_lat, justify="TL", offset="0.1c/-0.1c", font="8p,gray15", fill="white", transparency=50 )
fig.text(text="Fig. 11a", x=ls_min_lon, y=ls_min_lat, justify="TL", offset="0.1c/-0.1c", font="8p,gray15" )


ls2_min_lon=-123.52859
ls2_max_lon=-123.42859
ls2_min_lat=40.02566
ls2_max_lat=40.10566

fig.plot(x = [ls2_min_lon, ls2_min_lon, ls2_max_lon, ls2_max_lon, ls2_min_lon], 
         y = [ls2_min_lat, ls2_max_lat, ls2_max_lat, ls2_min_lat, ls2_min_lat], 
         pen="0.8p,black,--", transparency=0)
fig.text(text="Fig. 11e", x=ls2_max_lon, y=ls2_min_lat, justify="BL", offset="0.1c/0.1c", font="8p,gray15" , fill="white", transparency=50)
fig.text(text="Fig. 11e", x=ls2_max_lon, y=ls2_min_lat, justify="BL", offset="0.1c/0.1c", font="8p,gray15" )



# Inset map of frames
fig.basemap(projection="M4.5c", frame=["lbrt"], region=[region])
fig.coast(shorelines=True, land="lightgray", water="white",  borders="2/0.5p,gray15")
pygmt.makecpt(cmap="inferno", series=[0, 3, 1])
fig.plot(data=common_paths['frames']['170_2800'] , pen="1p,purple4", transparency=30, label='Track 170')
fig.plot(data=common_paths['frames']['170_2850'] , pen="1p,purple4", transparency=30)
fig.text(text="T170", x=-126.341, y=36.767, justify="BL", offset="0.15c/0.15c", font="10p,purple4", angle=350 )
fig.plot(data=common_paths['frames']['169_2800'] , pen="1p,deeppink3", transparency=30, label='Track 169')
fig.plot(data=common_paths['frames']['169_2850'] , pen="1p,deeppink3", transparency=30)
fig.text(text="T169", x=-124.598, y=36.749, justify="BL", offset="0.15c/0.15c", font="10p,deeppink3", angle=350 )
fig.plot(data=common_paths['frames']['068_0800'] , pen="1p,darkorange", transparency=30, label='Track 068')
fig.plot(data=common_paths['frames']['068_0750'] , pen="1p,darkorange", transparency=30)
fig.text(text="T068", x=-124.571, y=41.843, justify="TL", offset="0.15c/-0.15c", font="10p,darkorange", angle=10 )

#fig.legend(box="+p1p+gwhite+c5p", position="jBL+o0.5c/5.5c", projection="M12c")  
fig.basemap(frame=["WSrt", "xa", "ya"], map_scale="jTR+w100k+o0.4/0.4c", projection=fig_size)

fig.shift_origin(xshift="12.5c")

###############################
# Plot network and burst
###############################

pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=9, FONT_TITLE=11, FONT_SUBTITLE = 11, MAP_TITLE_OFFSET= "7p")


size = "X?" #"X4.3/3c"
fig_region = [2014.75, 2024.5, -500, 500]
coh_btemp_region = [-50, 2000, -20, 600]

with fig.subplot(nrows=3, ncols=1, figsize=("5c", "13.4c"), autolabel="b)", margins=["0.1c", "0.6c"], sharex="b", sharey="l", frame=["Wsrt", 'ya', 'xa']):
    pygmt.makecpt(cmap="plasma", series=[0.15, 0.75, 0.1])
    
    ###### Des 169. 
    fig.basemap(region=fig_region, projection=size, frame=["ya+lBperp (m)", "lStE+tDescending 169", "xa"], panel=True)   
    
    for i in range(len(dic_169["pairs"])):  
        fig.plot(x=dic_169["pairs"][i], y=dic_169["pairs_bperp"][i], pen="0.4p,black", projection=size, transparency=80) # ****** , cmap=True, zvalue=dic_169["mean"][i],
    
    fig.plot(x=dic_169["ref_date"],y=0, style="c.1c", fill="black", pen="0.1p", projection=size) # ******
    fig.plot(x=dic_169["sec_date"],y=dic_169["centered_perp_base"], style="c.1c", fill="black", projection=size) # ******
    
    ###### Des 170. 
    fig.basemap(region=fig_region, projection=size, frame=["ya+lBperp (m)", "lStE+tDescending 170", "xa"], panel=True)   
    
    for i in range(len(dic_170["pairs"])):  
        fig.plot(x=dic_170["pairs"][i], y=dic_170["pairs_bperp"][i], pen="0.4p,black", projection=size, transparency = 80) # ******cmap=True, zvalue=dic_170["mean"][i],
        
    fig.plot(x=dic_170["ref_date"],y=0, style="c.1c", fill="black", pen="0.1p", projection=size) # ******
    fig.plot(x=dic_170["sec_date"],y=dic_170["centered_perp_base"], style="c.1c", fill="black", projection=size) # ******
    
    ##### Asc 068
    # Network
    fig.basemap(region=fig_region, projection=size, frame=["ya+lBperp (m)", "lStE+tAscending 068","xa"], panel=True)
    
    for i in range(len(dic_068["pairs"])):  
        fig.plot(x=dic_068["pairs"][i], y=dic_068["pairs_bperp"][i], pen="0.4p,black", projection=size, transparency=80) # ****** cmap=True, zvalue=dic_068["mean"][i], 
    
    fig.plot(x=dic_068["ref_date"],y=0, style="c.1c", fill="black", pen="0.1p", projection=size) # ******
    fig.plot(x=dic_068["sec_date"],y=dic_068["centered_perp_base"], style="c.1c", fill="black", projection=size) # ******

fig.savefig(common_paths['fig_dir']+"Fig_1_IntroMap.png", crop=True, anti_alias=True, show=False)
fig.savefig(common_paths['fig_dir']+"Fig_1_IntroMap.pdf", crop=True, anti_alias=True, show=False)

fig.show()


