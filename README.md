# NC_ALOS-2
# 3D Deformation in Northern California with ALOS-2

This repository contains all of the scripts and post-processing tools used **'Nine-Year L-band InSAR Time Series of Tectonic and Non-tectonic Surface Deformation in Northern California'**. It is organized to generate figures, perform geospatial and time-series analyses, and reproduce all results in the article.

## Overview

Co-registered stacks of interferograms were produced with the ALOS-2 ScanSAR stack processing tools in isce-2 
https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/contrib/stack/alosStack/alosStack_tutorial.txt

This codebase supports the post-processing and analysis of ALOS-2 ScanSAR data to produce 3D surface deformation maps, time-series plots, and statistical summaries for Northern California. It includes:

- **Preparing GNSS velocities** from UNR daily solutions
- **Prepares files for analysis** 
- **3D Decompositions** 
- **InSAR Validation** between GNSS, Sentinel-1 and ALOS-2 results  
- **Time-series extraction** for landslides and Central Valley
- **Visualization** with PyGMT 


## Thesis Context

The work demonstrates how L-band ScanSAR can reveal multi-scale tectonic and non-tectonic deformation across Northern California. Key applications include:

1. Identifying transient deformation events (landslides, aquifer changes)   
2. Mapping long‐wavelength regional uplift/subsidence 
3. Validating InSAR velocities against continuous GNSS stations and Sentinel-1 time series

