### Background ### 
- The National Solar Radiation Data Base (NSRDB) stores many publically available weather datasets that are used as a foundation for today's green energy projects, including NREL's software 
- NSRDB data sets include:
    - weather data for a **"typical meteorological year" (TMY)** in select locations
    - raw weather data from 1998-2021
    - data points such as: temperature, Global Horizontal Irradiance (GHI), Direct Normal Illuminance (DNI), snow, humidity, wind point, etc. in each file 

### Context ### 
Any analysis done on a green energy product that utilizes NREL software and therefore NSRDB data must analyze the TMY datasets. However, the raw data behind NSRDB's TMY datasets is in a data archive that is difficult to read. Furthermore, there is an entire statistical process that transforms this raw data into the TMY file. If data scientists can understand and replicate this process, they can understand their projects deeply and find potential areas to improve the project.

## Sources ## 
Research papers behind TMY files: 
- "Quantifying Interannual Variability for Photovoltaic Systems in PVWatts" by David Severin Ryberg, Janine Freeman, and Nate Blair, National Renewable Energy Laboratory
- "P50/P90 Analysis for Solar Energy Systems Using the System Advisor Model" by A. Dobos and P. Gilman, National Renewable Energy Laboratory, M. Kasberg, Pariveda Solutions
