ID -> File Path Mapping (From context/data_local)

Purpose
- Provide a dataset-by-dataset entity mapping derived from context/data_local JSONs.
- Preserve entity and variable integrity (no deduplication or merging).
- Document when filenames are not specified in the JSONs.

Rules Used
- If id_info.json exists: entity IDs are top-level keys in that file.
- If id_info.json does not exist: entity IDs are keys under static_info.json.channel_info.
- File path is only listed when explicitly specified in the JSONs.
- Variable lists come from static_info.json.channel_info when available.

Dataset: Bear_room
Source:
- context/data_local/Bear_room/time_series/id_info.json
- context/data_local/Bear_room/static_info.json
Time series path resolution (data config)
- Data config used: data_configs/Bear_room/fullBear_hetero_TGTSF.yaml
- root_path: ./data/Bear_room/time_series
- formatter: {i}.parquet
- id selection in config: explicit subset list (see config file)
- Resolved rule: file path = {root_path}/{id}.parquet for each selected id

Entity -> file path
- 104 -> context/data_local/Bear_room/time_series/104.parquet
- 105 -> context/data_local/Bear_room/time_series/105.parquet
- 107 -> context/data_local/Bear_room/time_series/107.parquet
- 108 -> context/data_local/Bear_room/time_series/108.parquet
- 110 -> context/data_local/Bear_room/time_series/110.parquet
- 114 -> context/data_local/Bear_room/time_series/114.parquet
- 120 -> context/data_local/Bear_room/time_series/120.parquet
- 121 -> context/data_local/Bear_room/time_series/121.parquet
- 122 -> context/data_local/Bear_room/time_series/122.parquet
- 208 -> context/data_local/Bear_room/time_series/208.parquet
- 213 -> context/data_local/Bear_room/time_series/213.parquet
- 216 -> context/data_local/Bear_room/time_series/216.parquet
- 217 -> context/data_local/Bear_room/time_series/217.parquet
- 221 -> context/data_local/Bear_room/time_series/221.parquet
- 223 -> context/data_local/Bear_room/time_series/223.parquet
- 227 -> context/data_local/Bear_room/time_series/227.parquet
- 229 -> context/data_local/Bear_room/time_series/229.parquet
- 240 -> context/data_local/Bear_room/time_series/240.parquet
- 245 -> context/data_local/Bear_room/time_series/245.parquet
- 247 -> context/data_local/Bear_room/time_series/247.parquet
- 248 -> context/data_local/Bear_room/time_series/248.parquet
- 249 -> context/data_local/Bear_room/time_series/249.parquet
- 252 -> context/data_local/Bear_room/time_series/252.parquet
- 254 -> context/data_local/Bear_room/time_series/254.parquet
- 261 -> context/data_local/Bear_room/time_series/261.parquet
- 264 -> context/data_local/Bear_room/time_series/264.parquet
- 268 -> context/data_local/Bear_room/time_series/268.parquet
- 280 -> context/data_local/Bear_room/time_series/280.parquet
- 281 -> context/data_local/Bear_room/time_series/281.parquet
- 288 -> context/data_local/Bear_room/time_series/288.parquet
- 290 -> context/data_local/Bear_room/time_series/290.parquet
- 302 -> context/data_local/Bear_room/time_series/302.parquet
- 304 -> context/data_local/Bear_room/time_series/304.parquet
- 308 -> context/data_local/Bear_room/time_series/308.parquet
- 314 -> context/data_local/Bear_room/time_series/314.parquet
- 317 -> context/data_local/Bear_room/time_series/317.parquet
- 323 -> context/data_local/Bear_room/time_series/323.parquet
- 328 -> context/data_local/Bear_room/time_series/328.parquet
- 329 -> context/data_local/Bear_room/time_series/329.parquet
- 330 -> context/data_local/Bear_room/time_series/330.parquet
- 336 -> context/data_local/Bear_room/time_series/336.parquet
- 340 -> context/data_local/Bear_room/time_series/340.parquet
- 345 -> context/data_local/Bear_room/time_series/345.parquet
- 348 -> context/data_local/Bear_room/time_series/348.parquet
- 350 -> context/data_local/Bear_room/time_series/350.parquet
- 353 -> context/data_local/Bear_room/time_series/353.parquet
- 361 -> context/data_local/Bear_room/time_series/361.parquet
- 362 -> context/data_local/Bear_room/time_series/362.parquet
- 363 -> context/data_local/Bear_room/time_series/363.parquet
- 368 -> context/data_local/Bear_room/time_series/368.parquet
- 371 -> context/data_local/Bear_room/time_series/371.parquet
- 375 -> context/data_local/Bear_room/time_series/375.parquet
- 380 -> context/data_local/Bear_room/time_series/380.parquet
- 386 -> context/data_local/Bear_room/time_series/386.parquet
- 387 -> context/data_local/Bear_room/time_series/387.parquet
- 402 -> context/data_local/Bear_room/time_series/402.parquet
- 403 -> context/data_local/Bear_room/time_series/403.parquet
- 405 -> context/data_local/Bear_room/time_series/405.parquet
- 409 -> context/data_local/Bear_room/time_series/409.parquet
- 413 -> context/data_local/Bear_room/time_series/413.parquet
- 415 -> context/data_local/Bear_room/time_series/415.parquet
- 417 -> context/data_local/Bear_room/time_series/417.parquet
- 420 -> context/data_local/Bear_room/time_series/420.parquet
- 423 -> context/data_local/Bear_room/time_series/423.parquet
- 428 -> context/data_local/Bear_room/time_series/428.parquet
- 429 -> context/data_local/Bear_room/time_series/429.parquet
- 432 -> context/data_local/Bear_room/time_series/432.parquet
- 436 -> context/data_local/Bear_room/time_series/436.parquet
- 437 -> context/data_local/Bear_room/time_series/437.parquet
- 445 -> context/data_local/Bear_room/time_series/445.parquet
- 448 -> context/data_local/Bear_room/time_series/448.parquet
- 450 -> context/data_local/Bear_room/time_series/450.parquet
- 453 -> context/data_local/Bear_room/time_series/453.parquet
- 461 -> context/data_local/Bear_room/time_series/461.parquet
- 462 -> context/data_local/Bear_room/time_series/462.parquet
- 463 -> context/data_local/Bear_room/time_series/463.parquet
- 469 -> context/data_local/Bear_room/time_series/469.parquet
- 470 -> context/data_local/Bear_room/time_series/470.parquet
- 484 -> context/data_local/Bear_room/time_series/484.parquet
- 490 -> context/data_local/Bear_room/time_series/490.parquet

Variables (per entity)
- Zone Temperature
- Real Power Mean
- Actual Supply Flow

Dataset: California_ISO
Source:
- context/data_local/California_ISO/id_info.json
- context/data_local/California_ISO/static_info.json
Time series path resolution (data config)
- Data config used: data_configs/California_ISO/fullCAISO_hetero_TGTSF.yaml
- root_path: ./data/California_ISO/time_series
- formatter: {i}.parquet
- id_info in config: id_info_except_battery.json (not in context/data_local)
- id selection in config: all_except_battery
- Resolved rule: file path = {root_path}/{id}.parquet for each id in id_info_except_battery.json

Entity -> file path
- all -> filename not specified in context JSONs

Resolved hetero paths (FidelTSPathResolver)
- Data config used: data_configs/California_ISO/fullCAISO_hetero_TGTSF.yaml
- base_data_path: ./data (from FidelTSEmbeddingLoader caller)
- old_embedding_paths:
  - ./data/California_ISO/weather/merged_report_embedding/fast_general_formal_embeddings_2018.pkl
  - ./data/California_ISO/weather/merged_report_embedding/fast_general_formal_embeddings_2019.pkl
  - ./data/California_ISO/weather/merged_report_embedding/fast_general_formal_embeddings_2020.pkl
  - ./data/California_ISO/weather/merged_report_embedding/fast_general_formal_embeddings_2021.pkl
  - ./data/California_ISO/weather/merged_report_embedding/fast_general_formal_embeddings_2022.pkl
  - ./data/California_ISO/weather/merged_report_embedding/fast_general_formal_embeddings_2023.pkl
  - ./data/California_ISO/weather/merged_report_embedding/fast_general_formal_embeddings_2024.pkl
  - ./data/California_ISO/weather/merged_report_embedding/fast_general_formal_embeddings_2025.pkl
- old_text_paths (only if directory exists):
  - ./data/California_ISO/weather/merged_general_report/merged_general_weather_forecast.json
- old_static_path:
  - ./data/California_ISO/weather/merged_report_embedding/static_info_embeddings_except_battery.pkl
- static_text_path:
  - ./data/California_ISO/static_info.json
- cache_base:
  - ./data/California_ISO/weather/embeddings_cache/merged_report_embedding

Variables (entity: all)
- Biogas CO2
- Biomass CO2
- Coal CO2
- Geothermal CO2
- Imports CO2
- Natural Gas CO2
- Current demand
- Day ahead forecast
- Hour ahead forecast
- BATTERIES
- BIOGAS
- BIOMASS
- COAL
- GEOTHERMAL
- IMPORTS
- LARGE HYDRO
- NATURAL GAS
- NUCLEAR
- SMALL HYDRO
- SOLAR
- WIND

Dataset: Canada_photovoltaics_plants
Source:
- context/data_local/Canada_photovoltaics_plants/id_info.json
- context/data_local/Canada_photovoltaics_plants/static_info.json
Time series path resolution (data config)
- Data config used: data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF.yaml
- root_path: ./data/Canada_photovoltaics_plants/time_series
- formatter: id_{i}.parquet
- id_info in config: id_info_imputed.json (not in context/data_local)
- id selection in config: all
- Resolved rule: file path = {root_path}/id_{id}.parquet for each id in id_info_imputed.json

Entity -> file path
- 314106 -> filename not specified in context JSONs
- 319086 -> filename not specified in context JSONs
- 164440 -> filename not specified in context JSONs
- 355827 -> filename not specified in context JSONs
- 331901 -> filename not specified in context JSONs
- 332785 -> filename not specified in context JSONs
- 577650 -> filename not specified in context JSONs
- 551172 -> filename not specified in context JSONs
- 570079 -> filename not specified in context JSONs

Resolved hetero paths (FidelTSPathResolver)
- Data config used: data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF.yaml
- base_data_path: ./data (from FidelTSEmbeddingLoader caller)
- old_embedding_paths:
  - ./data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/fast_general_formal_embeddings_2015.pkl
  - ./data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/fast_general_formal_embeddings_2016.pkl
  - ./data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/fast_general_formal_embeddings_2017.pkl
  - ./data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/fast_general_formal_embeddings_2018.pkl
  - ./data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/fast_general_formal_embeddings_2019.pkl
  - ./data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/fast_general_formal_embeddings_2020.pkl
  - ./data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/fast_general_formal_embeddings_2021.pkl
  - ./data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/fast_general_formal_embeddings_2022.pkl
  - ./data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/fast_general_formal_embeddings_2023.pkl
- old_text_paths:
  - ./data/Canada_photovoltaics_plants/weather/calgary/weather_report/formal_report/fast_general_formal_forecast_2015.json
  - ./data/Canada_photovoltaics_plants/weather/calgary/weather_report/formal_report/fast_general_formal_forecast_2016.json
  - ./data/Canada_photovoltaics_plants/weather/calgary/weather_report/formal_report/fast_general_formal_forecast_2017.json
  - ./data/Canada_photovoltaics_plants/weather/calgary/weather_report/formal_report/fast_general_formal_forecast_2018.json
  - ./data/Canada_photovoltaics_plants/weather/calgary/weather_report/formal_report/fast_general_formal_forecast_2019.json
  - ./data/Canada_photovoltaics_plants/weather/calgary/weather_report/formal_report/fast_general_formal_forecast_2020.json
  - ./data/Canada_photovoltaics_plants/weather/calgary/weather_report/formal_report/fast_general_formal_forecast_2021.json
  - ./data/Canada_photovoltaics_plants/weather/calgary/weather_report/formal_report/fast_general_formal_forecast_2022.json
  - ./data/Canada_photovoltaics_plants/weather/calgary/weather_report/formal_report/fast_general_formal_forecast_2023.json
- old_static_path:
  - ./data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/static_info_embeddings.pkl
- static_text_path:
  - ./data/Canada_photovoltaics_plants/static_info.json
- cache_base:
  - ./data/Canada_photovoltaics_plants/weather/embeddings_cache/calgary/report_embedding/formal_report

Variables (per entity)
- Solar power generation (kWh) is implied by general_info.
- Per-variable names are not listed in static_info.json.

Dataset: Germany_Renewable_Power_Grid
Source:
- context/data_local/Germany_Renewable_Power_Grid/id_info.json
- context/data_local/Germany_Renewable_Power_Grid/static_info.json
Time series path resolution (data config)
- Data config used: data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_TGTSF.yaml
- root_path: ./data/Germany_Renewable_Power_Grid/impute_data
- formatter: {i}.parquet
- id_info in config: id_info.json (under ./data/Germany_Renewable_Power_Grid/impute_data)
- id selection in config: all
- Resolved rule: file path = {root_path}/{id}.parquet for each id in impute_data/id_info.json

Entity -> file path
- solar_50Hertz -> filename not specified in context JSONs
- wind_50Hertz -> filename not specified in context JSONs
- solar_TenneT -> filename not specified in context JSONs
- wind_TenneT -> filename not specified in context JSONs
- solar_Amprion -> filename not specified in context JSONs
- wind_Amprion -> filename not specified in context JSONs
- solar_TransnetBW -> filename not specified in context JSONs
- wind_TransnetBW -> filename not specified in context JSONs

Resolved hetero paths (FidelTSPathResolver)
- Data config used: data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_TGTSF.yaml
- base_data_path: ./data (from FidelTSEmbeddingLoader caller)
- old_embedding_paths:
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2011.pkl
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2012.pkl
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2013.pkl
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2014.pkl
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2015.pkl
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2016.pkl
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2017.pkl
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2018.pkl
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2019.pkl
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2020.pkl
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2021.pkl
- old_text_paths (only if directory exists):
  - ./data/Germany_Renewable_Power_Grid/weather/merged_general_report/merged_general_weather_report.json
- old_static_path:
  - ./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/static_info_embeddings.pkl
- static_text_path:
  - ./data/Germany_Renewable_Power_Grid/static_info.json
- cache_base:
  - ./data/Germany_Renewable_Power_Grid/weather/embeddings_cache/merged_report_embedding

Variables (per entity)
- Renewable energy supply for the entity (solar or wind), implied by general_info.
- Per-variable names are not listed in static_info.json.

Dataset: Jena_Atmospheric_Physics
Source:
- context/data_local/Jena_Atmospheric_Physics/id_info.json
- context/data_local/Jena_Atmospheric_Physics/time_series/id_info.json
- context/data_local/Jena_Atmospheric_Physics/static_info.json
Time series path resolution (data config)
- Data config used: data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_TGTSF.yaml
- root_path: ./data/Jena_Atmospheric_Physics/time_series
- formatter: {i}.parquet
- id_info in config: id_info.json (under ./data/Jena_Atmospheric_Physics/time_series)
- id selection in config: all
- Resolved rule: file path = {root_path}/{id}.parquet for each id in time_series/id_info.json

Entity -> file path
- weather_large -> filename not specified in context JSONs

Resolved hetero paths (FidelTSPathResolver)
- Data config used: data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_TGTSF.yaml
- base_data_path: ./data (from FidelTSEmbeddingLoader caller)
- old_embedding_path:
  - ./data/Jena_Atmospheric_Physics/weather/report_embedding/formal_report/wm_messages_v3.pkl
- old_text_path:
  - ./data/Jena_Atmospheric_Physics/weather/weather_report/formal_report/wm_messages_v3.json
- old_static_path:
  - ./data/Jena_Atmospheric_Physics/weather/report_embedding/formal_report/static_info_embeddings.pkl
- static_text_path:
  - ./data/Jena_Atmospheric_Physics/static_info.json
- cache_base:
  - ./data/Jena_Atmospheric_Physics/weather/embeddings_cache/report_embedding/formal_report

Variables (entity: weather_large)
- p (mbar)
- T (degC)
- Tpot (K)
- Tdew (degC)
- rh (%)
- VPmax (mbar)
- VPact (mbar)
- VPdef (mbar)
- sh (g/kg)
- H2OC (mmol/mol)
- rho (g/m³)
- wv (m/s)
- max. wv (m/s)
- wd (deg)
- rain (mm)
- raining (s)
- SWDR (W/m²)
- PAR (μmol/m²/s)
- max. PAR (μmol/m²/s)
- Tlog (degC)
- CO2 (ppm)

Dataset: NYC_traffic_speed
Source:
- context/data_local/NYC_traffic_speed/static_info.json
Time series path resolution (data config)
- Data config used: data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF_H.yaml
- root_path: ./data/NYC_traffic_speed/time_series
- formatter: id_{i}.parquet
- id_info in config: id_info_imputed.json (not in context/data_local)
- id selection in config: all
- Resolved rule: file path = {root_path}/id_{id}.parquet for each id in id_info_imputed.json

Entity -> file path
- Sensor IDs are the keys in static_info.json.channel_info.
- Filenames are not specified in context JSONs.

Resolved hetero paths (FidelTSPathResolver)
- Data config used: data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF_H.yaml
- base_data_path: ./data (from FidelTSEmbeddingLoader caller)
- old_embedding_paths:
  - ./data/NYC_traffic_speed/weather/merged_report_embedding/fast_general_formal_embeddings_2018.pkl
  - ./data/NYC_traffic_speed/weather/merged_report_embedding/fast_general_formal_embeddings_2019.pkl
  - ./data/NYC_traffic_speed/weather/merged_report_embedding/fast_general_formal_embeddings_2020.pkl
  - ./data/NYC_traffic_speed/weather/merged_report_embedding/fast_general_formal_embeddings_2021.pkl
  - ./data/NYC_traffic_speed/weather/merged_report_embedding/fast_general_formal_embeddings_2022.pkl
- old_text_paths (only if directory exists):
  - ./data/NYC_traffic_speed/weather/merged_general_report/merged_general_weather_report.json
- old_static_path:
  - ./data/NYC_traffic_speed/weather/merged_report_embedding/static_info_embeddings.pkl
- static_text_path:
  - ./data/NYC_traffic_speed/static_info.json
- cache_base:
  - ./data/NYC_traffic_speed/weather/embeddings_cache/merged_report_embedding

Variables (per entity)
- Average speed (km/h) is implied by general_info.
- Per-variable names are not listed in static_info.json.

Integrity Note
- Every entity and variable is listed per dataset without deduplication or merging.
- Missing filenames are explicitly labeled as unspecified in the context JSONs.
