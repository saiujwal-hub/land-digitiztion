# Offline GIS Dataset Provenance & Metadata

## Overview
This directory contains verified open administrative and village GIS datasets for India (Karnataka & Telangana) derived from geoBoundaries Open Data, DataMeet Open Maps, and TGRAC (Telangana State Remote Sensing Applications Centre).

## Bundled Spatial Registries & Layers
1. **Karnataka Spatial Registry (ADM2/ADM3)**: `data/gis/karnataka/karnataka_spatial_registry.json`
2. **Karnataka Revenue Village Spatial Registry (ADM4)**: `data/gis/karnataka/karnataka_villages.geojson.gz` (**29,731 polygons**)
3. **Telangana Spatial Registry (ADM2/ADM3)**: `data/gis/telangana/telangana_spatial_registry.json`
4. **Telangana Master Village Spatial Registry (ADM4)**: `data/gis/telangana/telangana_villages.geojson.gz` (**10,906 polygons**)

## Provenance & Attribution

### 1. geoBoundaries ADM2 & ADM3 Datasets
- **Provider**: William & Mary GeoLab / geoBoundaries
- **Website**: https://www.geoboundaries.org
- **Source API & URLs**:
  - ADM2 (Districts): `https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/`
  - ADM3 (Subdistricts/Taluks/Mandals): `https://www.geoboundaries.org/api/current/gbOpen/IND/ADM3/`
- **Acquisition Date**: September 3, 2026
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Attribution**: `geoBoundaries (wmgeolab.org)`
- **Coordinate Reference System**: EPSG:4326 (WGS 84)
- **Scope**: District and Subdistrict boundary polygons for Karnataka & Telangana.

### 2. DataMeet Indian Village Boundaries Dataset (Karnataka)
- **Provider**: DataMeet Open Data Community
- **Website**: http://datameet.org
- **Source Repository**: `https://github.com/datameet/indian_village_boundaries/tree/master/ka`
- **Source File**: `ka/ka.geojson`
- **Acquisition Date**: September 3, 2026
- **License**: Open Database License (ODbL 1.0) / CC-BY 4.0
- **Attribution**: `DataMeet Community (datameet.org)`
- **Coordinate Reference System**: EPSG:4326 (WGS 84)
- **Original & Bundled Feature Count**: 29,731 Karnataka Revenue Villages
- **Transformation**: Extracted from raw DataMeet GeoJSON; normalized property keys (`village`, `taluk`, `district`, `census_loc_code`). Compressed with gzip (14.38 MB).

### 3. TGRAC Master Administrative Boundary Dataset (Telangana)
- **Provider**: Telangana State Remote Sensing Applications Centre (TGRAC), Planning Department, Government of Telangana
- **Website**: https://tgrac.telangana.gov.in
- **Source Service URL**: `https://tgrac.telangana.gov.in/arcgis/rest/services/Master_Administrative_Folder/Master_Administrative_Boundary_test/FeatureServer/5` (Master_Village_Boundary, Layer ID 5)
- **Acquisition Date**: September 3, 2026
- **License**: Official Government Open Administrative Data / Government of Telangana
- **Attribution**: `TGRAC Telangana Master Administrative Boundary (tgrac.telangana.gov.in)`
- **Coordinate Reference System**: EPSG:4326 (WGS 84)
- **Original & Bundled Feature Count**: 10,906 Telangana Master Revenue Villages
- **Transformation**: Acquired via deterministic OBJECTID-paginated ArcGIS REST query; preserved 100% polygon geometries and attributes (`district`, `mandal`, `village`, `district_code`, `subdistrict_code`, `village_code`, `census_2011_code`, `old_dist`, `old_mandal`). Compressed with gzip (28.07 MB).

## Scope & Limitations
- **Karnataka Village Resolution**: Active for Karnataka revenue villages via DataMeet source polygons.
- **Telangana Village Resolution**: Active for Telangana revenue villages via TGRAC official master administrative polygons.
- **Geographic Authority Consistency**: State-level authority checks are validated against official state administrative GIS datasets. National geographical master registry status remains `NOT_AVAILABLE` in local offline prototype.
- **Cadastral Survey Parcel Coverage**: `cadastral_status: "NOT_AVAILABLE"`, `survey_geometry: null`. Neither dataset contains authoritative revenue survey-number parcel boundaries. Estimated parcel boundary rectangles are approximate visualizations derived from document dimensions.
