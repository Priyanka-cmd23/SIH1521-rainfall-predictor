"""Feature registry - the single source of truth for model features.

Every meteorology + derived feature used by the model is listed here with its
units, data source and physical reason for inclusion.  To add a future feature
(e.g. an INSAT-3D/3DR brightness-temperature column) you only add one dict here;
nothing else in the pipeline changes.
"""

FEATURES = [
    {"name": "temp_mean", "unit": "degC", "source": "ERA5",
     "why": "Air temperature; warm air holds more moisture available to rain out."},
    {"name": "temp_max", "unit": "degC", "source": "ERA5",
     "why": "Daily peak heating drives afternoon convection and storm growth."},
    {"name": "temp_min", "unit": "degC", "source": "ERA5",
     "why": "Night-time minimum relates to humidity build-up and stability."},
    {"name": "relative_humidity", "unit": "%", "source": "ERA5",
     "why": "Near-surface moisture - the direct fuel for heavy rainfall."},
    {"name": "dew_point", "unit": "degC", "source": "ERA5",
     "why": "Absolute moisture content of the air, independent of heating."},
    {"name": "surface_pressure", "unit": "hPa", "source": "ERA5",
     "why": "Low pressure means converging, rising air that produces clouds/rain."},
    {"name": "cloud_cover", "unit": "%", "source": "ERA5",
     "why": "Pre-monsoon/convective cloud build-up precedes heavy rain."},
    {"name": "wind_speed_mean", "unit": "km/h", "source": "ERA5",
     "why": "Transports moist ocean air into the region (monsoon flow)."},
    {"name": "wind_speed_max", "unit": "km/h", "source": "ERA5",
     "why": "Peak wind bursts signal active convective (storm) cells."},
    {"name": "wind_gust", "unit": "km/h", "source": "ERA5",
     "why": "Gustiness is a marker of convective (storm) activity."},
    {"name": "wind_direction", "unit": "deg", "source": "ERA5",
     "why": "South-westerly flow = monsoon moisture push; land/sea flow matters."},
    {"name": "rain_lag1", "unit": "mm", "source": "Derived (IMD prev. day)",
     "why": "Persistence - a wet day is often followed by another wet day."},
    {"name": "day_of_year", "unit": "day", "source": "Derived",
     "why": "Captures monsoon seasonality (June-September peak)."},
]

MODEL_FEATURES = [f["name"] for f in FEATURES]

# metadata columns carried along for analysis / spatial reference (NOT model inputs)
META_COLUMNS = ["date", "latitude", "longitude", "rainfall_mm"]

TARGET = "target"
RAIN_COL = "rainfall_mm"
THRESHOLD_MM = 64.5  # IMD "Very Heavy Rain" lower bound
