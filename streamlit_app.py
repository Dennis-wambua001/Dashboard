"""
streamlit_app.py
----------------
Interactive AI Dashboard for Rice Growth & Yield Forecasting
Mwea Irrigation Scheme 🌾

Features:
✅ Upload or use existing CSV + raster data
✅ Display rasters on map (Folium basemap)
✅ Train models (Persistence, SARIMAX, LightGBM)
✅ Visualize historical + forecasted yield up to 2030
✅ Download forecast CSV
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime
import subprocess
import rasterio
from rasterio.plot import reshape_as_image
from matplotlib import cm

# --------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------
st.set_page_config(
    page_title="Mwea Rice Growth & Yield AI Dashboard",
    page_icon="assets/Dw.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------
logo_path = "assets/Dw.png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    with st.sidebar:
        st.image(logo, use_container_width=True)
        st.markdown("### 🌾 Mwea Rice Monitoring System")
        st.markdown("AI-Powered Yield Forecasting up to 2030")
        st.markdown("---")

st.sidebar.header("📂 Data & Layers")

# --------------------------------------------------------------
# LOAD CSV DATA
# --------------------------------------------------------------
CSV_path = "D:\ Documents\GIS practicals\GIS Prac\MWEA\AI yield system\CSV\Mwea data.csv"
if os.path.exists(CSV_path):
    df = pd.read_csv(CSV_path)
    df["Year"] = df["Year"].astype(int)
    st.sidebar.success("✅ Mwea_data.csv loaded successfully!")
else:
    st.sidebar.error("❌ CSV not found. Please ensure it is in the 'CSV' folder.")
    st.stop()

# --------------------------------------------------------------
# MAIN HEADER
# --------------------------------------------------------------
col_logo, col_title = st.columns([1, 8])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo, width=80)
with col_title:
    st.markdown(
        "<h1 style='margin-top: 10px;'>AI-Based Rice Growth & Yield Monitoring — Mwea Irrigation Scheme</h1>",
        unsafe_allow_html=True
    )

st.markdown(
    """
    This project develops an artificial intelligence (AI)-driven geospatial system 
    that uses satellite imagery and environmental data to monitor rice growth stages and predict 
    yield across the Mwea Irrigation Scheme. 
    It integrates remote sensing (RS), geographic information systems (GIS), and machine learning (ML) 
    to analyze vegetation indices (NDVI, EVI, NDWI), rainfall, soil and topography 
    to estimate crop performance and spatial yield variability.
    """
)
st.markdown("---")

map_col, charts_col = st.columns([2, 1])


# --------------------------------------------------------------
# LAYOUT: CENTER (MAP) | RIGHT (CHARTS)
"""
map_visualization.py
---------------------
Interactive map visualization for Mwea Irrigation Scheme 🌾

- NDVI layers: green symbology (YlGn)
- NDWI layers: blue symbology (Blues)
- LULC: categorical color palette
- Shapefiles overlay (roads, rivers, boundaries, etc.)
"""

import os
import folium
import rasterio
import numpy as np
from matplotlib import cm
from PIL import Image
import io, base64
import geopandas as gpd
from streamlit_folium import st_folium
import streamlit as st

# --- Paths ---
RASTER_DIR ="D:\ Documents\GIS practicals\GIS Prac\MWEA\AI yield system\Raster"
SHAPEFILE_DIR ="D:\ Documents\GIS practicals\GIS Prac\MWEA\AI yield system\Shapefiles"

st.subheader("🗺️ Mwea Irrigation Scheme Map Visualization")

# --- Validate directories ---
if not os.path.exists(RASTER_DIR):
    st.error(f"Raster folder not found: {RASTER_DIR}")
    st.stop()

if not os.path.exists(SHAPEFILE_DIR):
    st.warning("⚠️ No shapefile folder found. Continuing with rasters only.")

# --- List rasters ---
raster_files = [f for f in os.listdir(RASTER_DIR) if f.lower().endswith(".tif")]
if not raster_files:
    st.error("❌ No raster files found in Raster folder.")
    st.stop()

st.success(f"✅ {len(raster_files)} raster(s) detected.")

# --- Create Folium map ---
mwea_center = [-0.8, 37.45]
fmap = folium.Map(location=mwea_center, zoom_start=10, tiles="CartoDB positron")
last_bounds = None


# --- Helper: Add raster overlay ---
def add_raster_overlay(path, name, cmap_name, categorical=False):
    global last_bounds
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            arr = np.nan_to_num(arr, nan=0)

            if categorical:
                # Simple categorical coloring for LULC
                unique_vals = np.unique(arr)
                colors = cm.get_cmap("tab20", len(unique_vals))
                rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
                for i, val in enumerate(unique_vals):
                    mask = arr == val
                    rgba[mask] = (np.array(colors(i)) * 255).astype(np.uint8)
            else:
                # Continuous raster (NDVI, NDWI)
                arr_min, arr_max = np.nanpercentile(arr, [1, 99])
                arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-9)
                arr_norm = np.clip(arr_norm, 0, 1)
                rgba = (cm.get_cmap(cmap_name)(arr_norm) * 255).astype("uint8")

            # Convert to base64 PNG
            img = Image.fromarray(rgba)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            img_url = f"data:image/png;base64,{b64}"

            # Geographic bounds (south, west, north, east)
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            last_bounds = bounds

            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=bounds,
                opacity=0.75,
                name=name
            ).add_to(fmap)
    except Exception as e:
        st.warning(f"❌ Failed to load {name}: {e}")


# --- Add rasters with correct symbology ---
for f in raster_files:
    fpath = os.path.join(RASTER_DIR, f)
    fname = os.path.basename(f)

    if "ndvi" in fname.lower():
        add_raster_overlay(fpath, fname, "YlGn")  # 🌿 green
    elif "ndwi" in fname.lower():
        add_raster_overlay(fpath, fname, "Blues")  # 💧 blue
    elif "lulc" in fname.lower():
        add_raster_overlay(fpath, fname, "tab20", categorical=True)  # 🗺️ categorical
    else:
        add_raster_overlay(fpath, fname, "viridis")  # default fallback

# --- Add shapefiles ---
if os.path.exists(SHAPEFILE_DIR):
    shapefiles = [f for f in os.listdir(SHAPEFILE_DIR) if f.lower().endswith(".shp")]
    for shp_file in shapefiles:
        shp_path = os.path.join(SHAPEFILE_DIR, shp_file)
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            # Colors for shapefiles
            name_lower = shp_file.lower()
            if "river" in name_lower:
                color = "blue"
            elif "road" in name_lower:
                color = "brown"
            elif "landuse" in name_lower or "lulc" in name_lower:
                color = "green"
            elif "boundary" in name_lower or "mwea" in name_lower:
                color = "black"
            else:
                color = "gray"

            folium.GeoJson(
                gdf,
                name=shp_file.replace(".shp", ""),
                style_function=lambda x, color=color: {"color": color, "weight": 1.5},
                tooltip=shp_file.replace(".shp", "")
            ).add_to(fmap)
        except Exception as e:
            st.warning(f"Failed to load shapefile {shp_file}: {e}")

# --- Fit to raster extent and add controls ---
if last_bounds:
    fmap.fit_bounds(last_bounds)

folium.LayerControl(collapsed=False).add_to(fmap)

# --- Display map in Streamlit ---
st_folium(fmap, width=950, height=600)



# ---------------------- 📊 CHARTS SECTION ----------------------
with charts_col:
    st.subheader("📈 Rice Yield & Rainfall Trends")

    # --- Line Chart: Rice Yield over Years ---
    st.markdown("#### Rice Yield Over Years")
    df_line = df.groupby("Year", as_index=False)["Rice_Yield_tonnes"].mean()
    fig_line = px.line(
        df_line,
        x="Year", y="Rice_Yield_tonnes",
        markers=True,
        labels={"Year": "Year", "Rice_Yield_tonnes": "Yield (Tonnes)"}
    )
    fig_line.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Bar Chart: Rainfall vs. Rice Yield ---
st.markdown("#### Rainfall vs. Rice Yield")

# Group by rainfall or take averages if needed
df_bar = df.groupby("Rainfall", as_index=False)["Rice_Yield_tonnes"].mean()

# Create bar chart (Rainfall → X, Yield → Y)
fig_bar = px.bar(
    df_bar,
    x="Rainfall",
    y="Rice_Yield_tonnes",
    color="Rice_Yield_tonnes",
    title="Rainfall vs Rice Yield",
    labels={
        "Rainfall": "Rainfall (mm)",
        "Rice_Yield_tonnes": "Rice Yield (Tonnes)"
    },
    color_continuous_scale="YlGn"
)
fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------------------
# FORECASTING SECTION
# --------------------------------------------------------------
st.markdown("---")
st.header("📈 Forecasting Results & Model Evaluation")

forecast_csv = "outputs/forecast_to_2030.csv"
metrics_csv = "outputs/model_metrics.csv"

if os.path.exists(forecast_csv):
    df_forecast = pd.read_csv(forecast_csv)
    df_forecast["forecast_date"] = pd.to_datetime(df_forecast["forecast_date"])

    fig_forecast = px.line(
        df_forecast,
        x="forecast_date", y="forecast_tonnes",
        title="Forecasted Monthly Rice Yield (Tonnes)",
        labels={"forecast_date": "Date", "forecast_tonnes": "Yield (Tonnes)"}
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.download_button(
        label="💾 Download Forecast CSV",
        data=df_forecast.to_csv(index=False).encode(),
        file_name="forecast_to_2030.csv",
        mime="text/csv"
    )

if os.path.exists(metrics_csv):
    df_metrics = pd.read_csv(metrics_csv)
    st.subheader("📊 Model Evaluation Metrics")
    st.dataframe(df_metrics)

# --------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------
st.markdown("---")
st.markdown("**Developed for:** AI-Based Rice Growth & Yield Monitoring System — Mwea Irrigation Scheme 🌾")
st.markdown("**Created with:** Python • Streamlit • Folium • LightGBM • Statsmodels")
