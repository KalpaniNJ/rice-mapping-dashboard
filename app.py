import streamlit as st
import geemap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import ee
from datetime import datetime
import calendar
import numpy as np
from scipy.signal import argrelextrema
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
import geemap.foliumap as geemap

# Sidebar (inputs, buttons & title) ===============================================================================================================================
# with st.sidebar:
#     # st.header("Data Inputs")
#     aoi = st.selectbox("Select AOI", ["MahaKanadarawa", "Anuradhapura", "Polonnaruwa"])
#     start_date = st.date_input("Start Date")
#     end_date = st.date_input("End Date")
#     run_button = st.button("Run Analysis")

# st.sidebar.header("Inputs")

with st.sidebar.expander("Time Series Analysis"):
    st.info("Plotting sample points over several years may be heavy. Use a limited date range (e.g., a single season).")
    
    aoi_option = st.selectbox(
        "Select AOI",
        ["MahaKanadarawa Water Influence Zone", "MahaKanadarawa Irrigable Area"],
        key="aoi_select_tab1"
    )

    start_date = st.date_input("Start Date", pd.to_datetime("2021-12-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2022-05-31"))
    run_button = st.button("Run Time Series Analysis")

with st.sidebar.expander("Outlier Analysis"):
    st.info("Perform Time Series analysis before Outlier analysis.")
    outlier_button = st.button("Run Outlier Analysis")

with st.sidebar.expander("Rice Mapping"):
    st.info("Select the Start, Peak, and Harvest dates. These will be used for further analysis.")
    season_start_date = st.date_input("Start of Season", value=pd.to_datetime("2021-12-13"))
    peak_date = st.date_input("Peak of Season", value=pd.to_datetime("2022-02-25"))
    harvest_date = st.date_input("Harvest Date", value=pd.to_datetime("2022-04-01"))
    run_paddy_button = st.button("Run Paddy Season Analysis")

with st.sidebar.expander("Statistical Analysis"):
    st.info("Calculate total paddy area, area by month, and area by start date.")
    run_stats = st.button("Run Statistical Analysis")

with st.sidebar.expander("Monitoring"):
    st.info("Monitor seasonal rice growth. Select the period and run the analysis")

    aoi_option = st.selectbox(
        "Select AOI",
        ["MahaKanadarawa Water Influence Zone", "MahaKanadarawa Irrigable Area"],
        key="aoi_select_tab2"
    )

    start_date_tab2 = st.date_input("Start Date", pd.to_datetime("2023-11-01"), key="start_tab2")
    end_date_tab2 = st.date_input("End Date", pd.to_datetime("2024-01-31"), key="end_tab2")
    run_monitor = st.button("Run Analysis")

# Title =========================================================================================================================================================
# Page config
st.set_page_config(page_title="Rice Mapping Dashboard", layout="wide")

st.markdown(
    """
    <div style="
        background-image: url('https://cdn.pixabay.com/photo/2021/05/25/08/13/paddy-field-6281737_960_720.jpg'); 
        background-size: cover; 
        background-position: center; 
        background-repeat: no-repeat;
        padding: 60px;
        text-align: center; 
        height: 450px;
        color: white;">
        <h1 style='font-size: 100px; margin-top: 75px; text-shadow: 3px 3px 6px #000000; '>Rice Mapping Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<hr style='border:2px solid #0d6efd'>", unsafe_allow_html=True)


# Initialize GEE =================================================================================================================================================
if 'gee_initialized' not in st.session_state:
    with st.spinner("Initializing Google Earth Engine..."):
        ee.Authenticate()  # only needed first time
        ee.Initialize(project='rice-mapping-472904')
    st.session_state['gee_initialized'] = True  # mark as initialized


# Define assets ==================================================================================================================================================
aoi_assets = {
    "MahaKanadarawa Water Influence Zone": "projects/rice-mapping-472904/assets/mk_influence_zone",
    "MahaKanadarawa Irrigable Area": "projects/rice-mapping-472904/assets/mk_Irrigable_Area"
}

points = ee.FeatureCollection("projects/rice-mapping-472904/assets/SamplePtsMahaKanadarawa")

water = ee.FeatureCollection('projects/rice-mapping-472904/assets/mkTanks')
roads = ee.FeatureCollection('projects/rice-mapping-472904/assets/mkRoads')

# Create tabs
tab1, tab2 = st.tabs(["Seasonal Analysis", "Seasonal Monitoring"])











# Tab1: Run Analysis ==================================================================================================================================================
with tab1:
    # Load AOI
    aoiCollection = ee.FeatureCollection(aoi_assets[aoi_option])
    aoi = aoiCollection.geometry()

    # Define dates
    startDate = ee.Date(str(start_date))
    endDate = ee.Date(str(end_date))

    # Creates a list of dekads (12-day periods per month) from the given date range
    # Calculates the number of months between startDate and endDate
    # Creates a list of months starting from startDate
    numMonths = endDate.difference(startDate, 'month').round()

    def func_ocb(month):
        return startDate.advance(ee.Number(month), 'month')

    monthSequence = ee.List.sequence(0, numMonths, 1).map(func_ocb)

    # Function to generate dekad dates for a given month

    def func_jha(date):
        date = ee.Date(date)
        y = date.get('year')
        m = date.get('month')

        dekad1 = ee.Date.fromYMD(y, m, 1)
        dekad2 = ee.Date.fromYMD(y, m, 13)
        dekad3 = ee.Date.fromYMD(y, m, 25)

        return [dekad1, dekad2, dekad3]

    generateDekads = func_jha

    # Get the dekadList
    dekadList = monthSequence.map(generateDekads).flatten()

    def func_kbb(date):
        return ee.Algorithms.If(
        ee.Date(date).millis().lte(endDate.millis()),
        date,
        None
        )

    filteredDekadList = dekadList.map(func_kbb).removeAll([None])

    # Remove duplicate dekad dates from filteredDekadList
    filteredDekadList = filteredDekadList.distinct()

    # Loads the ESA WorldCover 2020 dataset - Extracts cropland areas (Class 40)
    polarization = 'VH'

    # Defining the mRVI formula, based on Agapiou, 2020 - "https":#doi.Org/10.3390/app10144764
    def func_hyl (img):
        mRVI = img \
        .select(['VV']) \
        .divide(img.select(['VV']).add(img.select(['VH']))) \
        .pow(0.5) \
        .multiply(
        img \
        .select(['VH']) \
        .multiply(ee.Image(4)) \
        .divide(img.select(['VV']).add(img.select(['VH'])))
        ) \
        .rename('mRVI')
        return img.addBands(mRVI)

    addmRVI = func_hyl

    rvi = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
    .filterBounds(aoi) \
    .filterDate(startDate, endDate) \
    .filter(ee.Filter.eq('instrumentMode','IW')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization)) \
    .filter(ee.Filter.eq('resolution_meters', 10)) \
    .map(addmRVI) \
    .select('mRVI')

    rvi_sorted = rvi.sort("system:time_start")

    def func_wxd(dekad):
        start_date = ee.Date(dekad)
        currentIndex = ee.Number(filteredDekadList.indexOf(dekad))
        nextIndex = currentIndex.add(1)
        nextDate = ee.Algorithms.If(
            nextIndex.lt(filteredDekadList.size()),
            ee.Date(filteredDekadList.get(nextIndex)),
            endDate
        )

        dekadImages = rvi_sorted.filterDate(start_date, nextDate)
        mRVIImages = dekadImages.select('mRVI')

        def make_image():
            img = mRVIImages.reduce(ee.Reducer.median())
            # Set dekad and system:time_start correctly
            return img.set({
                'dekad': dekad,
                'system:time_start': start_date.millis()
            })

        return ee.Algorithms.If(
            mRVIImages.size().gt(0),
            make_image(),
            None
        )

    createMosaic = func_wxd

    # Convert List to ImageCollection & Remove Nulls
    mosaicImages = ee.List(filteredDekadList.map(createMosaic)).removeAll([None])
    mosaicCollection = ee.ImageCollection.fromImages(mosaicImages)

    def func_zty(img):
        # preserve properties
        img2 = img.multiply(10000).toUint16()
        return img2.copyProperties(img, ['dekad', 'system:time_start'])

    mosaicCollectionUInt16 = mosaicCollection.map(func_zty)


    # Time-series + Point graph (GEE data) ===========================================================================================================================
    st.markdown("## Rice Time Series Analysis")
    st.markdown(
        "<span style='font-size:14px; color:gray;'>"
        "Analyze rice growth over time using <b>mean mRVI</b> at sample points. "
        "Select an AOI and date range to generate the time series chart."
        "</span>",
        unsafe_allow_html=True
    )

    # ----------------------------------------------------------------------------------------------------------------------- Run Analysis on Button Click
    if run_button:
        with st.spinner("Running Analysis... This may take a few minutes..."):

            # ----------------------------------------------------------------------------------------------------------------------- Time Series

            def sample_image(image, fc):
                fc = ee.FeatureCollection(fc)
                samples = image.sampleRegions(collection=points, scale=10, geometries=True)

                def add_time(f):
                    return f.set('time', ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'))
                return fc.merge(samples.map(add_time))

            initial_fc = ee.FeatureCollection([])
            sampled_fc = ee.FeatureCollection(mosaicCollectionUInt16.iterate(sample_image, initial_fc))

            # Convert to Pandas (df1)
            sampled_info = sampled_fc.getInfo()
            rows = [{"time": f['properties'].get('time'),
                    "mRVI": f['properties'].get('mRVI_median'),
                    "point_id": f['properties'].get('system:index')} for f in sampled_info['features']]
            df1 = pd.DataFrame(rows)
            df1['time'] = pd.to_datetime(df1['time'])
            df1 = df1.sort_values('time')

            # ----------------------------------------------------------------------------------------------------------------------- Point Graph

            def sample_image2(image):
                return image.sampleRegions(collection=points, scale=10, geometries=True)\
                            .map(lambda f: f.set('time', image.date().format('YYYY-MM-dd')))

            sampled_fc2 = mosaicCollectionUInt16.map(sample_image2).flatten()

            sampled_info2 = sampled_fc2.getInfo()
            rows2 = [{"time": f['properties'].get('time'),
                    "mRVI_median": f['properties'].get('mRVI_median'),
                    "point_id": f.get('id')} for f in sampled_info2['features']]
            df2 = pd.DataFrame(rows2)
            df2['time'] = pd.to_datetime(df2['time'])
            df2 = df2.sort_values('time')

            # Store in session_state
            st.session_state['df1'] = df1
            st.session_state['df2'] = df2

    df1 = st.session_state.get('df1')
    df2 = st.session_state.get('df2')

    if df1 is not None and df2 is not None:
        plt.style.use("dark_background")

        # ---------------------- First plot (line graph) ---------------------- #
        fig1, ax1 = plt.subplots(figsize=(12,8))
        for pid, group in df1.groupby("point_id"):
            ax1.plot(group['time'], group['mRVI'], marker='o', label=f"Point {pid}")

        mean_df = df1.groupby('time')['mRVI'].mean().reset_index()
        ax1.plot(mean_df['time'], mean_df['mRVI'], color="#49EC44", linewidth=2, marker='o', markersize=6, label='Mean mRVI')

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.tick_params(axis='x', rotation=45, labelsize=10, colors='white')
        ax1.tick_params(axis='y', labelsize=10, colors='white')
        ax1.grid(alpha=0.2, linestyle="--", color="white")
        ax1.set_facecolor("#181717")
        fig1.patch.set_facecolor("#0d0d0d")
        ax1.set_xlabel("Date", fontsize=10, fontweight="bold", color="white")
        ax1.set_ylabel("mRVI Value", fontsize=10, fontweight="bold", color="white")
        ax1.set_title("Time Series of mean mRVI at Sample Points", fontsize=18, fontweight="bold", color="#FFFFFF")
        plt.tight_layout()
        st.session_state["lineplot_fig"] = fig1

        # ---------------------- Second plot (point graph) ---------------------- #
        fig2, ax2 = plt.subplots(figsize=(12,8))
        for pid, group in df2.groupby("point_id"):
            ax2.plot(group['time'], group['mRVI_median'], marker='o', linestyle='-', markersize=5, alpha=0.7)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.tick_params(axis='x', rotation=45, labelsize=12, colors='white')
        ax2.tick_params(axis='y', labelsize=12, colors='white')
        ax2.grid(alpha=0.2, linestyle="--", color="white")
        ax2.set_facecolor("#181717")
        fig2.patch.set_facecolor("#0d0d0d")
        ax2.set_xlabel("Date", fontsize=12, fontweight="bold", color="white")
        ax2.set_ylabel("mRVI Value", fontsize=12, fontweight="bold", color="white")
        ax2.set_title("Time Series of mRVI at Sample Points", fontsize=18, fontweight="bold", color="#FFFFFF")
        plt.tight_layout()
        st.session_state["pointplot_fig"] = fig2

    # ---------------------- Always render plots ---------------------- #
    col1, col2 = st.columns(2)
    with col1:
        if "lineplot_fig" in st.session_state:
            st.pyplot(st.session_state["lineplot_fig"])
    with col2:
        if "pointplot_fig" in st.session_state:
            st.pyplot(st.session_state["pointplot_fig"])


    st.markdown("<hr style='border:2px solid #1A1B1A'>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------------------------------------------------------- Outlier Analysis Section

    st.markdown("## Outlier Analysis")
    st.markdown(
        "<span style='font-size:14px; color:gray;'>"
        "Visualize dispersion of mRVI values and detect potential outliers at sample points."
        "</span>",
        unsafe_allow_html=True
    )

    if outlier_button:
        df2 = st.session_state.get('df2')
        if df2 is None:
            st.error("Please run the Time Series Analysis first.")
        else:
            # Reshape data (long format) for boxplot
            df_long = df2.melt(
                id_vars=["time"],                # Keep time as identifier
                value_vars=["mRVI_median"],      # The values to plot
                var_name="variable",
                value_name="value"
            )
            # Add point identifier as a category (so seaborn can distinguish points)
            df_long['point'] = df2['point_id']

            # Plot boxplot
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.boxplot(
                x="time",
                y="value",
                data=df_long,
                fliersize=4,            # outlier marker size
                width=0.55,
                boxprops=dict(facecolor="#2C9429", edgecolor="white", linewidth=0.5),  # box border
                whiskerprops=dict(color="white", linewidth=0.5),  # whiskers
                capprops=dict(color="white", linewidth=0.5),      # caps
                medianprops=dict(color="black", linewidth=0.5),  # median line
                flierprops=dict(marker='o', markersize=0.9, markerfacecolor="#257C22", markeredgecolor="white")  # outliers
            )
            ax.tick_params(axis='x', rotation=45, labelsize=4, colors='white')
            ax.tick_params(axis='y', labelsize=4, colors='white')
            ax.grid(axis='y', alpha=0.2, linestyle="--", color="white")
            ax.set_facecolor("#181717")
            fig.patch.set_facecolor("#0d0d0d")
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_xlabel("Date", fontsize=6, fontweight="bold", color="white")
            ax.set_ylabel("mRVI Value", fontsize=6, fontweight="bold", color="white")
            ax.set_title("mRVI Dispersion and Outlier Analysis at Sample Points", fontsize=8, fontweight="bold", color="#FFFFFF")

            plt.tight_layout()

        st.session_state["boxplot_fig"] = fig   # save figure in session_state

    # Always render if available
    if "boxplot_fig" in st.session_state:
        st.pyplot(st.session_state["boxplot_fig"])

    # ----------------------------------------------------------------------------------------------------------------------- Parameter Extraction

    df2 = st.session_state.get('df2')
    if df2 is not None:
        # Convert to long format
        df_long = df2.melt(
            id_vars=["time"],
            value_vars=["mRVI_median"],
            var_name="variable",
            value_name="value"
        )
        df_long['time'] = pd.to_datetime(df_long['time'])

        # ---------------------- Quartile Assignment ---------------------- #
        # Get sorted unique dates
        unique_times = df_long['time'].sort_values().unique()
        n_times = len(unique_times)
        quarter_size = n_times // 4

        quartiles = {}
        for i, date in enumerate(unique_times):
            if i < quarter_size:
                quartiles[date] = 1
            elif i < 2*quarter_size:
                quartiles[date] = 2
            elif i < 3*quarter_size:
                quartiles[date] = 3
            else:
                quartiles[date] = 4

        # ---------------------- Quantile Calculation ---------------------- #
        start_values = df_long[df_long['time'] == pd.to_datetime(start_date)]['value']
        peak_values = df_long[df_long['time'] == pd.to_datetime(peak_date)]['value']
        harvest_values = df_long[df_long['time'] == pd.to_datetime(harvest_date)]['value']

        q3_start = start_values.quantile(0.75)
        q1_peak = peak_values.quantile(0.25)

        # ---------------------- Difference Calculation ---------------------- #
        mean_start = start_values.mean()
        mean_peak = peak_values.mean()
        mean_harvest = harvest_values.mean()

        diff_start_peak = mean_peak - mean_start
        diff_peak_harvest = mean_peak - mean_harvest

        # st.markdown("### Quartile Information")
        # st.write(f"Start date {start_date} is in Quartile {quartiles.get(pd.to_datetime(start_date), 'NA')}")
        # st.write(f"Peak date {peak_date} is in Quartile {quartiles.get(pd.to_datetime(peak_date), 'NA')}")
        # st.write(f"Harvest date {harvest_date} is in Quartile {quartiles.get(pd.to_datetime(harvest_date), 'NA')}")
        # st.markdown("### Quantiles at Key Dates")
        # st.write(f"Q3 of start date ({start_date}): {q3_start}")
        # st.write(f"Q1 of peak date ({peak_date}): {q1_peak}")
        # st.markdown("### Mean and Differences")
        # st.write(f"Mean of start date ({start_date}): {mean_start}")
        # st.write(f"Mean of peak date ({peak_date}): {mean_peak}")
        # st.write(f"Mean of harvest date ({harvest_date}): {mean_harvest}")
        # st.write(f"Difference (Peak - Start): {diff_start_peak}")
        # st.write(f"Difference (Peak - Harvest): {diff_peak_harvest}")

    st.markdown("<hr style='border:2px solid #1A1B1A'>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------------------------------------------------------- mRVI SOS-Peak-Fall analysis

    st.markdown("## Map Visualization")
    st.markdown(
        "<span style='font-size:14px; color:gray;'>"
        "Visualizes the spatial distribution of paddy fields within the area of interest including the paddy map and start of rice cropping (by month and day)."
        "</span>",
        unsafe_allow_html=True
    )

    # Define dates
    sosDate = ee.Date(str(season_start_date))
    peakDate = ee.Date(str(peak_date))
    fallDate = ee.Date(str(harvest_date))

    if run_paddy_button:
        with st.spinner("Rendering the map..."):
            # Function to get adjacent dekads
            def getAdjacentDekads(targetDate, dekadList):
                index = dekadList.indexOf(targetDate)
                return ee.List([
                    dekadList.get(ee.Number(index).subtract(1)),
                    targetDate,
                    dekadList.get(ee.Number(index).add(1))
                ]).filter(ee.Filter.neq('item', None))

            # Extract SOS, Peak, Fall Images
            sosWindow = getAdjacentDekads(sosDate, filteredDekadList)
            sosImages = mosaicCollectionUInt16.filter(ee.Filter.inList('dekad', sosWindow))
            sosMin = sosImages.reduce(ee.Reducer.min())

            peakWindow = getAdjacentDekads(peakDate, filteredDekadList)
            peakImages = mosaicCollectionUInt16.filter(ee.Filter.inList('dekad', peakWindow))
            peakMax = peakImages.reduce(ee.Reducer.max())

            fallWindow = getAdjacentDekads(fallDate, filteredDekadList)
            fallImages = mosaicCollectionUInt16.filter(ee.Filter.inList('dekad', fallWindow))
            fallMin = fallImages.reduce(ee.Reducer.min())

            # Main Conditions
            positiveGrowth = peakMax.subtract(sosMin).gt(diff_start_peak/2)
            negativeDecline = peakMax.subtract(fallMin).gt(diff_peak_harvest/2)

            # Additional Temporal and Quartile Checks
            # thresholds from quartile analysis
            sosMaxThreshold = q3_start
            peakMinThreshold = q1_peak

            # Check SOS < Q3 and Peak > Q1
            valuePatternMask = sosMin.lte(sosMaxThreshold).And(peakMax.gte(peakMinThreshold))

            # Time difference in months between SOS min and Peak max
            timeDiffMonths = peakDate.difference(sosDate, 'month')
            timePatternMask = timeDiffMonths.gte(1)

            # Combine All Conditions
            paddyMask = positiveGrowth.And(negativeDecline).And(valuePatternMask).And(timePatternMask)

            paddyClassification = paddyMask.clip(aoi).rename('paddy_classified').selfMask()

            def clean_paddy_mask(paddy_mask, aoi, kernel_radius=1, min_object_area=10000):
                """Clean a paddy mask by masking tree cover and built-up areas, applying dilation, and removing small objects."""
                # Load ESA WorldCover and clip
                esa = ee.ImageCollection('ESA/WorldCover/v200').first().clip(aoi)
                
                # Mask tree cover and built-up areas
                tree_cover = esa.eq(10)
                built_up = esa.eq(50)
                paddy_clean = paddy_mask.updateMask(tree_cover.Not()).updateMask(built_up.Not())
                
                # Apply dilation
                kernel = ee.Kernel.circle(radius=kernel_radius, units='pixels')
                paddy_clean = paddy_clean.focal_max(kernel=kernel, iterations=1)
                
                # Object-based noise removal
                object_size = paddy_clean.connectedPixelCount(maxSize=128, eightConnected=False)
                pixel_area = ee.Image.pixelArea()
                object_area = object_size.multiply(pixel_area)
                
                # Mask small objects
                paddy_clean = paddy_clean.updateMask(object_area.gte(min_object_area))
                
                return paddy_clean

            # Add generalization
            cleaned_paddy = clean_paddy_mask(paddyClassification, aoi)

            #....................................................Mask roads & water features....................................................#
            # Set a mask property for each feature
            water = water.map(lambda f: f.set('mask', 1))
            roads = roads.map(lambda f: f.set('mask', 1))

            # Optional: buffer roads (e.g., 3 meters)
            roadsBuffer = roads.map(lambda f: f.buffer(3))

            # Convert features to raster mask
            waterMask = water.reduceToImage(properties=['mask'], reducer=ee.Reducer.first()).clip(aoi).unmask(0).gt(0)
            roadsMask = roadsBuffer.reduceToImage(properties=['mask'], reducer=ee.Reducer.first()).clip(aoi).unmask(0).gt(0)

            # Combine masks
            eraseMask = waterMask.Or(roadsMask)

            # Apply mask to paddyClassification
            maskedPaddyClassification = cleaned_paddy.updateMask(eraseMask.Not()).rename('masked_paddy_classified')
            maskedPaddyClassification = maskedPaddyClassification.updateMask(maskedPaddyClassification.gt(0))

            #...........................................................Get differences............................................................#
            def calculateDifference(prevImage, nextImage):
                diff = nextImage.subtract(prevImage)
                return diff.set({
                    'dekad1': prevImage.get('system:index'),
                    'dekad2': nextImage.get('system:index'),
                    'system:time_start': nextImage.get('system:time_start'),
                    'dekad1_time': prevImage.get('system:time_start'),
                    'dekad2_time': nextImage.get('system:time_start')
                })

            # Create list of consecutive image pairs
            mosaicList = mosaicCollectionUInt16.toList(mosaicCollectionUInt16.size())

            def func_ycf(i):
                prev = ee.Image(mosaicList.get(ee.Number(i).subtract(1)))
                next = ee.Image(mosaicList.get(i))
                return calculateDifference(prev, next)

            differences = ee.ImageCollection(
                ee.List.sequence(1, mosaicList.size().subtract(1)).map(func_ycf)
            )


            #..........................................................Check the continuation of positive differences..........................................................#
            def findSequentialGrowth(differences):
                diffList = differences.toList(differences.size())
                size = differences.size()

                def func_yth(index):
                    currImg = ee.Image(diffList.get(index))
                    nextIndex = ee.Number(index).add(1)
                    hasNext = nextIndex.lt(size)
                    nextImg = ee.Image(ee.Algorithms.If(hasNext, diffList.get(nextIndex), ee.Image(0)))

                    seqGrowth = currImg.gt(0).rename('sequential_growth') \
                        .set('start_dekad', currImg.get('dekad1')) \
                        .set('end_dekad', ee.Algorithms.If(hasNext, nextImg.get('dekad2'), currImg.get('dekad2'))) \
                        .set('system:time_start', currImg.get('system:time_start')) \
                        .set('start_time', currImg.get('dekad1_time')) \
                        .set('end_time', ee.Algorithms.If(hasNext, nextImg.get('dekad2_time'), currImg.get('dekad2_time')))

                    isContinuous = ee.Image(ee.Algorithms.If(
                        hasNext,
                        currImg.gt(0).And(nextImg.gt(0)),
                        ee.Image(0)
                    ))

                    return seqGrowth.addBands(isContinuous.rename('is_continuous')).set('growth_period', index)

                # Map over all indices and create ImageCollection
                images = ee.List.sequence(0, size.subtract(2)).map(func_yth)
                return ee.ImageCollection.fromImages(images)

            # Create sequential growth map
            sequentialDiffs = findSequentialGrowth(differences)

            def func_wun(img):
                return img.select('sequential_growth') \
                    .multiply(img.select('is_continuous')) \
                    .rename('sequential_growth') \
                    .round() \
                    .set('growth_period', img.get('growth_period')) \
                    .set('start_time', ee.Number(img.get('start_time'))) \
                    .set('end_time', ee.Number(img.get('end_time')))

            sequentialImgs = sequentialDiffs.map(func_wun)

            imgList = sequentialImgs.toList(sequentialImgs.size())

            #..........................................................Track start date of the longest streak..........................................................#
            # Initial dictionary for iterate
            init = ee.Dictionary({
                'currentLength': ee.Image(0),
                'longestLength': ee.Image(0),
                'currentStartDate': ee.Image(0),
                'longestStartDate': ee.Image(0)
            })

            def func_hxg(imgObj, prev):
                img = ee.Image(imgObj).clip(aoi)
                prev = ee.Dictionary(prev)

                prevCurrentLength = ee.Image(prev.get('currentLength'))
                prevLongestLength = ee.Image(prev.get('longestLength'))
                prevCurrentStartDate = ee.Image(prev.get('currentStartDate'))
                prevLongestStartDate = ee.Image(prev.get('longestStartDate'))

                prevCurrentStartMonth = ee.Image(prev.get('currentStartMonth'))
                prevLongestStartMonth = ee.Image(prev.get('longestStartMonth'))

                prevCurrentStartMonthDay = ee.Image(prev.get('currentStartMonthDay'))
                prevLongestStartMonthDay = ee.Image(prev.get('longestStartMonthDay'))

                isOne = img.eq(1)

                # Increment current streak if 1, reset if 0
                newCurrentLength = prevCurrentLength.add(isOne).multiply(isOne)

                # --- Start date (millis) ---
                newCurrentStartDate = prevCurrentStartDate.where(
                    prevCurrentLength.eq(0).And(isOne),
                    ee.Image.constant(ee.Number(img.get('start_time')))
                )

                # --- Start month (MM) ---
                newCurrentStartMonth = prevCurrentStartMonth.where(
                    prevCurrentLength.eq(0).And(isOne),
                    ee.Image.constant(ee.Date(img.get('start_time')).get('month'))
                )

                # --- Start month-day (MMDD, e.g., March 5 = 305) ---
                newCurrentStartMonthDay = prevCurrentStartMonthDay.where(
                    prevCurrentLength.eq(0).And(isOne),
                    ee.Image.constant(
                        ee.Number(ee.Date(img.get('start_time')).get('month')).multiply(100)
                        .add(ee.Number(ee.Date(img.get('start_time')).get('day')))
                    )
                )

                # --- Update longest streak length ---
                newLongestLength = prevLongestLength.max(newCurrentLength)

                # --- Update longest streak start date/month/month-day if this is a new max ---
                newLongestStartDate = prevLongestStartDate \
                    .where(newCurrentLength.gt(prevLongestLength), newCurrentStartDate) \
                    .where(newCurrentLength.eq(prevLongestLength)
                        .And(newCurrentStartDate.lt(prevLongestStartDate)),
                        newCurrentStartDate)

                newLongestStartMonth = prevLongestStartMonth \
                    .where(newCurrentLength.gt(prevLongestLength), newCurrentStartMonth) \
                    .where(newCurrentLength.eq(prevLongestLength)
                        .And(newCurrentStartDate.lt(prevLongestStartDate)),
                        newCurrentStartMonth)

                newLongestStartMonthDay = prevLongestStartMonthDay \
                    .where(newCurrentLength.gt(prevLongestLength), newCurrentStartMonthDay) \
                    .where(newCurrentLength.eq(prevLongestLength)
                        .And(newCurrentStartDate.lt(prevLongestStartDate)),
                        newCurrentStartMonthDay)

                return ee.Dictionary({
                    'currentLength': newCurrentLength,
                    'longestLength': newLongestLength,
                    'currentStartDate': newCurrentStartDate,
                    'longestStartDate': newLongestStartDate,
                    'currentStartMonth': newCurrentStartMonth,
                    'longestStartMonth': newLongestStartMonth,
                    'currentStartMonthDay': newCurrentStartMonthDay,
                    'longestStartMonthDay': newLongestStartMonthDay
                })

            init = ee.Dictionary({
                'currentLength': ee.Image(0),
                'longestLength': ee.Image(0),
                'currentStartDate': ee.Image(0),
                'longestStartDate': ee.Image(0),
                'currentStartMonth': ee.Image(0),
                'longestStartMonth': ee.Image(0),
                'currentStartMonthDay': ee.Image(0),
                'longestStartMonthDay': ee.Image(0)
            })

            # Run iteration
            result = imgList.iterate(func_hxg, init)
            final = ee.Dictionary(result)

            # Final maps
            finalLongest = ee.Image(final.get('longestLength')).clip(aoi).rename('Longest_Streak')
            finalStartDate = ee.Image(final.get('longestStartDate')).clip(aoi).rename('Longest_Streak_Start')
            finalStartMonth = ee.Image(final.get('longestStartMonth')).clip(aoi).rename('Longest_Streak_Start_MM')
            finalStartMonthDay = ee.Image(final.get('longestStartMonthDay')).clip(aoi).rename('Longest_Streak_Start_MMDD')

            # Mask to paddy
            maskedLongest = finalLongest.updateMask(maskedPaddyClassification)
            maskedStartDate = finalStartDate.updateMask(maskedPaddyClassification)
            maskedStartMonth = finalStartMonth.updateMask(maskedPaddyClassification)
            maskedStartMonthDay = finalStartMonthDay.updateMask(maskedPaddyClassification)

            st.session_state["maskedPaddyClassification"] = maskedPaddyClassification
            st.session_state["maskedStartMonth"] = maskedStartMonth
            st.session_state["maskedStartMonthDay"] = maskedStartMonthDay

    # Restore EE images if available
    if "maskedPaddyClassification" in st.session_state:
        maskedPaddyClassification = st.session_state["maskedPaddyClassification"]
        maskedStartMonth = st.session_state["maskedStartMonth"]
        maskedStartMonthDay = st.session_state["maskedStartMonthDay"]

        # Re-create the map
        aoi_centroid = aoi.centroid().coordinates().getInfo()
        center_coords = [aoi_centroid[1], aoi_centroid[0]]
        m = geemap.Map(center=center_coords, zoom=13)

        rviVis = {"min": 0.0, "max":8000, "palette": ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901', '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01', '012E01', '011D01', '011301']}
        m.add_basemap("Esri.WorldImagery")
        
        m.addLayer(mosaicCollectionUInt16.median().clip(aoi), rviVis, 'mRVI')
        # m.addLayer(paddyClassification, {"min": 0, "max": 1, "palette": ['red', 'green']}, '(mRVI SOS-Peak-Fall) Paddy Map')
        m.addLayer(maskedPaddyClassification, {"min": 0, "max": 1, "palette": ['green']}, 'Masked Paddy Map', False)
        # m.addLayer(waterMask.updateMask(waterMask), {"palette": ['#0000FF']}, 'Water Mask', False)
        # m.addLayer(roadsMask.updateMask(roadsMask), {"palette": ["#FDEE69"]}, 'Roads Mask', False)
        # m.addLayer(maskedLongest, {"min": 0, "max": 10, "palette": ['white', 'blue']}, 'Longest Streak', False)
        # m.addLayer(maskedStartDate, {"min": 1638297000000, "max": 1653935400000, "palette": ['#ffeda0', "#bd008e"]}, 'Longest Streak Start Date', False)
        m.addLayer(maskedStartMonth, {"min": 1, "max": 12, "palette": ["blue", "cyan", "green", "lime", "yellow", "orange", "red", "pink", "purple", "brown", "gray", "black"]}, "Start Month", False)
        m.addLayer(maskedStartMonthDay, {"min": 101, "max": 1231, "palette": ["blue", "cyan", "green", "yellow", "orange", "red"]}, "Start MMDD", False)
        m.addLayerControl()

        # Render map
        folium_static(m, width=1200, height=700)

    st.markdown("<hr style='border:2px solid #1A1B1A'>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------------------------------------------------------- Statistical Analysis

    st.markdown("## Statistical Analysis")
    st.markdown(
        "<span style='font-size:14px; color:gray;'>"
        "Calculates the total paddy extent and visualizes the distribution of cropping start dates both by month and by specific day (MM-DD)."
        "</span>",
        unsafe_allow_html=True
    )

    if run_stats:
        with st.spinner("Calculating areas..."):
            # Total area (all paddy pixels)
            total_area = maskedPaddyClassification.multiply(ee.Image.pixelArea()) \
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e13
                ).getInfo()["masked_paddy_classified"] / 10000   # m² → ha
            
            # Area By Month
            month_area = ee.Image.pixelArea().addBands(maskedStartMonth) \
                .reduceRegion(
                    reducer=ee.Reducer.sum().group(
                        groupField=1,
                        groupName='month'
                    ),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e13
                ).getInfo()

            month_groups = month_area["groups"]
            month_stats = {g["month"]: g["sum"] / 10000 for g in month_groups}

            # Area By MMDD
            mmdd_area = ee.Image.pixelArea().addBands(maskedStartMonthDay) \
                .reduceRegion(
                    reducer=ee.Reducer.sum().group(
                        groupField=1,
                        groupName='mmdd'
                    ),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e13
                ).getInfo()

            mmdd_groups = mmdd_area["groups"]
            mmdd_stats = {g["mmdd"]: g["sum"] / 10000 for g in mmdd_groups}

            # -------------------------------
            # Convert month numbers to names
            df_month = pd.DataFrame(list(month_stats.items()), columns=["Month", "Area_ha"])
            df_month = df_month[df_month["Month"] != 0]
            df_month["Month_Name"] = df_month["Month"].apply(lambda x: calendar.month_name[x])
            df_month = df_month.sort_values("Month")

            # Convert MMDD
            df_mmdd = pd.DataFrame(list(mmdd_stats.items()), columns=["MMDD", "Area_ha"])
            df_mmdd = df_mmdd[df_mmdd["MMDD"] != 0]
            df_mmdd["Month_Day"] = df_mmdd["MMDD"].apply(lambda x: f"{str(x).zfill(4)[:2]}-{str(x).zfill(4)[2:]}")
            df_mmdd = df_mmdd.sort_values("MMDD")

            # -------------------------------
            # Display total paddy extent
            st.markdown(f"### 🌾 Total Paddy Extent: **{total_area:,.0f} ha**")

            # -------------------------------
            # First row: 3 charts
            row1_col1, row1_col2, row1_col3 = st.columns(3)

            with row1_col1:
                # Bar chart by Month
                fig, ax = plt.subplots(figsize=(5,4))
                ax.bar(df_month["Month_Name"], df_month["Area_ha"], color='skyblue')
                ax.set_xlabel("Month"); ax.set_ylabel("Area (ha)")
                ax.set_title("By Month")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

            with row1_col2:
                # Bar chart by MMDD
                fig, ax = plt.subplots(figsize=(5,4))
                ax.bar(df_mmdd["Month_Day"], df_mmdd["Area_ha"], color='lightgreen')
                ax.set_xlabel("Date"); ax.set_ylabel("Area (ha)")
                ax.set_title("By Start Date")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

            with row1_col3:
                # Pie chart by Month
                fig, ax = plt.subplots(figsize=(5,4))
                wedges, texts, autotexts = ax.pie(
                    df_month["Area_ha"],
                    startangle=90,
                    colors=plt.cm.tab20.colors,
                    autopct='%1.1f%%',  # show percentage inside slices
                    pctdistance=0.75,   # move text closer to center
                    wedgeprops=dict(width=0.5)
                )
                ax.set_title("Month %")
                ax.legend(
                    wedges,
                    df_month["Month_Name"],
                    title="Start Month",
                    loc="center left",
                    bbox_to_anchor=(1,0,0.3,1),
                    fontsize=8  # smaller legend font
                )
                st.pyplot(fig)


                # Second row: 1 chart (Pie chart by MMDD)
                row2_col1 = st.columns(1)[0]

                with row2_col1:
                    labels = df_mmdd["Month_Day"]
                    sizes = df_mmdd["Area_ha"]
                    cmap = plt.cm.viridis(np.linspace(0, 1, len(labels)))
                    fig, ax = plt.subplots(figsize=(5,4))  # smaller, same as 3rd chart
                    wedges, texts, autotexts = ax.pie(
                        sizes,
                        startangle=45,
                        colors=cmap,
                        autopct='%1.1f%%',
                        pctdistance=1.2,
                        wedgeprops=dict(width=0.5)
                    )
                    ax.set_title("Date %")
                    ax.legend(
                        wedges,
                        labels,
                        title="Start Date (MM-DD)",
                        loc="lower center",
                        bbox_to_anchor=(0.5, -0.30),
                        fontsize=8,
                        ncol=4
                    )
                    st.pyplot(fig)












# Tab2: Run Analysis ==================================================================================================================================================
with tab2:
    if run_monitor:
            with st.spinner("Running seasonal analysis... Please wait."):

                # Load AOI
                aoiCollection = ee.FeatureCollection(aoi_assets[aoi_option])
                aoi = aoiCollection.geometry()

                # Define dates
                startDate = ee.Date(str(start_date_tab2))
                endDate = ee.Date(str(end_date_tab2))

                # Creates a list of dekads (12-day periods per month) from the given date range
                # Calculates the number of months between startDate and endDate
                # Creates a list of months starting from startDate
                numMonths = endDate.difference(startDate, 'month').round()

                def func_ocb(month):
                    return startDate.advance(ee.Number(month), 'month')

                monthSequence = ee.List.sequence(0, numMonths, 1).map(func_ocb)

                # Function to generate dekad dates for a given month

                def func_jha(date):
                    date = ee.Date(date)
                    y = date.get('year')
                    m = date.get('month')

                    dekad1 = ee.Date.fromYMD(y, m, 1)
                    dekad2 = ee.Date.fromYMD(y, m, 13)
                    dekad3 = ee.Date.fromYMD(y, m, 25)

                    return [dekad1, dekad2, dekad3]

                generateDekads = func_jha

                # Get the dekadList
                dekadList = monthSequence.map(generateDekads).flatten()

                def func_kbb(date):
                    return ee.Algorithms.If(
                    ee.Date(date).millis().lte(endDate.millis()),
                    date,
                    None
                    )

                filteredDekadList = dekadList.map(func_kbb).removeAll([None])

                # Remove duplicate dekad dates from filteredDekadList
                filteredDekadList = filteredDekadList.distinct()

                # Loads the ESA WorldCover 2020 dataset - Extracts cropland areas (Class 40)
                polarization = 'VH'

                # Defining the mRVI formula, based on Agapiou, 2020 - "https":#doi.Org/10.3390/app10144764
                def func_hyl (img):
                    mRVI = img \
                    .select(['VV']) \
                    .divide(img.select(['VV']).add(img.select(['VH']))) \
                    .pow(0.5) \
                    .multiply(
                    img \
                    .select(['VH']) \
                    .multiply(ee.Image(4)) \
                    .divide(img.select(['VV']).add(img.select(['VH'])))
                    ) \
                    .rename('mRVI')
                    return img.addBands(mRVI)

                addmRVI = func_hyl

                rvi = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
                .filterBounds(aoi) \
                .filterDate(startDate, endDate) \
                .filter(ee.Filter.eq('instrumentMode','IW')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization)) \
                .filter(ee.Filter.eq('resolution_meters', 10)) \
                .map(addmRVI) \
                .select('mRVI')

                rvi_sorted = rvi.sort("system:time_start")

                def func_wxd(dekad):
                    start_date = ee.Date(dekad)
                    currentIndex = ee.Number(filteredDekadList.indexOf(dekad))
                    nextIndex = currentIndex.add(1)
                    nextDate = ee.Algorithms.If(
                        nextIndex.lt(filteredDekadList.size()),
                        ee.Date(filteredDekadList.get(nextIndex)),
                        endDate
                    )

                    dekadImages = rvi_sorted.filterDate(start_date, nextDate)
                    mRVIImages = dekadImages.select('mRVI')

                    def make_image():
                        img = mRVIImages.reduce(ee.Reducer.median())
                        # Set dekad and system:time_start correctly
                        return img.set({
                            'dekad': dekad,
                            'system:time_start': start_date.millis()
                        })

                    return ee.Algorithms.If(
                        mRVIImages.size().gt(0),
                        make_image(),
                        None
                    )

                createMosaic = func_wxd

                # Convert List to ImageCollection & Remove Nulls
                mosaicImages = ee.List(filteredDekadList.map(createMosaic)).removeAll([None])
                mosaicCollection = ee.ImageCollection.fromImages(mosaicImages)

                def func_zty(img):
                    # preserve properties
                    img2 = img.multiply(10000).toUint16()
                    return img2.copyProperties(img, ['dekad', 'system:time_start'])

                mosaicCollectionUInt16 = mosaicCollection.map(func_zty)


                # Time-series + Point graph (GEE data) ===========================================================================================================================

                col1, col2 = st.columns(2)
                # ----------------------------------------------------------------------------------------------------------------------- Time Series
                
                with col1:
                    st.markdown("### Time Series Analysis")
                    # fig1, ax1 = plt.subplots(figsize=(4, 2))

                    def sample_image(image, fc):
                        fc = ee.FeatureCollection(fc)
                        samples = image.sampleRegions(collection=points, scale=10, geometries=True)

                        def add_time(f):
                            return f.set('time', ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'))
                        return fc.merge(samples.map(add_time))

                    initial_fc = ee.FeatureCollection([])
                    sampled_fc = ee.FeatureCollection(mosaicCollectionUInt16.iterate(sample_image, initial_fc))

                    # Convert to Pandas (df1)
                    sampled_info = sampled_fc.getInfo()
                    rows = [{"time": f['properties'].get('time'),
                            "mRVI": f['properties'].get('mRVI_median'),
                            "point_id": f['properties'].get('system:index')} for f in sampled_info['features']]
                    df1 = pd.DataFrame(rows)
                    df1['time'] = pd.to_datetime(df1['time'])
                    df1 = df1.sort_values('time')

                    # ---------------------- First plot (line graph) ---------------------- #
                    fig1, ax1 = plt.subplots(figsize=(12,8))
                    for pid, group in df1.groupby("point_id"):
                        ax1.plot(group['time'], group['mRVI'], marker='o', label=f"Point {pid}")

                    mean_df = df1.groupby('time')['mRVI'].mean().reset_index()
                    ax1.plot(mean_df['time'], mean_df['mRVI'], color="#49EC44", linewidth=2, marker='o', markersize=6, label='Mean mRVI')

                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax1.tick_params(axis='x', rotation=45, labelsize=10, colors='white')
                    ax1.tick_params(axis='y', labelsize=10, colors='white')
                    ax1.grid(alpha=0.2, linestyle="--", color="white")
                    ax1.set_facecolor("#181717")
                    fig1.patch.set_facecolor("#0d0d0d")
                    ax1.set_xlabel("Date", fontsize=10, fontweight="bold", color="white")
                    ax1.set_ylabel("mRVI Value", fontsize=10, fontweight="bold", color="white")
                    ax1.set_title("Time Series of mean mRVI at Sample Points", fontsize=18, fontweight="bold", color="#FFFFFF")
                    plt.tight_layout()
                    st.pyplot(fig1)

                # ----------------------------------------------------------------------------------------------------------------------- Outlier Analysis Section
                
                with col2:
                    st.markdown("### Outlier Analysis")
                    # fig, ax = plt.subplots(figsize=(6, 4))

                    def sample_image3(image):
                        return image.sampleRegions(collection=points, scale=10, geometries=True)\
                                    .map(lambda f: f.set('time', image.date().format('YYYY-MM-dd')))

                    sampled_fc3 = mosaicCollectionUInt16.map(sample_image3).flatten()

                    sampled_info3 = sampled_fc3.getInfo()
                    rows3 = [{"time": f['properties'].get('time'),
                            "mRVI_median": f['properties'].get('mRVI_median'),
                            "point_id": f.get('id')} for f in sampled_info3['features']]
                    df3 = pd.DataFrame(rows3)
                    df3['time'] = pd.to_datetime(df3['time'])
                    df3 = df3.sort_values('time')

                    # Reshape data (long format) for boxplot
                    df_long = df3.melt(
                        id_vars=["time"],                # Keep time as identifier
                        value_vars=["mRVI_median"],      # The values to plot
                        var_name="variable",
                        value_name="value"
                    )
                    # Add point identifier as a category (so seaborn can distinguish points)
                    df_long['point'] = df3['point_id']

                    # Plot boxplot
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.boxplot(
                        x="time",
                        y="value",
                        data=df_long,
                        fliersize=4,            # outlier marker size
                        width=0.55,
                        boxprops=dict(facecolor="#2C9429", edgecolor="white", linewidth=0.5),  # box border
                        whiskerprops=dict(color="white", linewidth=0.5),  # whiskers
                        capprops=dict(color="white", linewidth=0.5),      # caps
                        medianprops=dict(color="black", linewidth=0.5),  # median line
                        flierprops=dict(marker='o', markersize=0.9, markerfacecolor="#257C22", markeredgecolor="white")  # outliers
                    )
                    
                    ax.tick_params(axis='x', rotation=45, labelsize=4, colors='white')
                    ax.tick_params(axis='y', labelsize=4, colors='white')
                    ax.grid(axis='y', alpha=0.2, linestyle="--", color="white")
                    ax.set_facecolor("#181717")
                    fig.patch.set_facecolor("#0d0d0d")
                    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    ax.set_xlabel("Date", fontsize=6, fontweight="bold", color="white")
                    ax.set_ylabel("mRVI Value", fontsize=6, fontweight="bold", color="white")
                    ax.set_title("mRVI Dispersion and Outlier Analysis at Sample Points", fontsize=8, fontweight="bold", color="#FFFFFF")
                    plt.tight_layout()
                    st.pyplot(fig)


                # ----------------------------------------------------------------------------------------------------------------------- Parameter Extraction

                # Compute median mRVI across points
                median_df = df3.groupby('time')['mRVI_median'].median().reset_index()
                mRVI_values = median_df['mRVI_median'].values
                time_values = median_df['time'].values

                # ------------------ Start Date ------------------ #
                prv_fall_date = pd.to_datetime(time_values[0])  # first available date

                # ------------------ SOS Date ------------------ #
                # Detect local minima in mRVI
                local_min_idx = argrelextrema(mRVI_values, np.less, order=1)[0]

                # Pick the first local minimum after the start
                next_sos_idx = local_min_idx[local_min_idx > 0][0] if len(local_min_idx) > 0 else 0
                next_sos_date = pd.to_datetime(time_values[next_sos_idx])

                # ------------------ Next Peak Date ------------------ #
                next_peak_date = pd.to_datetime(time_values[-1])  # last available date

                # print("Previous Fall (Start) Date:", prv_fall_date.date())
                # print("Detected SOS Date:", next_sos_date.date())
                # print("Next Peak (End) Date:", next_peak_date.date())

                # Convert time to datetime
                df_long['time'] = pd.to_datetime(df_long['time'])

                # Use the detected dates
                start_date = prv_fall_date
                sos_date = next_sos_date
                peak_date = next_peak_date

                # ---------------------- Quantile Calculation ---------------------- #
                start_values = df_long[df_long['time'] == start_date]['value']
                sos_values = df_long[df_long['time'] == sos_date]['value']
                peak_values = df_long[df_long['time'] == peak_date]['value']

                # Calculate quartiles
                q3_sos = sos_values.quantile(0.75)
                q1_peak = peak_values.quantile(0.25)

                # ---------------------- Difference Calculation ---------------------- #
                mean_start = start_values.mean()
                mean_sos = sos_values.mean()
                mean_peak = peak_values.mean()

                diff_start_sos = mean_start - mean_sos
                diff_sos_peak = mean_peak - mean_sos

                # print(f"Q3 of SOS Date ({sos_date.date()}): {q3_sos}")
                # print(f"Q1 of Peak Date ({peak_date.date()}): {q1_peak}")                        

                # print(f"\nMean of Previous Fall Date ({start_date.date()}): {mean_start}")
                # print(f"Mean of SOS Date ({sos_date.date()}): {mean_sos}")
                # print(f"Mean of Next Peak Date ({peak_date.date()}): {mean_peak}\n")

                # print(f"Difference (Previous Fall - SOS): {diff_start_sos}")
                # print(f"Difference (Next Peak - SOS): {diff_sos_peak}")

                #..........................................................mRVI SOS-Peak-Fall analysis..........................................................#
                #  Function to get adjacent dekads
                def getAdjacentDekads(targetDate, dekadList):
                    index = dekadList.indexOf(targetDate)
                    return ee.List([
                        dekadList.get(ee.Number(index).subtract(1)),
                        targetDate,
                        dekadList.get(ee.Number(index).add(1))
                    ]).filter(ee.Filter.neq('item', None))

                # Extract SOS, Peak, Fall Images
                start_Window = getAdjacentDekads(start_date, filteredDekadList)
                start_Images = mosaicCollectionUInt16.filter(ee.Filter.inList('dekad', start_Window))
                start_Max = start_Images.reduce(ee.Reducer.max())

                sos_Window = getAdjacentDekads(sos_date, filteredDekadList)
                sos_Images = mosaicCollectionUInt16.filter(ee.Filter.inList('dekad', sos_Window))
                sos_Min = sos_Images.reduce(ee.Reducer.min())

                peak_Window = getAdjacentDekads(peak_date, filteredDekadList)
                peak_Images = mosaicCollectionUInt16.filter(ee.Filter.inList('dekad', peak_Window))
                peak_Max = peak_Images.reduce(ee.Reducer.max())

                # Main Conditions
                positive_Growth = peak_Max.subtract(sos_Min).gt(diff_sos_peak/2)
                negative_Decline = start_Max.subtract(sos_Min).gt(diff_start_sos/2)

                # Additional Temporal and Quartile Checks
                # thresholds from quartile analysis
                sos_MaxThreshold = q3_sos
                peak_MinThreshold = q1_peak

                # Check SOS < Q3 and Peak > Q1
                value_PatternMask = sos_Min.lte(sos_MaxThreshold).And(peak_Max.gte(peak_MinThreshold))

                # Combine All Conditions
                paddyMask = positive_Growth.And(negative_Decline).And(value_PatternMask)

                paddyClassification = paddyMask.clip(aoi).rename('paddy_classified').selfMask()

                def clean_paddy_mask(paddy_mask, aoi, kernel_radius=1, min_object_area=10000):
                    """Clean a paddy mask by masking tree cover and built-up areas, applying dilation, and removing small objects."""
                    # Load ESA WorldCover and clip
                    esa = ee.ImageCollection('ESA/WorldCover/v200').first().clip(aoi)
                    
                    # Mask tree cover and built-up areas
                    tree_cover = esa.eq(10)
                    built_up = esa.eq(50)
                    paddy_clean = paddy_mask.updateMask(tree_cover.Not()).updateMask(built_up.Not())
                    
                    # Apply dilation
                    kernel = ee.Kernel.circle(radius=kernel_radius, units='pixels')
                    paddy_clean = paddy_clean.focal_max(kernel=kernel, iterations=1)
                    
                    # Object-based noise removal
                    object_size = paddy_clean.connectedPixelCount(maxSize=128, eightConnected=False)
                    pixel_area = ee.Image.pixelArea()
                    object_area = object_size.multiply(pixel_area)
                    
                    # Mask small objects
                    paddy_clean = paddy_clean.updateMask(object_area.gte(min_object_area))
                    
                    return paddy_clean

                # Add generalization
                cleaned_paddy = clean_paddy_mask(paddyClassification, aoi)

                #....................................................Mask roads & water features....................................................#
                # Set a mask property for each feature
                water = water.map(lambda f: f.set('mask', 1))
                roads = roads.map(lambda f: f.set('mask', 1))

                # Optional: buffer roads (e.g., 3 meters)
                roadsBuffer = roads.map(lambda f: f.buffer(3))

                # Convert features to raster mask
                waterMask = water.reduceToImage(properties=['mask'], reducer=ee.Reducer.first()).clip(aoi).unmask(0).gt(0)
                roadsMask = roadsBuffer.reduceToImage(properties=['mask'], reducer=ee.Reducer.first()).clip(aoi).unmask(0).gt(0)

                # Combine masks
                eraseMask = waterMask.Or(roadsMask)

                # Apply mask to paddyClassification
                maskedPaddyClassification = cleaned_paddy.updateMask(eraseMask.Not()).rename('masked_paddy_classified')
                maskedPaddyClassification = maskedPaddyClassification.updateMask(maskedPaddyClassification.gt(0))

                #...........................................................Get differences............................................................#
                def calculateDifference(prevImage, nextImage):
                    diff = nextImage.subtract(prevImage)
                    return diff.set({
                        'dekad1': prevImage.get('system:index'),
                        'dekad2': nextImage.get('system:index'),
                        'system:time_start': nextImage.get('system:time_start'),
                        'dekad1_time': prevImage.get('system:time_start'),
                        'dekad2_time': nextImage.get('system:time_start')
                    })

                # Create list of consecutive image pairs
                mosaicList = mosaicCollectionUInt16.toList(mosaicCollectionUInt16.size())

                def func_ycf(i):
                    prev = ee.Image(mosaicList.get(ee.Number(i).subtract(1)))
                    next = ee.Image(mosaicList.get(i))
                    return calculateDifference(prev, next)

                differences = ee.ImageCollection(
                    ee.List.sequence(1, mosaicList.size().subtract(1)).map(func_ycf)
                )

                #..........................................................Check the continuation of positive differences..........................................................#
                def findSequentialGrowth(differences):
                    diffList = differences.toList(differences.size())
                    size = differences.size()

                    def func_yth(index):
                        currImg = ee.Image(diffList.get(index))
                        nextIndex = ee.Number(index).add(1)
                        hasNext = nextIndex.lt(size)
                        nextImg = ee.Image(ee.Algorithms.If(hasNext, diffList.get(nextIndex), ee.Image(0)))

                        seqGrowth = currImg.gt(0).rename('sequential_growth') \
                            .set('start_dekad', currImg.get('dekad1')) \
                            .set('end_dekad', ee.Algorithms.If(hasNext, nextImg.get('dekad2'), currImg.get('dekad2'))) \
                            .set('system:time_start', currImg.get('system:time_start')) \
                            .set('start_time', currImg.get('dekad1_time')) \
                            .set('end_time', ee.Algorithms.If(hasNext, nextImg.get('dekad2_time'), currImg.get('dekad2_time')))

                        isContinuous = ee.Image(ee.Algorithms.If(
                            hasNext,
                            currImg.gt(0).And(nextImg.gt(0)),
                            ee.Image(0)
                        ))

                        return seqGrowth.addBands(isContinuous.rename('is_continuous')).set('growth_period', index)

                    # Map over all indices and create ImageCollection
                    images = ee.List.sequence(0, size.subtract(2)).map(func_yth)
                    return ee.ImageCollection.fromImages(images)

                # Create sequential growth map
                sequentialDiffs = findSequentialGrowth(differences)

                def func_wun(img):
                    return img.select('sequential_growth') \
                        .multiply(img.select('is_continuous')) \
                        .rename('sequential_growth') \
                        .round() \
                        .set('growth_period', img.get('growth_period')) \
                        .set('start_time', ee.Number(img.get('start_time'))) \
                        .set('end_time', ee.Number(img.get('end_time')))

                sequentialImgs = sequentialDiffs.map(func_wun)

                imgList = sequentialImgs.toList(sequentialImgs.size())

                #..........................................................Track start date of the longest streak..........................................................#
                # Initial dictionary for iterate
                init = ee.Dictionary({
                    'currentLength': ee.Image(0),
                    'longestLength': ee.Image(0),
                    'currentStartDate': ee.Image(0),
                    'longestStartDate': ee.Image(0)
                })

                def func_hxg(imgObj, prev):
                    img = ee.Image(imgObj).clip(aoi)
                    prev = ee.Dictionary(prev)

                    prevCurrentLength = ee.Image(prev.get('currentLength'))
                    prevLongestLength = ee.Image(prev.get('longestLength'))
                    prevCurrentStartDate = ee.Image(prev.get('currentStartDate'))
                    prevLongestStartDate = ee.Image(prev.get('longestStartDate'))

                    prevCurrentStartMonth = ee.Image(prev.get('currentStartMonth'))
                    prevLongestStartMonth = ee.Image(prev.get('longestStartMonth'))

                    prevCurrentStartMonthDay = ee.Image(prev.get('currentStartMonthDay'))
                    prevLongestStartMonthDay = ee.Image(prev.get('longestStartMonthDay'))

                    isOne = img.eq(1)

                    # Increment current streak if 1, reset if 0
                    newCurrentLength = prevCurrentLength.add(isOne).multiply(isOne)

                    # --- Start date (millis) ---
                    newCurrentStartDate = prevCurrentStartDate.where(
                        prevCurrentLength.eq(0).And(isOne),
                        ee.Image.constant(ee.Number(img.get('start_time')))
                    )

                    # --- Start month (MM) ---
                    newCurrentStartMonth = prevCurrentStartMonth.where(
                        prevCurrentLength.eq(0).And(isOne),
                        ee.Image.constant(ee.Date(img.get('start_time')).get('month'))
                    )

                    # --- Start month-day (MMDD, e.g., March 5 = 305) ---
                    newCurrentStartMonthDay = prevCurrentStartMonthDay.where(
                        prevCurrentLength.eq(0).And(isOne),
                        ee.Image.constant(
                            ee.Number(ee.Date(img.get('start_time')).get('month')).multiply(100)
                            .add(ee.Number(ee.Date(img.get('start_time')).get('day')))
                        )
                    )

                    # --- Update longest streak length ---
                    newLongestLength = prevLongestLength.max(newCurrentLength)

                    # --- Update longest streak start date/month/month-day if this is a new max ---
                    newLongestStartDate = prevLongestStartDate \
                        .where(newCurrentLength.gt(prevLongestLength), newCurrentStartDate) \
                        .where(newCurrentLength.eq(prevLongestLength)
                            .And(newCurrentStartDate.lt(prevLongestStartDate)),
                            newCurrentStartDate)

                    newLongestStartMonth = prevLongestStartMonth \
                        .where(newCurrentLength.gt(prevLongestLength), newCurrentStartMonth) \
                        .where(newCurrentLength.eq(prevLongestLength)
                            .And(newCurrentStartDate.lt(prevLongestStartDate)),
                            newCurrentStartMonth)

                    newLongestStartMonthDay = prevLongestStartMonthDay \
                        .where(newCurrentLength.gt(prevLongestLength), newCurrentStartMonthDay) \
                        .where(newCurrentLength.eq(prevLongestLength)
                            .And(newCurrentStartDate.lt(prevLongestStartDate)),
                            newCurrentStartMonthDay)

                    return ee.Dictionary({
                        'currentLength': newCurrentLength,
                        'longestLength': newLongestLength,
                        'currentStartDate': newCurrentStartDate,
                        'longestStartDate': newLongestStartDate,
                        'currentStartMonth': newCurrentStartMonth,
                        'longestStartMonth': newLongestStartMonth,
                        'currentStartMonthDay': newCurrentStartMonthDay,
                        'longestStartMonthDay': newLongestStartMonthDay
                    })

                init = ee.Dictionary({
                    'currentLength': ee.Image(0),
                    'longestLength': ee.Image(0),
                    'currentStartDate': ee.Image(0),
                    'longestStartDate': ee.Image(0),
                    'currentStartMonth': ee.Image(0),
                    'longestStartMonth': ee.Image(0),
                    'currentStartMonthDay': ee.Image(0),
                    'longestStartMonthDay': ee.Image(0)
                })

                # Run iteration
                result = imgList.iterate(func_hxg, init)
                final = ee.Dictionary(result)

                # Final maps
                finalLongest = ee.Image(final.get('longestLength')).clip(aoi).rename('Longest_Streak')
                finalStartDate = ee.Image(final.get('longestStartDate')).clip(aoi).rename('Longest_Streak_Start')
                finalStartMonth = ee.Image(final.get('longestStartMonth')).clip(aoi).rename('Longest_Streak_Start_MM')
                finalStartMonthDay = ee.Image(final.get('longestStartMonthDay')).clip(aoi).rename('Longest_Streak_Start_MMDD')

                # Mask to paddy
                maskedLongest = finalLongest.updateMask(maskedPaddyClassification)
                maskedStartDate = finalStartDate.updateMask(maskedPaddyClassification)
                maskedStartMonth = finalStartMonth.updateMask(maskedPaddyClassification)
                maskedStartMonthDay = finalStartMonthDay.updateMask(maskedPaddyClassification)

                # -------------------- Create Map --------------------
                aoi_centroid = aoi.centroid().coordinates().getInfo()
                center_coords = [aoi_centroid[1], aoi_centroid[0]]
                m = geemap.Map(center=center_coords, zoom=13)

                rviVis = {"min": 0.0, "max":8000, "palette": ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901', '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01', '012E01', '011D01', '011301']}
                m.add_basemap("Esri.WorldImagery")
                
                m.addLayer(mosaicCollectionUInt16.median().clip(aoi), rviVis, 'mRVI')
                # m.addLayer(paddyClassification, {"min": 0, "max": 1, "palette": ['red', 'green']}, '(mRVI SOS-Peak-Fall) Paddy Map')
                m.addLayer(maskedPaddyClassification, {"min": 0, "max": 1, "palette": ['green']}, 'Masked Paddy Map', False)
                m.addLayer(waterMask.updateMask(waterMask), {"palette": ['#0000FF']}, 'Water Mask', False)
                m.addLayer(roadsMask.updateMask(roadsMask), {"palette": ["#FDEE69"]}, 'Roads Mask', False)
                # m.addLayer(maskedLongest, {"min": 0, "max": 10, "palette": ['white', 'blue']}, 'Longest Streak', False)
                # m.addLayer(maskedStartDate, {"min": 1638297000000, "max": 1653935400000, "palette": ['#ffeda0', "#bd008e"]}, 'Longest Streak Start Date', False)
                m.addLayer(maskedStartMonth, {"min": 1, "max": 12, "palette": ["blue", "cyan", "green", "lime", "yellow", "orange", "red", "pink", "purple", "brown", "gray", "black"]}, "Start Month", False)
                m.addLayer(maskedStartMonthDay, {"min": 101, "max": 1231, "palette": ["blue", "cyan", "green", "yellow", "orange", "red"]}, "Start MMDD", False)
                m.addLayerControl()

                # -------------------- Display Map in Streamlit --------------------
                st.markdown("### Rice Seasonal Map")
                st.markdown("<span style='color:gray; font-size:13px;'>Visualize the paddy classification and growth pattern layers.</span>", unsafe_allow_html=True)
                m.to_streamlit(width=1200, height=700)


                # ----------------------------------------------------------------------------------------------------------------------- Statistical analysis
                
                st.markdown("## Statistical Analysis")
                st.markdown(
                    "<span style='font-size:14px; color:gray;'>"
                    "Calculates the total paddy extent and visualizes the distribution of cropping start dates both by month and by specific day (MM-DD)."
                    "</span>",
                    unsafe_allow_html=True
                )

                # Total area (all paddy pixels)
                total_area = maskedPaddyClassification.multiply(ee.Image.pixelArea()) \
                    .reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=aoi,
                        scale=10,
                        maxPixels=1e13
                    ).getInfo()["masked_paddy_classified"] / 10000   # m² → ha
                
                # Area By Month
                month_area = ee.Image.pixelArea().addBands(maskedStartMonth) \
                    .reduceRegion(
                        reducer=ee.Reducer.sum().group(
                            groupField=1,
                            groupName='month'
                        ),
                        geometry=aoi,
                        scale=10,
                        maxPixels=1e13
                    ).getInfo()

                month_groups = month_area["groups"]
                month_stats = {g["month"]: g["sum"] / 10000 for g in month_groups}

                # Area By MMDD
                mmdd_area = ee.Image.pixelArea().addBands(maskedStartMonthDay) \
                    .reduceRegion(
                        reducer=ee.Reducer.sum().group(
                            groupField=1,
                            groupName='mmdd'
                        ),
                        geometry=aoi,
                        scale=10,
                        maxPixels=1e13
                    ).getInfo()

                mmdd_groups = mmdd_area["groups"]
                mmdd_stats = {g["mmdd"]: g["sum"] / 10000 for g in mmdd_groups}

               # -------------------------------
                # Convert month numbers to names
                df_month = pd.DataFrame(list(month_stats.items()), columns=["Month", "Area_ha"])
                df_month = df_month[df_month["Month"] != 0]
                df_month["Month_Name"] = df_month["Month"].apply(lambda x: calendar.month_name[x])
                df_month = df_month.sort_values("Month")

                # Convert MMDD
                df_mmdd = pd.DataFrame(list(mmdd_stats.items()), columns=["MMDD", "Area_ha"])
                df_mmdd = df_mmdd[df_mmdd["MMDD"] != 0]
                df_mmdd["Month_Day"] = df_mmdd["MMDD"].apply(lambda x: f"{str(x).zfill(4)[:2]}-{str(x).zfill(4)[2:]}")
                df_mmdd = df_mmdd.sort_values("MMDD")

                # -------------------------------
                # Display total paddy extent
                st.markdown(f"### 🌾 Total Paddy Extent: **{total_area:,.0f} ha**")

                # -------------------------------
                # First row: 3 charts
                row1_col1, row1_col2, row1_col3 = st.columns(3)

                with row1_col1:
                    # Bar chart by Month
                    fig, ax = plt.subplots(figsize=(5,4))
                    ax.bar(df_month["Month_Name"], df_month["Area_ha"], color='skyblue')
                    ax.set_xlabel("Month"); ax.set_ylabel("Area (ha)")
                    ax.set_title("By Month")
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)

                with row1_col2:
                    # Bar chart by MMDD
                    fig, ax = plt.subplots(figsize=(5,4))
                    ax.bar(df_mmdd["Month_Day"], df_mmdd["Area_ha"], color='lightgreen')
                    ax.set_xlabel("Date"); ax.set_ylabel("Area (ha)")
                    ax.set_title("By Start Date")
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)

                with row1_col3:
                    # Pie chart by Month
                    fig, ax = plt.subplots(figsize=(5,4))
                    wedges, texts, autotexts = ax.pie(
                        df_month["Area_ha"],
                        startangle=90,
                        colors=plt.cm.tab20.colors,
                        autopct='%1.1f%%',  # show percentage inside slices
                        pctdistance=0.75,   # move text closer to center
                        wedgeprops=dict(width=0.5)
                    )
                    ax.set_title("Month %")
                    ax.legend(
                        wedges,
                        df_month["Month_Name"],
                        title="Start Month",
                        loc="center left",
                        bbox_to_anchor=(1,0,0.3,1),
                        fontsize=8  # smaller legend font
                    )
                    st.pyplot(fig)


                    # Second row: 1 chart (Pie chart by MMDD)
                    row2_col1 = st.columns(1)[0]

                    with row2_col1:
                        labels = df_mmdd["Month_Day"]
                        sizes = df_mmdd["Area_ha"]
                        cmap = plt.cm.viridis(np.linspace(0, 1, len(labels)))
                        fig, ax = plt.subplots(figsize=(5,4))  # smaller, same as 3rd chart
                        wedges, texts, autotexts = ax.pie(
                            sizes,
                            startangle=45,
                            colors=cmap,
                            autopct='%1.1f%%',
                            pctdistance=1.2,
                            wedgeprops=dict(width=0.5)
                        )
                        ax.set_title("Date %")
                        ax.legend(
                            wedges,
                            labels,
                            title="Start Date (MM-DD)",
                            loc="lower center",
                            bbox_to_anchor=(0.5, -0.30),
                            fontsize=8,
                            ncol=4
                        )
                        st.pyplot(fig) 