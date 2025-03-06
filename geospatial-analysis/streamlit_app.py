# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_folium import st_folium
import folium

#########################
#  1. DATA LOADING
#########################

@st.cache_data
def load_clean_data():
    # Load the final pickled DataFrame
    df = pd.read_pickle("cleaned_data.pkl")
    return df

#########################
#  2. MAIN APP LOGIC
#########################

def main():
    st.title("Enhanced Olist Shipping Dashboard")

    # 2.1 Load Data
    df = load_clean_data()
    st.write(f"Data loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # 2.2 Sidebar Filters
    st.sidebar.header("Filter Options")

    # A) Delivery Days Range
    min_days = st.sidebar.slider("Min Delivery Days", 0, 100, 0)
    max_days = st.sidebar.slider("Max Delivery Days", 0, 200, 30)
    
    # B) State Selection
    if 'customer_state' in df.columns:
        all_states = sorted(df['customer_state'].dropna().unique())
        selected_states = st.sidebar.multiselect(
            "Select Customer States", 
            all_states, 
            default=all_states[:5]  # pre-select the first 5 states
        )
    else:
        selected_states = []

    # 2.3 Filter the DataFrame
    df_filtered = df[
        (df["purchase_to_delivery_days"] >= min_days) &
        (df["purchase_to_delivery_days"] <= max_days)
    ].copy()

    # Only filter by state if 'customer_state' column exists
    if 'customer_state' in df_filtered.columns and len(selected_states) > 0:
        df_filtered = df_filtered[df_filtered['customer_state'].isin(selected_states)]

    st.write(f"Filtered down to {df_filtered.shape[0]} rows.")
    
    # 2.4 Additional Sidebar Options for Visualization
    st.sidebar.header("Map Settings")

    map_style = st.sidebar.selectbox(
        "Select Map Style", 
        ["open-street-map","carto-positron","carto-darkmatter","stamen-terrain","stamen-toner"]
    )
    color_scale = st.sidebar.selectbox(
        "Select Color Scale",
        ["Viridis","Plasma","Temps","RdBu","YlOrRd","Cividis"]
    )

    # A user-set limit on how many rows to plot on the map (performance safeguard)
    max_points_for_map = st.sidebar.slider("Max Points on Map", 100, 5000, 1000)
    
    # 2.5 Show Key Metrics (KPI Cards)
    col1, col2, col3 = st.columns(3)
    mean_days = df_filtered["purchase_to_delivery_days"].mean() if len(df_filtered) else 0
    max_del_days = df_filtered["purchase_to_delivery_days"].max() if len(df_filtered) else 0
    order_count = len(df_filtered)

    col1.metric("Mean Delivery Days", f"{mean_days:.2f}")
    col2.metric("Max Delivery Days", f"{max_del_days:.2f}")
    col3.metric("Order Count", f"{order_count}")
    
    # 2.6 TABS for Different Sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Map", "State Comparison", "Data Table"])
    
    #########################
    #   TAB 1: OVERVIEW
    #########################
    with tab1:
        st.subheader("Delivery Days Distribution")
        # Basic histogram
        fig_hist = px.histogram(
            df_filtered, 
            x="purchase_to_delivery_days", 
            nbins=20,
            color_discrete_sequence=["#636EFA"],  # custom color
            title="Histogram of Purchase to Delivery Days"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Box plot
        st.subheader("Box Plot of Delivery Days")
        fig_box = px.box(
            df_filtered, 
            y="purchase_to_delivery_days",
            points="all",   # to show outliers
            title="Box Plot (Detailed)"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    #########################
    #   TAB 2: MAP
    #########################
    with tab2:
        st.subheader("Customer Locations Map")
        
        df_map = df_filtered.dropna(subset=["customer_lat","customer_lng"]).head(max_points_for_map).copy()
        
        if len(df_map) == 0:
            st.write("No points to display on the map after filtering.")
        else:
            # Using Plotly scatter_mapbox
            df_map["lat"] = df_map["customer_lat"]
            df_map["lon"] = df_map["customer_lng"]

            fig_map = px.scatter_mapbox(
                df_map,
                lat="lat",
                lon="lon",
                hover_name="order_id",
                hover_data=["purchase_to_delivery_days","customer_state"],
                color="purchase_to_delivery_days",
                color_continuous_scale=color_scale,
                zoom=3,
                height=600
            )
            fig_map.update_layout(mapbox_style=map_style)
            st.plotly_chart(fig_map, use_container_width=True)
    
    #########################
    #   TAB 3: STATE COMPARISON
    #########################
    with tab3:
        st.subheader("Compare Average Delivery by State")
        
        if 'customer_state' in df_filtered.columns:
            state_group = df_filtered.groupby('customer_state', as_index=False)['purchase_to_delivery_days'].mean()
            state_group.rename(columns={"purchase_to_delivery_days":"avg_delivery_days"}, inplace=True)
            
            # Sort descending by avg_delivery_days
            state_group.sort_values("avg_delivery_days", ascending=False, inplace=True)

            fig_state = px.bar(
                state_group,
                x="customer_state",
                y="avg_delivery_days",
                color="avg_delivery_days",
                color_continuous_scale=color_scale,
                title="Average Delivery Days by State"
            )
            st.plotly_chart(fig_state, use_container_width=True)
        else:
            st.write("No 'customer_state' column found in data.")
    
    #########################
    #   TAB 4: DATA TABLE
    #########################
    with tab4:
        st.subheader("Filtered Data")
        st.dataframe(df_filtered.head(300))  # limit to show top 300 rows

if __name__ == "__main__":
    main()
