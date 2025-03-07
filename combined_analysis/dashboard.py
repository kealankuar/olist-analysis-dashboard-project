import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

###########################################
# 1. DATA LOADING FUNCTIONS
###########################################

def load_item_data():
    """
    Load the row-level item DataFrame.
    This DataFrame includes order details, product categories, and other item-level info.
    """
    df_item = pd.read_pickle("../rfm_analysis/df_item.pkl")
    return df_item

@st.cache_data
def load_rfm_data():
    """
    Load the customer-level RFM DataFrame.
    Contains one row per customer with recency, frequency, and monetary metrics.
    """
    rfm_df = pd.read_pickle("../rfm_analysis/rfm_df.pkl")
    return rfm_df

@st.cache_data
def load_geo_data():
    """
    Load the geospatial DataFrame.
    This DataFrame contains geospatial information such as customer latitude and longitude.
    """
    geo_df = pd.read_pickle("../geospatial-analysis/cleaned_data.pkl")
    return geo_df

###########################################
# 2. HELPER FUNCTIONS
###########################################

def run_kmeans(df, n_clusters=4):
    """
    Run K-Means clustering on the continuous RFM metrics (recency, frequency, monetary).
    Returns a DataFrame with an added 'cluster' column.
    """
    required_cols = ["recency", "frequency", "monetary"]
    if not all(col in df.columns for col in required_cols):
        st.warning("One or more RFM columns are missing; cannot perform K-Means clustering.")
        df["cluster"] = 0
        return df

    X = df[required_cols].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    df["cluster"] = kmeans.labels_
    return df

###########################################
# 3. MAIN DASHBOARD APP
###########################################

def main():
    st.set_page_config(page_title="Advanced RFM + K-Means Dashboard", layout="wide")

    st.title("Olist E-commerce Dashboard: Advanced RFM + K-Means Segmentation")
    st.markdown(
        """
        This dashboard integrates multiple levels of data:
        - **Item-Level Data (df_item)**: Detailed order and product category information.
        - **Customer-Level RFM Data (rfm_df)**: Aggregated metrics for each customer (Recency, Frequency, Monetary).
        - **Geospatial Data (cleaned_data)**: Location information for geospatial analysis.
        
        Use the filters in the sidebar to narrow down the dataset by date and product categories.
        Navigate the tabs below for a comprehensive view:
        - **Item-Level Analysis**: Explore product categories and item-level details.
        - **RFM-Level Analysis**: Visualize the distributions of RFM metrics.
        - **Geospatial Analysis**: Analyze customer geospatial data with state filtering and product category labels.
        - **K-Means Clustering**: Run K-Means segmentation on the RFM data and view clustered results on a map.
        - **Combined Insight**: See item-level data with corresponding RFM metrics.
        - **Data Tables**: Inspect the underlying data.
        """
    )

    ###############################
    # 3.1 LOAD GLOBAL DATA
    ###############################
    df_item = load_item_data()
    rfm_df = load_rfm_data()
    geo_df = load_geo_data()
    st.write(f"**df_item**: {df_item.shape[0]} rows, {df_item.shape[1]} columns")
    st.write(f"**rfm_df**: {rfm_df.shape[0]} rows, {rfm_df.shape[1]} columns")
    st.write(f"**geo_df**: {geo_df.shape[0]} rows, {geo_df.shape[1]} columns")

    ###############################
    # 3.2 GLOBAL SIDEBAR FILTERS
    ###############################
    st.sidebar.header("Global Filters")
    
    # Date Filter for df_item (assumes 'order_purchase_timestamp' exists)
    if "order_purchase_timestamp" in df_item.columns:
        min_date_pandas = df_item["order_purchase_timestamp"].min()
        max_date_pandas = df_item["order_purchase_timestamp"].max()
        min_date_native = min_date_pandas.to_pydatetime()
        max_date_native = max_date_pandas.to_pydatetime()
        chosen_dates = st.sidebar.slider(
            "Filter by Purchase Date",
            min_value=min_date_native,
            max_value=max_date_native,
            value=(min_date_native, max_date_native)
        )
        date_mask = (df_item["order_purchase_timestamp"] >= chosen_dates[0]) & (df_item["order_purchase_timestamp"] <= chosen_dates[1])
        df_item = df_item[date_mask].copy()
    else:
        st.sidebar.warning("No purchase timestamp column found in df_item.")
    
    # Global Product Category Filter with Select All option (dropdown always visible)
    with st.sidebar.expander("Product Category Filter", expanded=False):
        if "product_category_name_en" in df_item.columns:
            cat_list = sorted(df_item["product_category_name_en"].dropna().unique().tolist())
            select_all_cat = st.checkbox("Select All Product Categories", value=True, key="select_all_cat")
            default_cats = cat_list if select_all_cat else cat_list[:5]
            selected_cats_global = st.multiselect("Select Product Categories", cat_list, default=default_cats, key="selected_cats")
            df_item = df_item[df_item["product_category_name_en"].isin(selected_cats_global)].copy()
        else:
            st.warning("Product category column not found in df_item.")

    
    st.sidebar.write(f"Filtered df_item: {df_item.shape[0]} rows")
    
    ###############################
    # 3.3 TABS FOR ORGANIZED VIEWS
    ###############################
    tab_item, tab_rfm, tab_geo, tab_cluster, tab_combined, tab_tables = st.tabs([
        "Item-Level Analysis", 
        "RFM-Level Analysis", 
        "Geospatial Analysis",
        "K-Means Clustering", 
        "Combined Insight", 
        "Data Tables"
    ])

    #####################################
    # TAB 1: ITEM-LEVEL ANALYSIS
    #####################################
    with tab_item:
        st.markdown("### Item-Level Analysis")
        st.markdown(
            """
            This tab presents the **row-level data** from `df_item`, which contains details 
            about individual orders, products, and categories. Use the bar chart below to view 
            the distribution of product categories among the filtered items.
            """
        )
        if "product_category_name_en" in df_item.columns:
            cat_counts = df_item["product_category_name_en"].value_counts().reset_index()
            cat_counts.columns = ["Product Category", "Count"]
            fig_cat = px.bar(
                cat_counts.head(20), 
                x="Product Category", 
                y="Count",
                title="Top 20 Product Categories (Item-Level)",
                color="Product Category"
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        st.dataframe(df_item.head(200))

    #####################################
    # TAB 2: RFM-LEVEL ANALYSIS
    #####################################
    with tab_rfm:
        st.markdown("### RFM-Level Analysis")
        st.markdown(
            """
            This tab shows the **aggregated customer-level RFM metrics**. Each row represents a single customer, 
            with calculated Recency (days since last purchase), Frequency (number of orders), and Monetary (total spend).
            Use the histograms below to explore the distributions.
            """
        )
        rfm_cols = ["recency", "frequency", "monetary"]
        for col in rfm_cols:
            if col in rfm_df.columns:
                fig = px.histogram(rfm_df, x=col, nbins=30, title=f"{col.capitalize()} Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Column {col} not found in rfm_df.")
        st.dataframe(rfm_df.head(50))

    #####################################
    # TAB 3: GEOSPATIAL ANALYSIS
    #####################################
    with tab_geo:
        st.markdown("### Geospatial Analysis")
        st.markdown(
            """
            This tab provides in-depth geospatial insights. In addition to the customer location scatter map,
            we analyze shipping routes by grouping data by seller and customer municipality. The dashboard displays:
             - Average Purchase-to-Delivery Days per route,
             - Average Lateness,
             - Average Price and Freight,
             - Total number of orders per route.
             
            Routes are visualized on an interactive map with lines drawn between seller and customer locations.
            """
        )
        
        # Merge the customer-level RFM data with geospatial data on customer_unique_id
        combined_df = pd.merge(rfm_df, geo_df, on="customer_unique_id", how="left")
        
        # Filter by Seller State using the seller_state_x column
        if "seller_state_x" in combined_df.columns:
            seller_states = sorted(combined_df["seller_state_x"].dropna().unique().tolist())
            selected_origin_state = st.selectbox("Select Origin (Seller) State", seller_states)
            combined_df = combined_df[combined_df["seller_state_x"] == selected_origin_state]
            st.write(f"Routes filtered for seller state: {selected_origin_state}")
        
        # Compute route metrics from the filtered data
        route_metrics = combined_df.groupby(["seller_code_muni", "customer_code_muni"], as_index=False).agg({
            "purchase_to_delivery_days": "mean",
            "lateness_days": "mean",
            "order_id": "count",
            "price": "mean",
            "freight_value": "mean",
            "seller_lat": "first",
            "seller_lng": "first",
            "customer_lat": "first",
            "customer_lng": "first",
            "seller_city": "first",
            "customer_city": "first"
        })
        route_metrics.rename(columns={"order_id": "order_count"}, inplace=True)
        
        # Optional filters for routes based on order count and number of routes to display
        min_orders = st.slider("Minimum Orders per Route", min_value=1, max_value=100, value=10)
        route_metrics = route_metrics[route_metrics["order_count"] >= min_orders]
        max_routes = st.slider("Maximum Number of Routes to Plot", min_value=10, max_value=200, value=50)
        if route_metrics.shape[0] > max_routes:
            route_metrics = route_metrics.nlargest(max_routes, "order_count")
        
        # Plot the filtered routes on the map using Plotly Graph Objects
        import plotly.graph_objects as go
        fig_routes = go.Figure()
        for idx, row in route_metrics.iterrows():
            # Color the route based on average delivery days
            color = "red" if row["purchase_to_delivery_days"] > 15 else "green"
            fig_routes.add_trace(go.Scattermapbox(
                mode="lines",
                lon=[row["seller_lng"], row["customer_lng"]],
                lat=[row["seller_lat"], row["customer_lat"]],
                line=dict(width=2, color=color),
                name=f"{row['seller_code_muni']} → {row['customer_code_muni']}",
                hoverinfo="text",
                text=(
                    f"Avg Delivery: {row['purchase_to_delivery_days']:.2f} days<br>"
                    f"Avg Lateness: {row['lateness_days']:.2f} days<br>"
                    f"Avg Price: ${row['price']:.2f}<br>"
                    f"Avg Freight: ${row['freight_value']:.2f}<br>"
                    f"Orders: {row['order_count']}<br>"
                    f"Seller City: {row['seller_city']}<br>"
                    f"Customer City: {row['customer_city']}"
                )
            ))
        fig_routes.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=4,
            mapbox_center={"lat": combined_df["customer_lat"].mean(), "lon": combined_df["customer_lng"].mean()},
            height=700,
            title="Route Insights: Shipping Metrics by Route"
        )
        st.plotly_chart(fig_routes, use_container_width=True)
    
        # Optionally, display a detailed table of the route metrics
        if st.checkbox("Show Detailed Route Metrics Table", key="show_route_table"):
            st.dataframe(route_metrics)
    
        # Display a scatter map of raw customer locations, enriched with product category and additional customer info
        st.markdown("#### Customer Locations by Product Category")
        if "customer_lat" in geo_df.columns and "customer_lng" in geo_df.columns:
            # Merge product category info from df_item based on customer_unique_id
            prod_cat = df_item[['customer_unique_id', 'product_category_name_en']].drop_duplicates(subset="customer_unique_id")
            geo_with_cat = pd.merge(geo_df, prod_cat, on="customer_unique_id", how="left")
            
            zoom_level = st.slider("Map Zoom Level", min_value=1, max_value=15, value=3, key="zoom_geo")
            fig_geo_raw = px.scatter_mapbox(
                geo_with_cat,
                lat="customer_lat",
                lon="customer_lng",
                hover_name="customer_unique_id",
                hover_data=["product_category_name_en", "customer_state", "customer_city", "customer_zip_code_prefix"],
                color="product_category_name_en",
                zoom=zoom_level,
                height=700,
                title="Customer Locations (Colored by Product Category)"
            )
            fig_geo_raw.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_geo_raw, use_container_width=True)
        else:
            st.error("Geospatial columns 'customer_lat' and 'customer_lng' not found in geo data.")


    #####################################
    # TAB 4: K-MEANS CLUSTERING
    #####################################
    with tab_cluster:
        st.markdown("### K-Means Clustering on RFM Metrics")
        st.markdown(
            """
            In this section, we perform K-Means clustering on the continuous RFM metrics (Recency, Frequency, Monetary)
            from the `rfm_df`. Adjust the slider below to change the number of clusters. The clustering is performed on the 
            scaled numeric data, and the resulting clusters are displayed below.
            You can choose to view the clusters in a traditional 2D scatter plot, an interactive 3D scatter plot, 
            or see the cluster assignments mapped onto customer locations.
            """
        )
        k = st.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=4, key="k_cluster")
        df_clustered = rfm_df.copy()
        df_clustered = run_kmeans(df_clustered, n_clusters=k)
        
        if "cluster" in df_clustered.columns:
            cluster_counts = df_clustered["cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            fig_cluster_bar = px.bar(
                cluster_counts,
                x="Cluster",
                y="Count",
                color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Customer Cluster Distribution"
            )
            st.plotly_chart(fig_cluster_bar, use_container_width=True)
            
            plot_type = st.radio("Select Plot Type", options=["2D Scatter", "3D Scatter", "Clustered Map"], index=0)
            if plot_type == "2D Scatter":
                st.markdown("#### 2D Scatter Plot")
                axis_options = ["recency", "frequency", "monetary"]
                x_axis = st.selectbox("Select X-Axis", axis_options, index=0, key="x_axis_cluster")
                y_axis = st.selectbox("Select Y-Axis", axis_options, index=1, key="y_axis_cluster")
                fig_scatter = px.scatter(
                    df_clustered,
                    x=x_axis,
                    y=y_axis,
                    color="cluster",
                    hover_data=["customer_unique_id"],
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title=f"Clusters: {x_axis.capitalize()} vs. {y_axis.capitalize()}"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            elif plot_type == "3D Scatter":
                st.markdown("#### 3D Scatter Plot")
                fig_3d = px.scatter_3d(
                    df_clustered,
                    x="recency",
                    y="frequency",
                    z="monetary",
                    color="cluster",
                    hover_data=["customer_unique_id"],
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title="3D Scatter Plot of RFM Clusters",
                    height=1000
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            else:  # Clustered Map
                st.markdown("#### Clustered Map")
                # Merge the cluster labels with geospatial data
                geo_clustered = pd.merge(geo_df, df_clustered[["customer_unique_id", "cluster"]], on="customer_unique_id", how="left")
                # Optional: add zoom slider if desired; here we reuse zoom_level from above or create a new one
                map_zoom = st.slider("Map Zoom Level for Clustered Map", min_value=1, max_value=15, value=3, key="zoom_cluster")
                if "customer_lat" in geo_clustered.columns and "customer_lng" in geo_clustered.columns:
                    fig_cluster_map = px.scatter_mapbox(
                        geo_clustered,
                        lat="customer_lat",
                        lon="customer_lng",
                        hover_name="customer_unique_id",
                        hover_data=["cluster"],
                        color="cluster",
                        zoom=map_zoom,
                        height=1000,
                        title="Customer Locations (K-Means Clustering Results)",
                        mapbox_style="open-street-map"
                    )
                    st.plotly_chart(fig_cluster_map, use_container_width=True)
                else:
                    st.error("Geospatial columns 'customer_lat' and 'customer_lng' not found in the data.")
        else:
            st.warning("K-Means clustering did not produce a 'cluster' column. Please check your RFM data.")

    #####################################
    # TAB 5: COMBINED INSIGHT
    #####################################
    with tab_combined:
        st.markdown("### Combined Insight: Enriched Item-Level Data with Customer RFM Metrics")
        st.markdown(
            """
            In this tab, we merge the row-level item data (`df_item`) with the aggregated customer-level RFM metrics (`rfm_df`)
            using the unique customer identifier. This merged DataFrame allows you to explore how each individual order (or item) 
            relates to the overall customer behavior. Use the sliders below to filter by Recency, Frequency, and Monetary ranges.
            
            You can also customize the scatter plot by selecting which variables appear on the x- and y-axes, and choose how to color the data points.
            """
        )
        df_combined = pd.merge(df_item, rfm_df, on="customer_unique_id", how="left")
        st.write("Combined data has", df_combined.shape[0], "rows.")
        if "recency" in df_combined.columns:
            rec_min = int(df_combined["recency"].min())
            rec_max = int(df_combined["recency"].max())
            rec_range = st.slider("Recency Range (days)", min_value=rec_min, max_value=rec_max, value=(rec_min, rec_max))
            df_combined = df_combined[(df_combined["recency"] >= rec_range[0]) & (df_combined["recency"] <= rec_range[1])]
        if "frequency" in df_combined.columns:
            freq_min = int(df_combined["frequency"].min())
            freq_max = int(df_combined["frequency"].max())
            freq_range = st.slider("Frequency Range", min_value=freq_min, max_value=freq_max, value=(freq_min, freq_max))
            df_combined = df_combined[(df_combined["frequency"] >= freq_range[0]) & (df_combined["frequency"] <= freq_range[1])]
        if "monetary" in df_combined.columns:
            mon_min = float(df_combined["monetary"].min())
            mon_max = float(df_combined["monetary"].max())
            mon_range = st.slider("Monetary Range", min_value=mon_min, max_value=mon_max, value=(mon_min, mon_max))
            df_combined = df_combined[(df_combined["monetary"] >= mon_range[0]) & (df_combined["monetary"] <= mon_range[1])]
        # Cluster Filter with Select All option
        if "cluster" in df_combined.columns:
            clusters = sorted(df_combined["cluster"].dropna().unique().tolist())
            select_all_clusters = st.checkbox("Select All Clusters", value=True, key="select_all_clusters")
            default_clusters = clusters if select_all_clusters else clusters[:2]
            selected_clusters = st.multiselect("Filter by Cluster", clusters, default=default_clusters)
            df_combined = df_combined[df_combined["cluster"].isin(selected_clusters)]
        st.write("After filtering, combined data has", df_combined.shape[0], "rows.")
        axis_options = []
        if "recency" in df_combined.columns:
            axis_options.append("recency")
        if "frequency" in df_combined.columns:
            axis_options.append("frequency")
        if "monetary" in df_combined.columns:
            axis_options.append("monetary")
        if "total_price" in df_combined.columns:
            axis_options.append("total_price")
        if not axis_options:
            st.error("No numeric columns available for plotting.")
        else:
            x_axis = st.selectbox("Select X-Axis", axis_options, index=0)
            y_axis = st.selectbox("Select Y-Axis", axis_options, index=1)
        color_options = []
        if "product_category_name_en" in df_combined.columns:
            color_options.append("product_category_name_en")
        if "cluster" in df_combined.columns:
            color_options.append("cluster")
        if color_options:
            color_by = st.selectbox("Color By", color_options, index=0)
        else:
            color_by = None
        if not df_combined.empty:
            fig_dynamic = px.scatter(
                df_combined.head(2000),
                x=x_axis,
                y=y_axis,
                color=color_by,
                title=f"{x_axis.capitalize()} vs. {y_axis.capitalize()} (Colored by {color_by})" if color_by else f"{x_axis.capitalize()} vs. {y_axis.capitalize()}",
                hover_data=["customer_unique_id", "order_id"]
            )
            st.plotly_chart(fig_dynamic, use_container_width=True)
        else:
            st.warning("No data available after filtering for plotting.")
        st.markdown("#### Filtered Combined Data (Sample)")
        st.dataframe(df_combined.head(200))
    
    #####################################
    # TAB 6: DATA TABLES
    #####################################
    with tab_tables:
        st.markdown("### Data Tables")
        st.markdown(
            """
            Below are samples of the underlying datasets after filtering:
            - **df_item**: Row-level data with product and order details.
            - **rfm_df**: Customer-level RFM metrics.
            """
        )
        st.write("**df_item** sample:")
        st.dataframe(df_item.head(200))
        st.write("**rfm_df** sample:")
        st.dataframe(rfm_df.head(50))

if __name__ == "__main__":
    main()
