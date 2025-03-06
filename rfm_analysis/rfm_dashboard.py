import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

###########################################
# 1. DATA LOADING FUNCTIONS
###########################################

@st.cache_data
def load_item_data():
    """
    Load the row-level item DataFrame.
    This DataFrame includes order details, product categories, and other item-level info.
    """
    df_item = pd.read_pickle("df_item.pkl")
    return df_item

@st.cache_data
def load_rfm_data():
    """
    Load the customer-level RFM DataFrame.
    Contains one row per customer with recency, frequency, and monetary metrics.
    """
    rfm_df = pd.read_pickle("rfm_df.pkl")
    return rfm_df

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
    # Set the page configuration for a professional wide layout
    st.set_page_config(page_title="Advanced RFM + K-Means Dashboard", layout="wide")

    st.title("Olist E-commerce Dashboard: Advanced RFM + K-Means Segmentation")
    st.markdown(
        """
        This dashboard integrates two levels of data:
        - **Item-Level Data (df_item)**: Detailed order and product category information.
        - **Customer-Level RFM Data (rfm_df)**: Aggregated metrics for each customer (Recency, Frequency, Monetary).
        
        Use the filters in the sidebar to narrow down the dataset by date and product categories.
        Navigate the tabs below for a comprehensive view:
        - **Item-Level Analysis**: Explore product categories and item-level details.
        - **RFM-Level Analysis**: Visualize the distributions of RFM metrics.
        - **K-Means Clustering**: Run K-Means segmentation on the RFM data.
        - **Combined Insight**: See item-level data with corresponding RFM metrics.
        - **Data Tables**: Inspect the underlying data.
        """
    )

    ###############################
    # 3.1 LOAD GLOBAL DATA
    ###############################
    df_item = load_item_data()
    rfm_df = load_rfm_data()
    st.write(f"**df_item**: {df_item.shape[0]} rows, {df_item.shape[1]} columns")
    st.write(f"**rfm_df**: {rfm_df.shape[0]} rows, {rfm_df.shape[1]} columns")

    ###############################
    # 3.2 GLOBAL SIDEBAR FILTERS
    ###############################
    st.sidebar.header("Global Filters")
    
    # Date Filter for df_item (assumes 'order_purchase_timestamp' exists)
    if "order_purchase_timestamp" in df_item.columns:
        min_date_pandas = df_item["order_purchase_timestamp"].min()
        max_date_pandas = df_item["order_purchase_timestamp"].max()
        # Convert to native datetime objects for compatibility
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
    
    # Product Category Filter with "Select All" Option
    if "product_category_name_en" in df_item.columns:
        cat_list = sorted(df_item["product_category_name_en"].dropna().unique().tolist())
        select_all = st.sidebar.checkbox("Select All Product Categories", value=True)
        if select_all:
            selected_cats = cat_list
        else:
            selected_cats = st.sidebar.multiselect("Select Product Categories", cat_list, default=cat_list[:5])
        df_item = df_item[df_item["product_category_name_en"].isin(selected_cats)].copy()
    else:
        st.sidebar.warning("Product category column not found in df_item.")
    
    st.sidebar.write(f"Filtered df_item: {df_item.shape[0]} rows")
    
    ###############################
    # 3.3 TABS FOR ORGANIZED VIEWS
    ###############################
    tab_item, tab_rfm, tab_cluster, tab_combined, tab_tables = st.tabs([
        "Item-Level Analysis", 
        "RFM-Level Analysis", 
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
            This tab shows the **aggregated customer-level RFM metrics**. Here, each row represents a single customer, 
            with calculated Recency (days since last purchase), Frequency (number of orders), and Monetary (total spend).
            Use the histograms below to explore the distributions.
            """
        )
        rfm_cols = ["recency", "frequency", "monetary"]
        for col in rfm_cols:
            if col in rfm_df.columns:
                fig = px.histogram(
                    rfm_df, 
                    x=col, 
                    nbins=30, 
                    title=f"{col.capitalize()} Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Column {col} not found in rfm_df.")
        st.dataframe(rfm_df.head(50))

    #####################################
    # TAB 3: K-MEANS CLUSTERING
    #####################################
    with tab_cluster:
        st.markdown("### K-Means Clustering on RFM Metrics")
        st.markdown(
            """
            In this section, we perform **K-Means clustering** on the continuous RFM metrics (Recency, Frequency, Monetary) 
            from the `rfm_df`. Adjust the slider below to change the number of clusters. The clustering is performed on the 
            scaled numeric data, and the resulting clusters are displayed with a distinct color palette.
            """
        )
        # This slider is inside the tab, so it appears only when this tab is active.
        k = st.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=4)
        df_clustered = rfm_df.copy()
        df_clustered = run_kmeans(df_clustered, n_clusters=k)

        # Display cluster distribution
        if "cluster" in df_clustered.columns:
            cluster_counts = df_clustered["cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            fig_cluster = px.bar(
                cluster_counts, 
                x="Cluster", 
                y="Count",
                color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Customer Cluster Distribution"
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

            st.markdown("#### 2D Scatter Plot of Clusters")
            axis_options = ["recency", "frequency", "monetary"]
            x_axis = st.selectbox("Select X-Axis", axis_options, index=0, key="x_axis")
            y_axis = st.selectbox("Select Y-Axis", axis_options, index=1, key="y_axis")
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
        else:
            st.warning("K-Means clustering did not produce a 'cluster' column.")

    #####################################
    # TAB 4: COMBINED INSIGHT
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
        
        # Merge the filtered df_item with rfm_df (each row now represents an item enriched with customer RFM values)
        df_combined = pd.merge(df_item, rfm_df, on="customer_unique_id", how="left")
        st.write("Combined data has", df_combined.shape[0], "rows.")
        
        # --- Interactive RFM Filters ---
        # Recency filter
        if "recency" in df_combined.columns:
            rec_min = int(df_combined["recency"].min())
            rec_max = int(df_combined["recency"].max())
            rec_range = st.slider("Recency Range (days)", min_value=rec_min, max_value=rec_max, value=(rec_min, rec_max))
            df_combined = df_combined[(df_combined["recency"] >= rec_range[0]) & (df_combined["recency"] <= rec_range[1])]
        
        # Frequency filter
        if "frequency" in df_combined.columns:
            freq_min = int(df_combined["frequency"].min())
            freq_max = int(df_combined["frequency"].max())
            freq_range = st.slider("Frequency Range", min_value=freq_min, max_value=freq_max, value=(freq_min, freq_max))
            df_combined = df_combined[(df_combined["frequency"] >= freq_range[0]) & (df_combined["frequency"] <= freq_range[1])]
        
        # Monetary filter
        if "monetary" in df_combined.columns:
            mon_min = float(df_combined["monetary"].min())
            mon_max = float(df_combined["monetary"].max())
            mon_range = st.slider("Monetary Range", min_value=mon_min, max_value=mon_max, value=(mon_min, mon_max))
            df_combined = df_combined[(df_combined["monetary"] >= mon_range[0]) & (df_combined["monetary"] <= mon_range[1])]
        
        # Optionally filter by cluster if available
        if "cluster" in df_combined.columns:
            clusters = sorted(df_combined["cluster"].unique().tolist())
            selected_clusters = st.multiselect("Filter by Cluster", clusters, default=clusters)
            df_combined = df_combined[df_combined["cluster"].isin(selected_clusters)]
        
        st.write("After filtering, combined data has", df_combined.shape[0], "rows.")
        
        # --- Dynamic Scatter Plot Settings ---
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
        
        # Choose color dimension (by product category or cluster)
        color_options = []
        if "product_category_name_en" in df_combined.columns:
            color_options.append("product_category_name_en")
        if "cluster" in df_combined.columns:
            color_options.append("cluster")
        if color_options:
            color_by = st.selectbox("Color By", color_options, index=0)
        else:
            color_by = None
        
        # Create a dynamic scatter plot using a sample of 2000 rows for performance
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
        
        # Show a data table for reference
        st.markdown("#### Filtered Combined Data (Sample)")
        st.dataframe(df_combined.head(200))

    #####################################
    # TAB 5: DATA TABLES
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
