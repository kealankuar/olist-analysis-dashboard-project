import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_rfm_data():
    # Reads the pickled RFM DataFrame created in the notebook
    df = pd.read_pickle("rfm_data.pkl")
    return df

def main():
    st.title("Olist RFM Analysis Dashboard")

    # 1) Load data
    rfm_df = load_rfm_data()
    st.write(f"RFM dataset loaded! {rfm_df.shape[0]} rows, {rfm_df.shape[1]} columns.")
    
    # 2) Basic Filters
    max_recency = int(rfm_df["recency"].max())
    recency_cutoff = st.slider("Max Recency (days)", 0, max_recency, 30)

    min_freq = int(rfm_df["frequency"].min())
    max_freq = int(rfm_df["frequency"].max())
    freq_range = st.slider("Frequency Range", min_freq, max_freq, (min_freq, max_freq))

    min_monetary = float(rfm_df["monetary"].min())
    max_monetary = float(rfm_df["monetary"].max())
    monetary_range = st.slider("Monetary Range", float(min_monetary), float(max_monetary), (float(min_monetary), float(max_monetary)))
    
    # Filter the DF
    df_filtered = rfm_df[
        (rfm_df["recency"] <= recency_cutoff) &
        (rfm_df["frequency"] >= freq_range[0]) & (rfm_df["frequency"] <= freq_range[1]) &
        (rfm_df["monetary"] >= monetary_range[0]) & (rfm_df["monetary"] <= monetary_range[1])
    ].copy()
    
    st.write(f"After filtering: {df_filtered.shape[0]} customers.")
    
    # 3) KPI Cards
    col1, col2, col3 = st.columns(3)
    mean_recency = df_filtered["recency"].mean() if len(df_filtered) else 0
    mean_frequency = df_filtered["frequency"].mean() if len(df_filtered) else 0
    mean_monetary = df_filtered["monetary"].mean() if len(df_filtered) else 0
    
    col1.metric("Avg Recency (days)", f"{mean_recency:.2f}")
    col2.metric("Avg Frequency", f"{mean_frequency:.2f}")
    col3.metric("Avg Monetary", f"{mean_monetary:.2f}")

    # 4) Distribution Charts
    st.subheader("Recency Distribution")
    fig_r = px.histogram(df_filtered, x="recency", nbins=30, title="Recency Histogram")
    st.plotly_chart(fig_r, use_container_width=True)

    st.subheader("Frequency Distribution")
    fig_f = px.histogram(df_filtered, x="frequency", nbins=30, title="Frequency Histogram")
    st.plotly_chart(fig_f, use_container_width=True)

    st.subheader("Monetary Distribution")
    fig_m = px.histogram(df_filtered, x="monetary", nbins=30, title="Monetary Histogram")
    st.plotly_chart(fig_m, use_container_width=True)

    # 5) Segment Chart (if RFM_Segment is present)
    if "RFM_Segment" in df_filtered.columns:
        seg_counts = df_filtered["RFM_Segment"].value_counts().reset_index()
        seg_counts.columns = ["RFM_Segment", "count"]
        fig_seg = px.bar(seg_counts, x="RFM_Segment", y="count", title="RFM Segment Counts")
        st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.write("No RFM_Segment column found.")

    # 6) Data Table
    with st.expander("Show Data Table"):
        st.dataframe(df_filtered.head(200))

if __name__ == "__main__":
    main()
