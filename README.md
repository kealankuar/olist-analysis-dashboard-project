# Olist Geospatial & RFM Analysis Dashboard

## Project Overview

This project analyzes customer purchase behavior using **Recency, Frequency, and Monetary (RFM) Analysis** while incorporating **Geospatial Data** to provide location-based insights. The analysis was conducted in three phases:

1. **Geospatial Analysis** - Mapping customer locations, purchase distribution, and shipping trends.
2. **RFM Analysis** - Customer segmentation based on transaction behavior.
3. **Integrated Dashboard** - A **Streamlit dashboard** that combines geospatial and RFM insights for a comprehensive view.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [How to Run the Analysis](#how-to-run-the-analysis)
- [Dashboard Features](#dashboard-features)
- [File Structure](#file-structure)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Dataset Description

This project uses data from the **Olist E-Commerce Dataset**, which contains transaction records from a Brazilian e-commerce platform. The key datasets used:

| **File**             | **Description** |
|----------------------|----------------|
| `df_item.pkl`       | Row-level order data, including product category and purchase details from rfm_analysis. |
| `rfm_df.pkl`        | Customer-level RFM analysis results from rfm_analysis. |
| `cleaned_data.pkl`  | Geospatial data, including customer locations and shipping trends from geospatial-analysis. |

Each dataset was preprocessed separately before being merged into a unified dataset for visualization.

---

## Installation

Ensure you have Python 3.8+ installed. It is recommended to create a virtual environment.

### 1. Clone the Repository

```bash
git clone https://github.com/kealankuar/olist-analysis-dashboard-project.git
cd olist-dashboard-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt or
pip install pandas numpy matplotlib seaborn plotly geopandas pygeobr scikit-learn streamlit
```
## How to Run the Analysis

### 1. Running the Geospatial Analysis

If you want to analyse geospatial trends separately, open and run:
```bash
jupyter notebook geospatial_analysis.ipynb`
```
This notebook includes:
  Mapping customer locations.   
  Clustering and spatial patterns.   
  Shipping delay trends by region.    

### 2. Running the RFM Analysis
For customer segmentation based on transaction behaviour, open and run:
```bash
jupyter notebook rfum_analysis.ipynb
```
This notebook computes:
  Recency (days since last purchase).   
  Frequency (number of purchases).   
  Monetary (total amount spent).  

### 3. Running the Integrated Dashboard
To view the combined **Geospatial + RFM dashboard**, run:
```bash
streamlit run combined_dashboard.py
```
The dashboard will open at http://localhost:8501/ on your default browser.

## Dashboard Features
The **Streamlit Dashboard** provides interactive visualisations of customer behaviour.  
It is divided into multiple tabs:
  1. **RFM Analysis**:
     1. Histograms showing Recency, Frequency and Monetary distributions.
     2. Identify high-value customers based on purchase history.
  2. **K-Means Clustering**:
     1. Cluster customers based on RFM Scores
     2. 2D and 3D scatter plots to visualise different segments
     3. Clusters plotted on a map
  3. **Geospatial Analysis**:
     1. Interactive maps showing customer distribution across Brazil
     2. Purchase behaviour analysis by region
     3. Shipping time analysis using geospatial data.
  4. **Combined Insight**:
     1. Enriched Item-Level Data combined with Customer RFM Metrics.
  5. **Data Tables**:
     1. View sample data

## File Structure

📂 olist-dashboard-project   
│── 📂 rfm_analysis         
│   ├── rfm_analysis.ipynb        # RFM segmentation notebook  
│   ├── df_item.pkl               # Row-level order data  
│   ├── rfm_df.pkl                # Customer-level RFM analysis  
│  
│── 📂 geospatial_analysis  
│   ├── geospatial_analysis.ipynb # Geospatial mapping and shipping trends  
│   ├── cleaned_data.pkl          # Processed geospatial data  
│  
│── 📂 combined_analysis  
│   ├── dashboard.py     # Streamlit dashboard (RFM + Geospatial)  
│  
│── 📄 requirements.txt           # Python dependencies  
│── 📄 README.md                  # Documentation  


## Future Work

1. **Machine Learning for Customer Segmentation**
   - Explore advanced clustering (DBSCAN, Hierarchical Clustering) to refine customer groups.
2. **Real-Time Data Updates**
   - Enhance the dashboard by integrating real-time data streaming
3. **Predicitive Analytics**
   - Use historical data to predict future purchasing behaviour and optimise marketing strategies

## Acknowledgments
1. Kaggle for providing open-source e-commerce data
    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data
2. Geobr for Brazilian geospatial datasets.
3. Streamlit for making interactive visualisations easy

