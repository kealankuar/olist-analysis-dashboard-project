# Olist RFM Analysis

This branch focuses on performing a detailed RFM (Recency, Frequency, Monetary) analysis on the Olist e-commerce dataset. The goal is to segment customers based on their purchasing behavior and overall spending patterns, which can be used for targeted marketing and further customer analysis.

## Project Overview

**RFM Analysis** is a proven technique to evaluate customer value based on:
- **Recency (R):** How recently did a customer make a purchase?
- **Frequency (F):** How often does a customer purchase?
- **Monetary (M):** How much does the customer spend?

In this project, we merge multiple Olist datasets to compute these metrics at a customer level.

## Repository Structure (rfm-analysis Branch)

```plaintext
.
├── rfm_analysis.ipynb         # Main Jupyter Notebook for RFM analysis
├── df_item.pkl                # Pickled Dataframe with row-level item details
├── rfm_df.pkl                 # Pickled Dataframe with aggregated customer-level RFM metrics
├── README.md                  # This README file
├── rfm_dashboard              # Pyton code for the streamlit dashboard
└── ...
```

## How to Run the Analysis

1. **Install Dependencies**

   Ensure you have Python 3.8+ installed. Then install the required packages. For example:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter streamlit
   ```

2. **Obtain the Data**
   ```
   olist_orders_dataset.csv
   olist_order_items_dataset.csv
   olist_customers_dataset.csv
   olist_products_dataset.csv
   product_category_name_translation.csv
   ```
3. **Run the RFM Analysis Notebook**

   1. Open the notebook ```rfm_analysis.ipynb``` using Jupyter Notebook or any other suitable IDE
   2. Run all cells sequentially
   3. Verify that both df_item.pkl and rfm_df.pkl are created in the project directory

4. **Run the Dashboard**

   1. Ensure that the notebook has run and the pickled files are present
   2. Run the streamlit dashboard using the following command
      ```bash
      streamlit run rfm_dashboard.py
      ```
   3. Your default web browser will open a new tab at http://localhost:8501 where you can interact with the dashboard

