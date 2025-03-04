# Geospatial Analysis: Olist E-commerce

This folder contains a **geospatial-focused** approach to analyzing the Olist e-commerce dataset. We dive into **shipping routes**, **municipality-level** patterns, **delivery times**, and more.

## Objective

- Identify geographic bottlenecks in delivery performance  
- Visualize customer/seller distributions using maps  
- Provide an interactive dashboard to filter data by time, state, or shipping metrics

## Folder Contents

```plaintext
geospatial_analysis/
├── geospatial_analysis.ipynb  # Jupyter notebook with all merges & cleaning
├── streamlit_app.py           # Streamlit dashboard code
├── cleaned_data.pkl           # Example final pickled data (if included)
├── README.md                  # This file
├── images                     # Contains images of montly order routes
├── animated_routes            # GIF of the images combined
└── ...
```

`geospatial_analysis.ipynb:`
Reads raw CSVs (orders, items, sellers, customers, geolocation).  
Merges and cleans data, creating purchase_to_delivery_days.  
Saves final DataFrame to cleaned_data.pkl.  

`streamlit_app.py:`
A Streamlit app that loads cleaned_data.pkl.  
Allows interactive filtering by state, delivery days, and more.  
Generates dynamic charts (histograms, box plots) and a map-based view.  

## Usage

This subfolder focuses on the geospatial analysis of Olist’s dataset, including:

- **Merging** data from multiple CSVs
- **Creating** `purchase_to_delivery_days`
- **Generating** route-level metrics
- **Interactive** Streamlit dashboard

## How to Use

1. **Prerequisites**  
   - Python 3.8+  
   - Libraries: pandas, geopandas, folium, streamlit, plotly, pygeobr, etc.  
   - See `requirements.txt` in the project root.

2. **Run the Notebook**  
   - `geospatial_analysis.ipynb` merges raw data and saves a final `cleaned_data.pkl` (or `.csv`).  
   - Open the notebook in JupyterLab or VSCode.  
   - Execute all cells to produce the final dataset.

3. **Launch the Streamlit Dashboard**  
   - After the notebook is done, you'll have a `cleaned_data.pkl`.  
   - From the command line:
     ```bash
     cd geospatial_analysis
     streamlit run streamlit_app.py
     ```
   - Your browser will open `http://localhost:8501`.  
   - Use the sidebar filters (min/max days, state selection, etc.) to explore shipping data in real-time.

4. **Folder Contents**  
   - `geospatial_analysis.ipynb`: Main Jupyter notebook for merges, cleaning, advanced geospatial steps.  
   - `streamlit_app.py`: Interactive dashboard code.  
   - `cleaned_data.pkl`: Final, cleaned dataset for quick loading in the dashboard.  
   - `README.md`: This file, with usage details.

