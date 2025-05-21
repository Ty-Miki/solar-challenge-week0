import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
from app.utils import (load_and_combine_each_country_data,
                       boxplot_by_country,
                       write_summary_table,
                       per_country_ghi_trends)

country_file_name_pairs = {"Benin": "data/benin-malanville_clean.csv", 
                           "Togo": "data/togo-dapaong_qc_clean.csv", 
                           "Sierra Leone": "data/sierraleone-bumbuna_clean.csv"}

combined_df = load_and_combine_each_country_data(country_file_name_pairs)

tab1, tab2, tab3 = st.tabs(["📊 Country-wise Boxplot", "📋 Summary Table", "📈 Per country GHI Trends"])
if combined_df is not None:
    with tab1:
        st.subheader("📊 Country-wise Boxplot")
        boxplot_by_country(combined_df)

    with tab2:
        st.subheader("📋 Summary Table")
        write_summary_table(combined_df, group_col="country")
        
        st.subheader("📋 Conclusion")
        st.write("""The summary table shows Benin exhibits a significantly higher mean GHI and DNI which shows the place gets the highest amount of sun irradiance.
                 One the other hand""")
    
    with tab3:
        st.subheader("📈 Benin GHI Trends")
        per_country_ghi_trends(filename="data/benin-malanville_clean.csv")

        st.subheader(" 📈 Togo GHI Trends")
        per_country_ghi_trends(filename="data/togo-dapaong_qc_clean.csv")

        st.subheader(" 📈 Sierra Leone GHI Trends")
        per_country_ghi_trends(filename="data/sierraleone-bumbuna_clean.csv")
