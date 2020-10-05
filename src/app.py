#!/usr/bin/env python3
 
##########################################################
# encoding: utf                                          #
# Copyright (c) Jesper Vang <jesper_vang@me.com>         #
# MIT Licence. See http://opensource.org/licenses/MIT    #
# Created on 11 Sep 2020                                 #
# Version:	0.0.1                                        #
##########################################################

import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import SessionState
from pathlib import Path
from report1 import display_reports
import base64
from PCA.dim_red_v2 import PCA_run
st.set_option('deprecation.showPyplotGlobalUse', False)

ABOUT = "About"
REPORTS = "Reports"
DATA = "Data"
CONFIG = "Config"


def main():
    PAGES = {
        ABOUT: display_about,
        REPORTS: display_reports,
        DATA: dispaly_data,
        CONFIG: display_config,
        
    }

    st.sidebar.header("Hepatitis")
    page = st.sidebar.radio("Select page", options=list(PAGES.keys()))
    display_sidebar_settings()

    PAGES[page]()

def display_sidebar_settings():
    st.sidebar.markdown("---")
    st.sidebar.markdown("ℹ️ ** Details **")
    desc_check = st.sidebar.checkbox("📃 Dataset Description")
    desc_markdown = read_markdown_file("src/desc_markdown.md")        

    if desc_check:
        st.sidebar.markdown(desc_markdown, unsafe_allow_html=True)




@st.cache
def load_data():
    return pd.read_csv(
        "src/data/HCV-Egy-Data.csv"
    ).reset_index(drop=True)

@st.cache
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def display_config():
    pass

#def display_reports():
    return display_reports


def display_about():
    intro_markdown = read_markdown_file("src/intro.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)
    st.markdown("---")

def dispaly_data():
     # Header
    st.title("PCA and (Data exploration - in the final version)")
    # For univariate distributions
    # histogram to better understand
    
    st.header("PCA")
    df = load_data()
    report1_check = st.checkbox("🤞 Run PCA on data")
    if report1_check:
        PCA_run(df)
        #st.plotly_chart(fig, use_container_width=True)

        st.pyplot()
        st.markdown("---")
        
        
    
       
    #hist_x = st.selectbox("Histogram variable", options=df.columns, index=df.columns.get_loc("Gender"))
    #hist_bins = st.slider(label="Histogram bins", min_value=5, max_value=50, value=25, step=1)
    #hist_cats = df[hist_x].sort_values().unique()
    #hist_fig = px.histogram(df, x=hist_x, nbins=hist_bins, title="Histogram of " + hist_x,
    #                        template="plotly_white", category_orders={hist_x: hist_cats})
    #st.write(hist_fig)



    # boxplots
    st.header("Boxplot")
    st.subheader("With a categorical variable - Fever, Gender or BMI")
    box_x = st.selectbox("Boxplot variable", options=df.columns, index=df.columns.get_loc("WBC"))
    box_cat = st.selectbox("Categorical variable", ["Fever", "Gender", "BMI"], 0)
    box_fig = px.box(df, x=box_cat, y=box_x, title="Box plot of " + box_x,
                            template="plotly_white", category_orders={"pos_simple": ["PG", "SG", "SF", "PF", "C"]})
    st.write(box_fig)

    # # min filter
    # st.header("Correlations")
    # corr_x = st.selectbox("Correlation - X variable", options=df.columns, index=df.columns.get_loc("Gender"))
    # corr_y = st.selectbox("Correlation - Y variable", options=df.columns, index=df.columns.get_loc("WBC"))
    # corr_col = st.radio("Correlation - color variable", options=["Gender", "Fever", "Headache"], index=1)
    # corr_filt = st.selectbox("Filter variable", options=df.columns, index=df.columns.get_loc("WBC"))
    # min_filt = st.number_input("Minimum value", value=6, min_value=0)
    # tmp_df = df[df[corr_filt] > min_filt]
    # fig = px.scatter(tmp_df, x=corr_x, y=corr_y, template="plotly_white", render_mode='webgl',
    #                  color=corr_col, hover_data=['WBC', 'RBC', 'Age', 'Gender'], color_continuous_scale=px.colors.sequential.OrRd,
    #                  category_orders={"pos_simple": ["PG", "SG", "SF", "PF", "C"]})
    # fig.update_traces(mode="markers", marker={"line": {"width": 0.4, "color": "slategrey"}})
    # st.subheader("Filtered scatterplot and dataframe")
    # st.write(fig)
    # st.write(tmp_df)

    # correlation heatmap
    hmap_params = st.multiselect("Select parameters to include on heatmap", options=list(df.columns), default=list(df.columns))
    hmap_fig = px.imshow(df[hmap_params].corr())
    st.write(hmap_fig)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    return base64.b64encode(img_bytes).decode()

header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes("src/image/header.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)


if __name__ == "__main__":
    main()