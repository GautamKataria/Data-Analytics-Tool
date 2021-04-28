import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from PCA_function import do_pca
from PCA_function import pca_reduction
from PCA_function import pca_chart
from PCA_function import corr_heatmap
from PCA_function import Scatter_plot
from PCA_function import Box_plot
from PCA_function import Line_chart
from PCA_function import Donut
from PCA_function import Hist
from PCA_function import Bar


st.set_page_config(layout="wide")

scatter_column, settings_column = st.beta_columns((5,1))

scatter_column.title("Data Analysis Tool")
settings_column.title("Settings")
uploaded_file = settings_column.file_uploader("Choose a csv file to get started")

if uploaded_file != None:
    try:
        data = pd.read_csv(uploaded_file)

    except Exception as e:
        scatter_column.write("only upload a csv file")

    scatter_column.write("Dataset")
    scatter_column.write(data.head())

    st.sidebar.title("Choose what do you want to do with your data")
    option = st.sidebar.selectbox("select EDA or PCA", ('< Click to select >',"EDA", "PCA"))


    num_cols1 = []
    non_num_cols1 = []
    for i in data.columns:
        if data[i].dtype == np.dtype("float64") or data[i].dtype == np.dtype("int64"):
            num_cols1.append(data[i])
        else:
            non_num_cols1.append(data[i])
    if len(non_num_cols1) == 0:
        data["sample"] = "obj"

    num_data = pd.concat(num_cols1, axis=1)
    non_num_data = pd.concat(non_num_cols1, axis=1)


    if (option=="PCA"):

        scatter_column.write("### Any missing values will be filled with (mean-numerical, mode - categorical)")
        colorway = st.sidebar.selectbox("Select Different colorways",
                                        options=["Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Light24",
                                                 "Set1", "Pastel1", "Dark2", "Set2", "Pastel2", "Set3",
                                                 "Antique", "Bold", "Pastel", "Prism", "Safe", "Vivid"])
        pca_data, cat_cols, pca_cols, num_cols = do_pca(data)

        categorical_variable = settings_column.selectbox("Legend Variable select", options=cat_cols)
        cat_cols.remove(categorical_variable)
        categorical_variable2 = settings_column.selectbox("Hover Variable select", options=cat_cols)

        PCA1 = settings_column.selectbox("First Principal Component", options=pca_cols)
        pca_cols.remove(PCA1)
        PCA2 = settings_column.selectbox("Second Principal Component", options=pca_cols)

        for i in list(pca_data.columns):
            if pca_data[i].dtype == np.dtype("float64") or pca_data[i].dtype == np.dtype("int64"):
                pass
            else:
                if pca_data[i].isnull().sum() != 0:
                    pca_data[i] = pca_data[i].fillna(pca_data[i].mode().iloc[0])

        chart1 = px.scatter(data_frame=pca_data, x=PCA1, y=PCA2, color=categorical_variable, template="simple_white",
                       hover_data=[categorical_variable2], height=800,color_discrete_sequence=getattr(px.colors.qualitative,colorway))

        scatter_column.plotly_chart(chart1, use_container_width=True)

        Target_column = settings_column.selectbox("Target select", options=num_cols)
        tot, var_exp, x_std = pca_reduction(data, Target_column)
        scatter_column.subheader("Principle components and the amount of information they have in regards to target")

        figure123 = px.bar(data_frame=var_exp,color_discrete_sequence=getattr(px.colors.qualitative,colorway))
        figure123.update_xaxes(title_text='Principal components')
        figure123.update_yaxes(title_text='Explained variance')
        scatter_column.plotly_chart(figure123)

        pca = pca_chart(x_std)
        scatter_column.subheader(
            "Chart showing % of information regarding target with each additional Principal component")
        figure234 = px.line(np.cumsum(pca.explained_variance_ratio_),color_discrete_sequence=getattr(px.colors.qualitative,colorway))
        figure234.update_xaxes(title_text='number of principal components')
        figure234.update_yaxes(title_text='Cumulative explained variance')
        scatter_column.plotly_chart(figure234)

    elif(option == "EDA"):
        scatter_column.write(" ## Data info")
        scatter_column.write(" #### null data")
        scatter_column.write(data.isnull().sum())
        scatter_column.write(" #### statistical analysis")
        scatter_column.write(data.describe())

        cat_cols_final = list(non_num_data.columns)
        num_cols_final = list(num_data.columns)
        all_cols = list(data.columns)
        scatter_column.write("## Visualisations:")
        graphs = st.sidebar.selectbox("Select type of graph", (" < Click to select > ","Boxplot","Scatterplot","Barchart","Histogram","Donutchart", "Linechart",
                                                      "Correlation_heatmap"))
        colorway = st.sidebar.selectbox("Select Different colorways",
                                            options=["Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Light24",
                                                     "Set1", "Pastel1", "Dark2", "Set2", "Pastel2", "Set3",
                                                     "Antique", "Bold", "Pastel", "Prism", "Safe", "Vivid"])

        if(graphs == "Barchart"):
            scatter_column.write("### Barchart")
            select_categorical_variable = scatter_column.selectbox("X-Axis variable select", options=all_cols)
            select_numerical_variable = scatter_column.selectbox("Y-Axis variable select", options=num_cols_final)
            color_variable = scatter_column.selectbox("Select Legend Variable", options=cat_cols_final)




            if data[color_variable].isnull().sum() != 0:
                data[color_variable] = data[color_variable].fillna(data[color_variable].mode().iloc[0])
                scatter_column.write("As there were some nan values present in color variable, we filled it in by mode")
            fig = Bar(data, select_categorical_variable, select_numerical_variable, color_variable, colorway)
            scatter_column.plotly_chart(fig)


        elif(graphs=="Histogram"):
            scatter_column.write("### Histogram")
            select_variable = scatter_column.selectbox("Select variable", options=all_cols)
            color_variable = scatter_column.selectbox("Select Legend variable", options=cat_cols_final)

            if data[color_variable].isnull().sum() != 0:
                data[color_variable] = data[color_variable].fillna(data[color_variable].mode().iloc[0])
                scatter_column.write("As there were some nan values present in Legend variable, we filled it in by mode")

            fig = Hist(data, select_variable,color_variable,colorway)
            scatter_column.plotly_chart(fig)


        elif(graphs == "Donutchart"):
            scatter_column.write("### Donutchart")
            select_numerical_variable = scatter_column.selectbox("Select Value variable",options=num_cols_final)
            select_categorical_variable = scatter_column.selectbox("Select legend variable",
                                                                   options=cat_cols_final)

            fig = Donut(data,select_numerical_variable,select_categorical_variable,colorway)
            scatter_column.plotly_chart(fig)


        elif(graphs=="Linechart"):
            scatter_column.write("### Linechart")
            select_numerical_variable = scatter_column.selectbox("X-Axis variable select", options=num_cols_final)
            select_num2_variable = scatter_column.selectbox("Y-Axis variable select",
                                                                   options=num_cols_final)
            color_variable = scatter_column.selectbox("Select Legend variable", options=cat_cols_final)

            if data[color_variable].isnull().sum() != 0:
                data[color_variable] = data[color_variable].fillna(data[color_variable].mode().iloc[0])
                scatter_column.write("As there were null values in the color variable, they have been filled in by the mode")
            fig =Line_chart(data, select_numerical_variable, select_num2_variable, color_variable,colorway)
            scatter_column.plotly_chart(fig)


        elif(graphs=="Boxplot"):
            scatter_column.write("### Boxplot")
            select_numerical_variable = scatter_column.selectbox("numerical variable (Y-Axis) select", options=num_cols_final)
            select_categorical_variable = scatter_column.selectbox("Categorical variable (X-Axis) select",
                                                                   options=cat_cols_final)
            color_variable = scatter_column.selectbox("Select Legend variable", options=cat_cols_final)

            if data[color_variable].isnull().sum() != 0:
                data[color_variable] = data[color_variable].fillna(data[color_variable].mode().iloc[0])
                scatter_column.write(
                    "As there were null values in the color variable, they have been filled in by the mode")

            fig = Box_plot(data,select_categorical_variable,select_numerical_variable,color_variable,colorway)
            scatter_column.plotly_chart(fig)


        elif(graphs=="Scatterplot"):
            scatter_column.write("### Scatterplot")
            select_numerical_variable = scatter_column.selectbox("X-Axis variable select", options=num_cols_final)
            select_numerical_variable2 = scatter_column.selectbox("Y-Axis variable select", options=num_cols_final)
            select_categorical_variable = scatter_column.selectbox("Legend variable select",
                                                                   options=cat_cols_final)
            select_hover = scatter_column.selectbox("Select hover data", options=all_cols)

            fig = Scatter_plot(data,select_numerical_variable, select_numerical_variable2, select_categorical_variable, select_hover,colorway)
            scatter_column.plotly_chart(fig)


        elif(graphs=="Correlation_heatmap"):
            scatter_column.write("### Correlation Heatmap")

            fig = corr_heatmap(data, num_data)
            scatter_column.plotly_chart(fig)

        scatter_column.write("##### NOTE: double click to zoom back to original scale graph")
    else:
        pass
