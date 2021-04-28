import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def do_pca(data):

    num_cols=[]
    non_num_cols = []


    for i in data.columns:
        if data[i].dtype == np.dtype("float64") or data[i].dtype == np.dtype("int64"):
            num_cols.append(data[i])
        else:
            non_num_cols.append(data[i])
    num_data = pd.concat(num_cols, axis=1)
    non_num_data = pd.concat(non_num_cols, axis=1)

    for cols in list(num_data.columns):
        if num_data[cols].isnull().sum() != 0:
            num_data[cols] = num_data[cols].fillna(num_data[cols].mean())

    for cols in list(non_num_data.columns):
        if non_num_data[cols].isnull().sum() != 0:
            non_num_data[cols] = non_num_data[cols].fillna(non_num_data[cols].mode().iloc[0])

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(num_data)
    pca = PCA()

    pca_data = pca.fit_transform(scaled_data)
    pca_data = pd.DataFrame(pca_data)

    new_cols_names = ["PCA_" + str(i) for i in range(1, len(pca_data.columns) +1)]

    list(pca_data.columns)
    cols_mapper = dict(zip(list(pca_data.columns), new_cols_names))

    pca_data = pca_data.rename(columns = cols_mapper)

    output = pd.concat([data, pca_data], axis =1)

    return output, list(non_num_data.columns), new_cols_names, list(num_data.columns)

def pca_reduction(data, target):

    num_cols=[]
    non_num_cols = []

    for i in data.columns:
        if data[i].dtype == np.dtype("float64") or data[i].dtype == np.dtype("int64"):
            num_cols.append(data[i])
        else:
            non_num_cols.append(data[i])

    num_data = pd.concat(num_cols, axis=1)
    non_num_data = pd.concat(non_num_cols, axis=1)

    for cols in list(num_data.columns):
        if num_data[cols].isnull().sum() != 0:
            num_data[cols] = num_data[cols].fillna(num_data[cols].mean())


    for cols in list(non_num_data.columns):
        if non_num_data[cols].isnull().sum() != 0:
            non_num_data[cols] = non_num_data[cols].fillna(non_num_data[cols].mode().iloc[0])

    cols = list(num_data.columns)
    cols.insert(0, cols.pop(cols.index(target)))
    num_data = num_data.reindex(columns = cols)
    x = num_data.iloc[:, 1:].values
    y = num_data.iloc[:, 0].values

    scaler1 = StandardScaler()

    scaled_data1 = scaler1.fit_transform(x)
    cov_mat = (np.cov(scaled_data1.T))

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]

    return tot,var_exp,scaled_data1

def pca_chart(x_std):

    pca = PCA().fit(x_std)
    return pca

def corr_heatmap(data, num_data):
    for cols in list(num_data.columns):
        if data[cols].isnull().sum() != 0:
            data[cols] = data[cols].fillna(data[cols].mean())

    corr = data.corr()
    fig = px.imshow(corr)
    return fig

def Scatter_plot(data,select_numerical_variable, select_numerical_variable2, select_categorical_variable, select_hover,colorway):

    fig = px.scatter(data,x=select_numerical_variable, y= select_numerical_variable2, color = select_categorical_variable, hover_data = [select_hover],color_discrete_sequence=getattr(px.colors.qualitative,colorway))
    return fig

def Box_plot(data,select_categorical_variable,select_numerical_variable,color_variable,colorway):
    fig = px.box(data, x=select_categorical_variable,y=select_numerical_variable,color=color_variable, points="all",color_discrete_sequence=getattr(px.colors.qualitative,colorway))
    return fig

def Line_chart(data, select_numerical_variable, select_num2_variable,color_variable,colorway):
    fig = px.line(data, x=select_numerical_variable, y=select_num2_variable, color=color_variable,color_discrete_sequence=getattr(px.colors.qualitative,colorway))
    return fig

def Donut(data,select_numerical_variable,select_categorical_variable,colorway):
    fig = px.pie(data, values=select_numerical_variable, names=select_categorical_variable, hole=.3,color_discrete_sequence=getattr(px.colors.qualitative,colorway))
    return fig

def Hist(data, select_variable,color_variable,colorway):
    fig = px.histogram(data, x=select_variable,color= color_variable, nbins=20,color_discrete_sequence=getattr(px.colors.qualitative,colorway))
    return fig

def Bar(data, select_categorical_variable, select_numerical_variable, color_variable,colorway):
    fig = px.bar(data, x=select_categorical_variable, y=select_numerical_variable, color=color_variable,color_discrete_sequence=getattr(px.colors.qualitative,colorway))
    return fig