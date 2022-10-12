import streamlit as st
import pandas as pd
import numpy as np
pd.read_csv("https://raw.githubusercontent.com/labeconometria/MLxE/main/proyectos1er/dataset_2.")

st.title("Proyecto 8")
st.write("En la presente página se realizará el análisis de un dataset que busca explicar qué tipo de cliente se tiene, si uno existente o uno retirado, a partir de datos como la edad, nivel de educación, ingresos,entre otras. Para la cual se realiza lo siguiente:")
st.dataframe(df)
st.caption("Dataframe interactiva")
st.subheade("Descripción suncita de las variables")

