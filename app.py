import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

df = pd.read_csv("https://raw.githubusercontent.com/labeconometria/MLxE/main/proyectos1er/dataset_2.csv")

st.title("Proyecto 8")
st.write("En la presente página se realizará el análisis de un dataset que busca explicar qué tipo de cliente se tiene, si uno existente o uno retirado, a partir de datos como la edad, nivel de educación, ingresos,entre otras. Para la cual se realiza lo siguiente:")
st.dataframe(df)
st.caption("Dataframe interactiva")
st.subheader("Descripción suncita de las variables")
st.write(""" * Attrition Flag "Desgaste(Cliente)" , es nuestra variable objetivo, de tipo category donde se tienen Existing Customer "Cliente Existente" y Attrited Customer "Cliente Atraido". """)
st.write(""" * Customer Age "Edad del Cliente", variable continua donde sus datos van desde 26 hasta 70 años, presenta una media de 46 años.""")
st.write(""" * Gender "Genero", variable categoria que nos muestra si el cliente es hombre o mujer. """)
st.write(""" * Dependent Count "Dependencia", variable continua que demuestra el nivel de ***, sus valores van desde 0 a 5, presenta una media de 2 aproximadamente. """)
st.write(""" * Education Level "Nivel de Educacion", variable categorica que permite conocer el nivel de educacion del cliente, posibles respuestas son escuela secundaria, pregrado, sin educación, desconocido, universidad, posgrado, doctorado. """)
st.write(""" * Marital Status "Estado Civil", variable de tipo category en donde se permite conocer en cuál de los siguientes estados se encuentra el cliente; Casado, soltero, divorciado , desconocido. """)
st.write(""" * Income Category "Categoría de Ingresos" , variable de tipo categórico donde se da información sobre el nivel de ingreso del cliente, esta variable presenta los datos a través de 6 rangos de ingreso desde menos de 40K hasta más de 120 K y con desconocido. """)
st.write(""" * Card Category "Categoría de la tarjeta", variable que nos da conocimiento sobre el tipo de tarjeta que posee el cliente puede ser Azul, Dorada, Plateada y Platino. """)
st.write(""" * Months on book-Discreta "Meses en el libro", variable discreta donde se da informacion sobre el periodo de tiempo dentro del libro, esta variable tiene datos desde 13 a 56 meses y con una media de 36 meses. """)
st.write(""" * Credit Limit "Credito Limite", variable de tipo continua donde se da informacion de los limites de credito para cada cliente, la media del credito es de 8606. """)
st.write(""" * Las demas Variables que tenemos informacion son totales estadisticos de las anteriores mencionadas, poseen datos mas grandes y su informacion estadistica se encuentra en el describe. """)
st.table(df.describe())
st.write("A partir de esto realizamos los graficos de datos relevantes " )
st.write("Cada gráfico está separado a partir de la variable objetivo Attrition Flag")
st.image("./2/des3.png")
st.image("./2/des4.png")
st.image("./2/des5.png")
st.image("./2/des6.png")
st.image("./2/des7.png")

st.image("./2/heat1.png")
st.image("./2/shap1.png")
st.image("./2/shap2.png")
st.image("./2/shap3.png")

st.write("[link](https://colab.research.google.com/drive/1unQpfTfqtcD1Ns9mFfYkIAwlN9drONfM#scrollTo=8KNX78rFum6r) entrenamiento modelo")
st.subheader("Prediccion")





nombre = st.text_input('Nombre completo')
Customer_Age =st.text_input('Edad')
Gender = st.text_input('Genero')

Gender = 1 if Gender == 'Maculino' else 0

dependent = st.text_input('Número de personas a cargo')
Education_Level = st.text_input('Nivel educativo')

if education_level == "uneducated": 
        education_level = 0 

if education_level == "High School": 
        education_level = 1 

if education_level == "College": 
        education_level = 2 

if education_level == "Graduate": 
        education_level = 3
       
if education_level == "Post-Graduate": 
        education_level = 4 

if education_level == "Doctorate": 
        education_level = 5
       
if education_level == "Doctorate": 
        education_level = 6
#education_level ==0 if education_level='Uneducated'
#education_level == 1 if education_level=='High School'
#education_level == 2 if education_level=='College'
#education_level == 3 if education_level=='Graduate'
#education_level == 4 if education_level=='Post-Graduate'
#education_level == 5 if education_level=='Doctorate'
#education_level == 6 if education_level=='Unknown'

,'High School':1, 'College':2,'Graduate':3,'Post-Graduate':4, 'Doctorate':5,'Unknown':6}

Marital_Status = st.text_input('Estado civil')

Marital_Status = 1 if Marital_Status == 'Casado' else 0

Income_Category= st.text_input('Categoría de ingresos ')
card = st.text_input('Tipo de tarjeta')
Months_on_book = st.text_input('Duración de la relación con el banco')
Total_Relationship_Count = st.text_input('Número total de productos')
Months_Inactive_12_mon= st.text_input('Número de meses de inactividad')
cc = st.text_input('Número de contactos')
Credit_Limit = st.text_input('Límite de crédito')
Total_Revolving_Bal = st.text_input('Saldo rotativo total')
Avg_Open_To_Buy = st.text_input('Línea de crédito abierta a la compra (media de los últimos 12 meses)')
Total_Amt_Chng_Q4_Q1 = st.text_input('Variación del importe de las transacciones(cuarto trimestre sobre primer trimestre)')
Total_Trans_Amt = st.text_input('Cantidad total de las transacciones(12 meses)')
Total_Trans_Ct = st.text_input('Recuento de transacciones')
Total_Ct_Chng_Q4_Q1 = st.text_input('Cambio en el recuento de transacciones')
Avg_Utilization_Ratio =  st.text_input('Utilización promedio de la tarjeta')


st.subheader("""Modelo """)

clsr_pickle = open('clsr_randomforest.pickle','rb')
clsr = pkl.load(clsr_pickle)
clsr_pickle.close()

prediction = clsr.predict( 
        [['Gender','Marital_Status','Customer_Age','Education_Level','Months_on_book','Income_Category','Total_Relationship_Count','Months_Inactive_12_mon',
'Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']]) 
    print(prediction) 
    

resultado = 'Existing customer 'if prediction ==1 else 'Attrited Customer'
      return resultado



clsr = DecisionTreeClassifier(max_depth=10)
clsr.fit(X_train, y_train)

y_pred = clsr.predict(X_test)
y_fit_train = clsr.predict(Attrition Flag)
