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
Customer_Age =st.number_input('Edad')
Gender = st.selectbox('Genero:',('Masculino','Femenino'))

Gender = 1 if Gender == 'Masculino' else 0

Education_Level = st.number_input('Nivel educativo:(Si es ineducado ingrese 0, si es secundaria ingrese 1, si es universitario ingrese 2, si es graduado ingrese 3, si es postgrado ingrese 4 y si es doctorado ingrese 5, si es desconocido ingrese 6)')

Marital_Status = st.selectbox('Estado Civil:',('Casado','Soltero'))

Marital_Status = 1 if Marital_Status == 'Casado' else 0

Income_Category= st.number_input('Categoría de ingresos:(Si es desconocido ingrese 0, si es menor a $40K ingrese 1, si está entre $40K - $60K ingrese 2,si está entre $60K - $80K ingrese 3, si está entre $80K - $120K ingrese 4, si es $120K o mayor ingrese 5)')
Card_Category = st.number_input('Tipo de tarjeta(Si es Azul ingrese 0, nsi es dorada ingrese 1, si es plata ingrese 2 y si es platino ingrese 3)')
Months_on_book = st.number_input('Duración de la relación con el banco')
Total_Relationship_Count = st.number_input('Número total de productos')
Months_Inactive_12_mon= st.number_input('Número de meses de inactividad')
Credit_Limit = st.number_input('Límite de crédito')
Total_Revolving_Bal = st.number_input('Saldo rotativo total')
Avg_Open_To_Buy = st.number_input('Línea de crédito abierta a la compra (media de los últimos 12 meses)')
Total_Amt_Chng_Q4_Q1 = st.number_input('Variación del importe de las transacciones(cuarto trimestre sobre primer trimestre)')
Total_Trans_Amt = st.number_input('Cantidad total de las transacciones(12 meses)')
Total_Trans_Ct = st.number_input('Recuento de transacciones')
Total_Ct_Chng_Q4_Q1 = st.number_input('Cambio en el recuento de transacciones')
Avg_Utilization_Ratio =  st.number_input('Utilización promedio de la tarjeta')

st.subheader("""Modelo """)

clsr_pickle = open('clsr_randomforest.pickle','rb')
clsr = pkl.load(clsr_pickle)
clsr_pickle.close()

prediction = clsr.predict([['Gender','Marital_Status','Customer_Age','Education_Level','Months_on_book','Income_Category','Total_Relationship_Count','Months_Inactive_12_mon',
'Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']]) 
print(prediction) 
    

resultado = 'Existing customer 'if prediction ==1 else 'Attrited Customer'
print(resultado)



#clsr = DecisionTreeClassifier(max_depth=10)
#clsr.fit(X_train, y_train)

#y_pred = clsr.predict(X_test)
#y_fit_train = clsr.predict(Attrition Flag)


#clsr_pickle = open('clsr_randomforest.pickle','rb')

#clsr = pkl.load(clsr_pickle)

#print(clsr)

