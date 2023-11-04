#%%
import pandas as pd
import numpy as np
import geopandas as gpd

# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ngboost.scores import CRPScore, LogScore
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBRegressor, NGBClassifier
from ngboost.distns import T, Normal, TFixedDf, Poisson, k_categorical

import streamlit as st


import joblib

from branca.colormap import linear

import folium
from streamlit_folium import st_folium

from plotly import graph_objects as go

from pathlib import Path

# formato de la base
# df_test[['casos_vecinos_t1', 'casos_t1', 'casos_t2','indicador_techos',
    #    'indicador_abast_agua', 'indicador_agua', 'indicador_desague',
    #    'indicador_movilidad','Hombre', 'Mujer', 'Poblacion',
    #    'indicador_sex','densidad_pob','Afectado_EN2017',
    #     'mean_PRECIP_distrito_t1','var_PRECIP_distrito_t1']]
st.title("Modelo de Machine Learning probabilístico para el pronóstico de brotes de Dengue en los distritos de Piura")

st.text("El presente proyecto tiene por objetivo pronosticar los brotes de Dengue a nivel distrital en Piura, considerando un grupo de variables independiente y utilizando técnicas de Machine Learning probabilístico.")

st.text("Definiciones:")

st.text("Brote: cuando se presenta 1 o más casos en un distrito a la semana y al menos en las 4 semanas anteriores no se reportó ningún caso en este distrito.")

st.text("Bajo esta definición, entre 2022 y lo que va de 2023, en los 65 distritos de Piura se han sucitado 115 episodios (semanas) de brote, y 2,833 semanas en las que no se diagnosticaron casos. Así, la proporción de semanas con episodios de brote es de apenas el 3.9% del total de semanas analizadas.")

st.text("Se ha utilizado como muestras de entrenamiento a la información de 2022 y hasta la semana 11 de 2023 y como test desde la semana 11 hasta la semana 27.")

st.text("Los resultados del pronóstico en la muestra de test indica un bajo desempeño, 44.8% de recall y 19.4% de precisión y un indicador F1-score de 0.27. Sin embargo, debe considerarse que: (1) la detección de episodios de brote es muy dificil, considerando que solo 3.9% de las semanas tuvo un episodio de brote y (2)\
         la información hidrometereológica ha sido de gran importancia; sin embargo, no se cuenta con la información completa de todas las estaciones en todos los distritos, para este proyecto se contó con la información de las estaciones de 7 distritos. El resto fue imputado con el promedio provincial o regional.")

st.text("Como variables independientes se han utilizado a variables relacionadas a las condiciones de vida obtenidas del Censo Nacional 2017, variables hidrológicas (precipitaciones) provenientes de los datos abiertos del Gobierno Regional de Piura y que solo comprenden a 2022 y 2023.")

st.text("Asimismo, se añadieron variables proxy relacionadas al modelo Susceptible-Infected-Recovered (SIR), como el número de personas en cada distrito (Susceptible) el número de infectados en semanas anteriores en el distrito y los distritos vecinos (Infected). Se ha tomado en cuenta la información de los distritos vecinos\
         debido a que es plausible suponer la movilidad de individuos entre distritos. Un distrito vecino es aquel distrito que comparte frontera con el distrito analizado. Se ha considerado la cantidad de infectados en los distritos vecinos una, dos y tres semanas antes.")

st.text("Con respecto a las variables hidrológicas (precipitaciones), se ha considerado la información promedio de las estaciones en cada distrito en las que se encuentran. Asimismo, se ha utilizado la información referida a una y dos semanas anteriores, debido a que las\
         precipitaciones y acumulación de agua permiten el crecimiento de la población de mosquitos y el periodo de incubación se encuentra en promedio en 7 días.")

st.text("Con respecto al aspecto técnico, se ha utilizado específicamente el algoritmo Natural Gradient Boosting, cuyo resultado no son valores únicos (como casos de dengue), sino parámetros de distribuciones, lo que permite, además, obtener intervalos de probabilidad de ocurrencia del pronóstico.")

st.text("La Tabla 01 contiene a la información sobre el número de casos de los distritos vecinos y las variables hidrometereológicas. Estas son las variables que cambian cada semana. Las variables de características sociales obtenidas del censo, como porcentaje de viviendas\
         con techo de concreto, porcentaje de viviendas con acceso a agua, etc. no cambian en el tiempo y no se encuentran en la tabla, pero sí fueron parte del modelo.")

st.text("Asimismo, se ha añadido una variable binaria que indica si el distrito fue afectado por el fenómeno de El Niño en 2007.")

st.text("PUEDE MODIFICAR EL VALOR DE LAS VARIABLES EN LA TABLA Y VER COMO CAMBIAN LAS PROBABILIDADES DE BROTE DE DENGUE EN CADA DISTRITO.")


#%%
pipeline_ngboost = joblib.load(Path(__file__).parent / 'Modelo_brote_NGBoosting_bernoulli.pkl')
df_indicadores = pd.read_pickle(Path(__file__).parent / 'df_indicadores_distritales.pickle')
#%%
# seleccione distrito de PIURA

# st.selectbox(options=[''])

# tabla para leer datos

dict_ubigeos_distritos = {200101: 'Piura', 200104: 'Castilla', 200105: 'Catacaos', 200107: 'Cura Mori', 200108: 'El Tallan', 200109: 'La Arena', 200110: 'La Unión', 200111: 'Las Lomas', 200114: 'Tambo Grande', 200115: 'Veintiseis de Octubre',
                          200201: 'Ayabaca', 200202: 'Frias', 200203: 'Jilili', 200204: 'Lagunas', 200205: 'Montero', 200206: 'Pacaipampa', 200207: 'Paimas', 200208: 'Sapillica', 200209: 'Sicchez', 200210: 'Suyo', 200301: 'Huancabamba', 200302: 'Canchaque',
                          200303: 'El Carmen de la Frontera', 200304: 'Huarmaca', 200305: 'Lalaquiz', 200306: 'San Miguel de El Faique', 200307: 'Sondor', 200308: 'Sondorillo', 200401: 'Chulucanas', 200402: 'Buenos Aires', 200403: 'Chalaco',
                          200404: 'La Matanza', 200405: 'Morropón', 200406: 'Salitral', 200407: 'San Juan de Bigote', 200408: 'Santa Catalina de Mossa', 200409: 'Santo Domingo', 200410: 'Yamango', 200501: 'Paita', 200502: 'Amotape', 200503: 'Arenal',
                          200504: 'Colan', 200505: 'La Huaca', 200506: 'Tamarindo', 200507: 'Vichayal', 200601: 'Sullana', 200602: 'Bellavista', 200603: 'Ignacio Escudero', 200604: 'Lancones', 200605: 'Marcavelica', 200606: 'Miguel Checa',
                          200607: 'Querecotillo', 200608: 'Salitral', 200701: 'Pariñas', 200702: 'El Alto', 200703: 'La Brea', 200704: 'Lobitos', 200705: 'Los Organos', 200706: 'Mancora', 200801: 'Sechura', 200802: 'Bellavista de la Unión',
                          200803: 'Bernal', 200804: 'Cristo Nos Valga', 200805: 'Vice', 200806: 'Rinconada Llicuar'}

dict_distritos_ubigeos = {'Piura': 200101, 'Castilla': 200104, 'Catacaos': 200105, 'Cura Mori': 200107, 'El Tallan': 200108, 'La Arena': 200109, 'La Unión': 200110, 'Las Lomas': 200111, 'Tambo Grande': 200114, 'Veintiseis de Octubre': 200115,
                          'Ayabaca': 200201, 'Frias': 200202, 'Jilili': 200203, 'Lagunas': 200204, 'Montero': 200205, 'Pacaipampa': 200206, 'Paimas': 200207, 'Sapillica': 200208, 'Sicchez': 200209, 'Suyo': 200210, 'Huancabamba': 200301,
                          'Canchaque': 200302, 'El Carmen de la Frontera': 200303, 'Huarmaca': 200304, 'Lalaquiz': 200305, 'San Miguel de El Faique': 200306, 'Sondor': 200307, 'Sondorillo': 200308, 'Chulucanas': 200401, 'Buenos Aires': 200402,
                          'Chalaco': 200403, 'La Matanza': 200404, 'Morropón': 200405, 'Salitral': 200406, 'San Juan de Bigote': 200407, 'Santa Catalina de Mossa': 200408, 'Santo Domingo': 200409, 'Yamango': 200410, 'Paita': 200501,
                          'Amotape': 200502, 'Arenal': 200503, 'Colan': 200504, 'La Huaca': 200505, 'Tamarindo': 200506, 'Vichayal': 200507, 'Sullana': 200601, 'Bellavista': 200602, 'Ignacio Escudero': 200603, 'Lancones': 200604,
                          'Marcavelica': 200605, 'Miguel Checa': 200606, 'Querecotillo': 200607, 'Salitral': 200608, 'Pariñas': 200701, 'El Alto': 200702, 'La Brea': 200703, 'Lobitos': 200704, 'Los Organos': 200705, 'Mancora': 200706,
                          'Sechura': 200801, 'Bellavista de la Unión': 200802, 'Bernal': 200803, 'Cristo Nos Valga': 200804, 'Vice': 200805, 'Rinconada Llicuar': 200806}

df_input = pd.read_pickle(Path(__file__).parent / "variables_dinamicas.pickle").drop_duplicates()
df_input['ubigeo'] = df_input['ubigeo'].map(dict_ubigeos_distritos)
df_input = df_input.rename({'ubigeo':'Distrito', 'casos_vecinos_t1':'Casos de los distritos vecinos hace 1 semana',
                               'casos_vecinos_t2':'Casos de los distritos vecinos hace 2 semanas',
                               'casos_vecinos_t3':'Casos de los distritos vecinos hace 3 semanas',
                               'mean_PRECIP_distrito_t1':'Promedio de precipitación en el distrito hace 1 semana (mm)',
                               'mean_PRECIP_distrito_t2':'Promedio de precipitación en el distrito hace 2 semanas (mm)'}, axis=1)
df_input = df_input.set_index('Distrito')
#%%

st.subheader("Tabla 01. Variables dinámicas de casos de dengue en los distritos vecinos y variables hidrometereológicas")
df_input = st.data_editor(df_input)

# if st.button("Actualizar pronóstico"):


def style_function(x):
    nbh_count_colormap = linear.YlGnBu_09.scale(0, 1)
    return {
        'fillColor': nbh_count_colormap(x['properties']['Probabilidad de brote de dengue']),
        'color': 'black',
        'weight': 1.5,
        'fillOpacity': 0.7
        }

# @st.cache_resource()
def get_map(df_input):
    file = "Piura_mapa.shp"

    gdf = gpd.read_file(Path(__file__).parent / file)

    gdf['IDDIST'] = gdf['IDDIST'].astype(int)

    gdf = gdf.merge(df_input[['ubigeo','Distrito','Probabilidad de brote de dengue']], left_on='IDDIST', right_on='ubigeo')

    

    nbh_locs_map = folium.Map(location=[-5.182657, -80.662686],
                            zoom_start = 8, tiles='cartodbpositron')
    
    
    # style_function = _get_color()

    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['Distrito','Probabilidad de brote de dengue'],
            aliases=['Distrito','Probabilidad de brote de dengue'],
            localize=True
        )
    ).add_to(nbh_locs_map)
    return nbh_locs_map

# @st.cache_data()
def _get_calculus(df_input=df_input):
    df_input = df_input.rename({'Casos de los distritos vecinos hace 1 semana':'casos_vecinos_t1',
                            'Casos de los distritos vecinos hace 2 semanas':'casos_vecinos_t2',
                            'Casos de los distritos vecinos hace 3 semanas':'casos_vecinos_t3',
                            'Promedio de precipitación en el distrito hace 1 semana (mm)':'mean_PRECIP_distrito_t1',
                            'Promedio de precipitación en el distrito hace 2 semanas (mm)':'mean_PRECIP_distrito_t2'}, axis=1)
    
    df_input = df_input.reset_index()
    df_input['ubigeo'] = df_input['Distrito'].map(dict_distritos_ubigeos)

    df_input = df_input.merge(df_indicadores, left_on='ubigeo', right_on='ubigeo', how='left')

    # df_input = df_input.set_index('ubigeo')

    y_prob = pipeline_ngboost.predict_proba(df_input[['casos_vecinos_t1', 'casos_vecinos_t2','casos_vecinos_t3',
    'indicador_techos', 'indicador_abast_agua', 'indicador_agua',
    'indicador_desague', 'indicador_movilidad', 'Hombre', 'Mujer',
    'Poblacion', 'indicador_sex', 'densidad_pob', 'Afectado_EN2017',
    'mean_PRECIP_distrito_t1', 'mean_PRECIP_distrito_t2']])
    return df_input, y_prob

# if st.button('Pronosticar'):

#%%
A, B = _get_calculus()
df_input = A
df_input['Probabilidad de brote de dengue'] = B[:,1] #y_prob[:,1]

    # df_input = df_input.reset_index()

    # df_input[['Distrito','Probabilidad de brote de dengue']]

    # mapa de colores de los distritos según probabilidad de brote
    
nbh_locs_map = get_map(df_input)
st.subheader("Figura 01. Mapa de la probabilidad de brote")
st_folium(nbh_locs_map)

# gráfico de barras de probabilidad de brote
#%%
ordered = df_input.set_index('Distrito')['Probabilidad de brote de dengue'].sort_values(ascending=False)
bar = go.Figure([go.Bar(y=ordered.to_numpy(), x=ordered.index)])
#%%
st.subheader("Figura 02. Probabilidad de brote de Dengue según distrito")
st.plotly_chart(bar)


# gráficos de densidad de probabilidad de brote

# list_distitos = ['Piura', 'Castilla', 'Catacaos', 'Cura Mori', 'El Tallan', 'La Arena', 'La Unión', 'Las Lomas', 'Tambo Grande', 'Veintiseis de Octubre', 'Ayabaca', 'Frias', 'Jilili', 
#              'Lagunas', 'Montero', 'Pacaipampa', 'Paimas', 'Sapillica', 'Sicchez', 'Suyo', 'Huancabamba', 'Canchaque', 'El Carmen de la Frontera', 'Huarmaca', 'Lalaquiz',
#               'San Miguel de El Faique', 'Sondor', 'Sondorillo', 'Chulucanas', 'Buenos Aires', 'Chalaco', 'La Matanza', 'Morropón', 'Salitral', 'San Juan de Bigote',
#               'Santa Catalina de Mossa', 'Santo Domingo', 'Yamango', 'Paita', 'Amotape', 'Arenal', 'Colan', 'La Huaca', 'Tamarindo', 'Vichayal', 'Sullana', 'Bellavista',
#               'Ignacio Escudero', 'Lancones', 'Marcavelica', 'Miguel Checa', 'Querecotillo', 'Salitral', 'Pariñas', 'El Alto', 'La Brea', 'Lobitos', 'Los Organos', 'Mancora',
#               'Sechura', 'Bellavista de la Unión', 'Bernal', 'Cristo Nos Valga', 'Vice', 'Rinconada Llicuar']
# %%
