# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 09:56:46 2026

@author: Andrés Felipe Arévalo Arévalo - Prueba Técnica - Casa de Cobranzas Beta
"""
#_________________________________________________________________________________________________________________________
# Modulos
#_________________________________________________________________________________________________________________________
import pandas as pd
from pathlib import Path
import unicodedata
import numpy as np

#_________________________________________________________________________________________________________________________
# Importando datos
#_________________________________________________________________________________________________________________________

# IMPORTANTE: Cambiese el objeto 'ruta', coloquese la ruta donde se almaceno la carpeta 'Arévalo_Andrés - Prueba_Técnica - Casa_Cobranzas_Beta' <----------------------
ruta = Path(r'C:\Users\USUARIO\Desktop\Main\Pruebas de conocimiento\Davivienda - Cobranzas Betas\Arévalo_Andrés - Prueba_Técnica - Casa_Cobranzas_Beta')

# El archivo 'EVOLUCION' presenta una mezcla de codificaciones. 
# Se utiliza 'latin1' porque permite decodificar todos los bytes sin generar errores
EVOLUCION = pd.read_csv(ruta / 'Datos_Originales/EVOLUCION.txt', sep='\t', encoding='latin1', engine='python', dtype= {'ESTADO CLIENTE': 'boolean'})
EVOLUCION.columns = EVOLUCION.columns.str.strip()
PAGOS = pd.read_csv(ruta / 'Datos_Originales/PAGOS.txt', sep='|')
PAGOS.columns = PAGOS.columns.str.strip()
TELEFONOS = pd.read_csv(ruta / 'Datos_Originales/TELEFONOS.txt', sep='\t')
TELEFONOS.columns = TELEFONOS.columns.str.strip()

#_________________________________________________________________________________________________________________________
# Validando consistencia de la información
#_________________________________________________________________________________________________________________________
## Analizando los tipos de datos
EVOLUCION.dtypes # -> Errores iniciales: 'ESTADO CLIENTE' no Booleano, 'SALDO CAPITAL MES' no Float, 'PAGO_MINIMO' no Float
PAGOS.dtypes # -> Errores iniciales: 'PAGOS' no Float
TELEFONOS.dtypes # -> Errores iniciales: 'TELEFONO 1' no String

## Evaluando consistencia de los tipos de datos
''' se emplea tipo de dato string para detectar categorías erróneas como texto (NA, None, null, etc).
    La variable 'ESTADO CLIENTE' solo presenta dos categorías, se puede convertir en booleano directamente. 
    Se corrige el formato directamente en la importación de la base'''
EVOLUCION['ESTADO CLIENTE'].astype(str).value_counts() 
                                                       
'''La variable 'SALDO CAPITAL MES' y 'PAGO_MINIMO' fue leída como texto debido a inconsistencias de formato; símbolos monetarios, separadores y valores no numéricos;.
   Se realizó una validación programática para identificar valores no convertibles y posteriormente se normalizó el formato antes de su conversión a
   tipo numérico.'''
patron_valido = r'^\s*\$?\s*\d{1,3}(\.\d{3})*\s*$'
col = EVOLUCION['SALDO CAPITAL MES'].astype(str)
invalidos = EVOLUCION.loc[~col.str.match(patron_valido),'SALDO CAPITAL MES'] # -> Vacío, todos los datos siguen el patron esperado
EVOLUCION['SALDO CAPITAL MES'] = (EVOLUCION['SALDO CAPITAL MES'].astype(str)
    .str.strip().str.replace('$', '', regex=False).str.replace('.', '', regex=False)
    .replace('', pd.NA).astype(float))

col = EVOLUCION['PAGO_MINIMO'].astype(str)
invalidos = EVOLUCION.loc[~col.str.match(patron_valido),'PAGO_MINIMO'] # -> Vacío, todos los datos siguen el patron esperado
EVOLUCION['PAGO_MINIMO'] = (EVOLUCION['PAGO_MINIMO'].astype(str)
    .str.strip().str.replace('$', '', regex=False).str.replace('.', '', regex=False)
    .replace('', pd.NA).astype(float))

patron_valido_pagos = r'^\s*\d+,\d+\s*$'
col = PAGOS['PAGOS'].astype(str)
invalidos = PAGOS.loc[~col.str.match(patron_valido_pagos),'PAGOS'] # -> Vacío, todos los datos siguen el patron esperado
PAGOS['PAGOS'] = (PAGOS['PAGOS'].astype(str).str.strip().str.replace(',', '.', regex=False)
    .replace('', pd.NA).astype(float))
PAGOS['CUENTA'] = PAGOS['CUENTA'].astype(str)

TELEFONOS['TELEFONO 1'] = TELEFONOS['TELEFONO 1'].apply(lambda x: str(int(x)) if pd.notna(x) else pd.NA)

## Análisis y corrección de las categorías al interior de la variable 'productos'
producto_categorias = EVOLUCION['producto'].value_counts()

'''Inicialmente se evidenció una mezcla de mayúsculas y minúsculas, así como palabras con y sin acentos. Por lo anterior, se procedió a convertir todas las 
    categorías a mayúsculas y a eliminar los acentos, con el objetivo de estandarizar la información. No obstante, de ser necesario, los acentos podrían conservarse, 
    siempre que se especifique previamente la norma de depuración de datos a emplear. Para el presente ejercicio, se optó por trabajar con categorías en mayúsculas
    y sin acentos. Posteriormente, se identificaron las categorías con errores evidentes y se procedió a su corrección.'''

EVOLUCION['producto'] = EVOLUCION['producto'].str.upper()

def quitar_acentos(texto):
    if not isinstance(texto, str):
        return texto
    return (
        unicodedata
        .normalize('NFKD', texto)
        .encode('ascii', 'ignore')
        .decode('ascii'))
EVOLUCION['producto'] = (EVOLUCION['producto'].str.upper().apply(quitar_acentos))

mask_raros = EVOLUCION['producto'].astype(str).str.contains(r'[^A-Z ]',regex=True)
categorias_raras = EVOLUCION.loc[mask_raros, 'producto'].value_counts()

'''Las categorías con errores evidentes fueron corregidas en función de las categorías ya existentes. Dada la falta de información sobre la naturaleza de los
     productos, no se unificaron categorías como “TARJETA” y “TARJETA DE CRÉDITO”. En ausencia de información adicional, únicamente se corrigieron errores 
     manifiestos en la denominación. Por otra parte, las categorías '0' y 'TARJETA PEN<71 EXC-DESEM2' no se modificaron debido a la falta de información que 
    permita una corrección confiable.'''
nuevas_categorias = {'LIBRE_INVERSION':'LIBRE INVERSION', 'ACUERDO_PACTADO': 'ACUERDO PACTADO',
                     'VEHICULO_SIN_PRENDA': 'VEHICULO SIN PRENDA', 'LIBRAAANZA': 'LIBRANZA',
                     'COLEEEEEGIO':'COLEGIO', 'CRAAADITO HIPOTECAR':'CREDITO HIPOTECARIO',
                     'TARJETA DE CRAAADITO':'TARJETA DE CREDITO', 
                     'VEHACULO MOVIL SIN PRENDA':'VEHICULO MOVIL SIN PRENDA',
                     'HIPOTECARI':'HIPOTECARIO', 'VEHICULO':'VEHICULOS'}
EVOLUCION['producto'] = EVOLUCION['producto'].str.strip()
EVOLUCION['producto'] = EVOLUCION['producto'].replace(nuevas_categorias) # -> se paso de 65 a 59 categorias
    
#_________________________________________________________________________________________________________________________
# Uniendo bases de datos
#_________________________________________________________________________________________________________________________

# Pagos agregados   
EVOLUCION['OBLIGACION_'] = (EVOLUCION['OBLIGACION'].astype(str).str.replace(r'\D+', '', regex=True))
EVOLUCION['OBLIGACION_'] = EVOLUCION['OBLIGACION_'].str.strip()
PAGOS['CUENTA'] = PAGOS['CUENTA'].str.strip()

PAGOS_OK = PAGOS[PAGOS['ESTADO_PAGO'] == 'APROBADO']
PAGOS_AGG = (PAGOS_OK.groupby('CUENTA', as_index=False).agg(TOTAL_PAGOS_APROBADOS=('PAGOS', 'sum'),
        N_PAGOS_APROBADOS=('PAGOS', 'count')))

df = pd.merge(EVOLUCION, PAGOS_AGG, left_on='OBLIGACION_', right_on='CUENTA', how='left')

# Telefonos en formato ancho
TELEFONOS['TELEFONO'] = (TELEFONOS['TELEFONO 1'].astype(str).str.replace(r'\D+', '', regex=True))
mask_celular = TELEFONOS['TELEFONO'].str.match(r'^7[0-5]\d{8}$')
mask_fijo    = TELEFONOS['TELEFONO'].str.match(r'^[1-5]\d{6}$')
TELEFONOS['TIPO_TEL'] = None
TELEFONOS.loc[mask_celular, 'TIPO_TEL'] = 'TELEFONO_CELULAR'
TELEFONOS.loc[mask_fijo, 'TIPO_TEL']    = 'TELEFONO_FIJO'
TELEFONOS_VALIDOS = TELEFONOS.dropna(subset=['TIPO_TEL'])
TELEFONOS_wide = (TELEFONOS_VALIDOS.pivot_table(
        index='IDENTIFICACION',
        columns='TIPO_TEL',
        values='TELEFONO',
        aggfunc='first'
    ).reset_index())

df_final = pd.merge(df, TELEFONOS_wide, left_on='IDENTIFICACION', right_on='IDENTIFICACION', how='left')
EVOLUCION_final = df_final.drop(columns=['OBLIGACION_', 'CUENTA', 'N_PAGOS_APROBADOS'])

#_________________________________________________________________________________________________________________________
# Columnas adicionales
#_________________________________________________________________________________________________________________________

EVOLUCION_final['TIPO_CLIENTE'] = (EVOLUCION_final.groupby('IDENTIFICACION')['producto'].transform('nunique')
                                   .map(lambda x: 'MONOPRODUCTO' if x == 1 else 'MULTIPRODUCTO'))
EVOLUCION_final['ESTADO_ORIGEN'] = (EVOLUCION_final['producto'].str.contains('ACUERDO', regex=False, na=False)
                                    .map({True: 'CON_ACUERDO', False: 'SIN_ACUERDO'}))
EVOLUCION_final['SALDO_TOTAL_CLIENTE'] = (EVOLUCION_final.groupby('IDENTIFICACION')['SALDO CAPITAL MES'].transform('sum'))
bins = [-1, 0, 30, 60, 90, 120, 180, 360, 540, float('inf')]
labels = [
    'AL DIA',
    'MENOS 30',
    'MENOS 60',
    'MENOS 90',
    'MENOS 120',
    'MENOS 180',
    'MENOS 360',
    'MENOS 540',
    'MAS DE 540']
EVOLUCION_final['RANGO_DIAS_MORA'] = pd.cut(EVOLUCION_final['Dias Mora'], bins=bins, labels=labels, right=False)
EVOLUCION_final['RANGO_DIAS_MORA'] = pd.Categorical(EVOLUCION_final['RANGO_DIAS_MORA'], categories=labels, ordered=True)
EVOLUCION_final['RANGO_MORA_CLIENTE'] = (EVOLUCION_final.groupby('IDENTIFICACION')['RANGO_DIAS_MORA'].transform('max'))

condiciones = [EVOLUCION_final['TOTAL_PAGOS_APROBADOS'] > EVOLUCION_final['PAGO_MINIMO'], 
               EVOLUCION_final['TOTAL_PAGOS_APROBADOS'] >= 0.7 * EVOLUCION_final['PAGO_MINIMO']]
resultados = ['CUMPLE TOTAL', 'CUMPLE PARCIAL']
EVOLUCION_final['CUMPLE_PAGO_MINIMO'] = np.select(condiciones, resultados, default='NO CUMPLE')


#_________________________________________________________________________________________________________________________
# Exportar base final
#_________________________________________________________________________________________________________________________

EVOLUCION_final.to_csv(ruta /'Resultados/df_evolucion_enriquecida.txt',sep='|',index=False,encoding='utf-8')
EVOLUCION_final.to_pickle(ruta / 'Códigos/Manipulación_Bases_Datos.pkl')