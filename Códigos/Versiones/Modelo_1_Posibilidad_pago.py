# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:29:44 2026

@author: Andrés Felipe Arévalo Arévalo
"""
#_________________________________________________________________________________________________________________________
# Modulos
#_________________________________________________________________________________________________________________________
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

#_________________________________________________________________________________________________________________________
# Importando datos
#_________________________________________________________________________________________________________________________
# IMPORTANTE: Cambiese el objeto 'ruta', coloquese la ruta donde se almaceno la carpeta 'Arévalo_Andrés - Prueba_Técnica - Casa_Cobranzas_Beta' <----------------------
ruta = Path(r'C:\Users\USUARIO\Desktop\Main\Pruebas de conocimiento\Davivienda - Cobranzas Betas\Arévalo_Andrés - Prueba_Técnica - Casa_Cobranzas_Beta')
df_evolucion_enriquecida = pd.read_csv(ruta /'Resultados/df_evolucion_enriquecida.txt',sep='|', encoding='utf-8')

#_________________________________________________________________________________________________________________________
# Consideraciones preliminares
#_________________________________________________________________________________________________________________________
'''Pregunta objetivo: ¿Es probable que el cliente realice al menos un pago?
    - Variable objetivo binaria.
    - Observación a nivel cliente.
    - Problema de clasificación binaria. 
  Con base en lo anterior, primero hay que definir que se consider como 'realizar un pago', para esto hare uso de la variable 'TOTAL_PAGOS_APROBADOS' mayor
  a cero para construir una variable binaria, donde 1 es el cliente realizó al menos un pago y 0 el cliente no realizó ningun pago.'''

df_evolucion_enriquecida['PAGO_REALIZA'] = (df_evolucion_enriquecida['TOTAL_PAGOS_APROBADOS'] > 0).astype(int)

'''Ahora, la base de datos tiene filas por obligación, no por cliente. Luego, dado que el análisis busca determinar si un individuo realizo por lo menos un pago,
   se unificara la base de datos por cliente; de esta forma, el foco de anlisis del modelo sera el cliente y no la obligación.
   
   Al analizar las variables disponibles que no tienen relación logica con la intención de pagar, tales como: numeros de telefono e identificadores. De igual forma,
   se excluyen las variables a nivel de obligación que resultan redundantes, tales como los saldos de capital y los días de mora. Por otra parte, se excluyo el 'producto'
   del modelo a nivel individuo. Lo anterior, debido a que el roducto esta a nivel obligación, lo que permite que un solo individuo tenga diferentes productos, luego,
   no es posible asignar un pruducto a un individuo a no ser que se realice una clasificación, puede ser por cercanias, de productos para generar una unica macrocatoria.
   Pero, en vista que existe la variable 'TIPO_CLIENTE', se opto por emplearla como una proxi de la complejidad financiera/productos de los individuos.'''

vars_modelo = [
    'SALDO_TOTAL_CLIENTE',
    'RANGO_MORA_CLIENTE',
    'TIPO_CLIENTE',
    'ESTADO_ORIGEN']

df_cliente = (
    df_evolucion_enriquecida
    .groupby('IDENTIFICACION')
    .agg(
        PAGO_REALIZA=('PAGO_REALIZA', 'max'),
        SALDO_TOTAL_CLIENTE=('SALDO_TOTAL_CLIENTE', 'first'),
        RANGO_MORA_CLIENTE=('RANGO_MORA_CLIENTE', 'first'),
        TIPO_CLIENTE=('TIPO_CLIENTE', 'first'),
        ESTADO_ORIGEN=('ESTADO_ORIGEN', 'first')).reset_index())

y = df_cliente['PAGO_REALIZA']
X = df_cliente[vars_modelo]
#_________________________________________________________________________________________________________________________
# Analisis descriptivo
#_________________________________________________________________________________________________________________________
df_cliente['PAGO_REALIZA'].value_counts(normalize=True)
df_cliente['RANGO_MORA_CLIENTE'].value_counts(normalize=True)

''' Se evidencia que más del 83 % de los datos coresponden a personas que no realizarón ningun pago y más del 87 % de los individuos 
    tiene más de 540 días de mora. Luego, el problema esta desbalanceado, por lo cual el accuracy y el umbtral del 0.5 no son una buenas medidas
    de evaluación.'''
    
prob_condicional_pago_dado_mora = pd.crosstab(df_cliente['RANGO_MORA_CLIENTE'], df_cliente['PAGO_REALIZA'], normalize='index')

''' El código de la linea 70 es la probabilidad condicional del pago realizado dado el rango de la mora: los valores de la categoria 'MAS DE 540'  implican que,
    un cliente con mora mayor a 540 días solo paga en el 16.6 % de los casos. La relación pago-mora presenta un comportamiento particular, pues los clientes con
    mora extrema preentan una probabilidad e pago superor que los clientes con moras intermedias, esto podría deberse a que los clientes con mora extrema pueden tener
    procesos especiales o acuerdos de pago, mienras que los clientes con mora media postergan los pagos. Luego, la mora no es lineal en su efecto sobre el pago.
    Por lo anterior, un modelo lineal simple seria ineficas para explicar esta relación'''

prob_condicional_pago_dado_tipo_cliente = pd.crosstab(df_cliente['TIPO_CLIENTE'], df_cliente['PAGO_REALIZA'], normalize='index')

''' Al igual que en el caso anterior, resalta que los clientes multiproducto pagan más del doble que los clientes monoproducto. Luego, entre más producto tienen 
    los clientes presentan una mayor relación con la entidad por lo que existe un mayor incentivo a mantenerse al día '''

prob_condicional_pago_dado_estado = pd.crosstab(df_cliente['ESTADO_ORIGEN'], df_cliente['PAGO_REALIZA'], normalize='index')

''' No existe una diferencia relativa. Tener un acuerdo registrado no cambia, en promedio, la probabilidad de realizar al menos un pago. Se observa que esta variable
    no separa bien a los pagadores de los no pagadores. No pareciera aportar información, inicialmente se va a sacar del modelo, posteriormente se evaluara su 
    incorporación en función del ajuste optenido.'''
    
#_________________________________________________________________________________________________________________________
# Modelo Probit
#_________________________________________________________________________________________________________________________

vars_modelo = ['SALDO_TOTAL_CLIENTE', 'RANGO_MORA_CLIENTE', 'TIPO_CLIENTE']
y = df_cliente['PAGO_REALIZA']
X = df_cliente[vars_modelo]
X = pd.get_dummies(X, drop_first=True, dtype=float)
X = sm.add_constant(X)

freq = y.value_counts(normalize=True)
weights = y.map({0: 1 / freq[0], 1: 1 / freq[1]})

probit_model = sm.Probit(y, X)
probit_results = probit_model.fit(weights=weights)
print(probit_results.summary())

df_cliente['PROB_PAGO'] = probit_results.predict(X)


# Evaluación
roc_auc_score(y, df_cliente['PROB_PAGO'])

''' Dado que tenemos un conjunto de datos desvalanceado el accurancy no es una metrica optima a emplear, luego la métrica ROC-AUC se convierte en nuestras metrica
    principal y, dado que su valor es 0.61, se considera un valor aceptable'''
    
fpr, tpr, thresholds = roc_curve(y, df_cliente['PROB_PAGO'])
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC – Modelo Probit')
plt.show()

# Umbral de clasificación
''' Como solo el 16 % de los individuos pagan, es imposibole usar el críterio del umbral del 0.5. Luego, se emplea el umbral que maimiza el puntaje F1'''
scores = []
for t in thresholds:
    pred = (df_cliente['PROB_PAGO'] >= t).astype(int)
    scores.append((t, f1_score(y, pred)))

best_threshold = max(scores, key=lambda x: x[1])[0]

# Clasificación final
df_cliente['PRED_PAGO'] = (df_cliente['PROB_PAGO'] >= best_threshold).astype(int)