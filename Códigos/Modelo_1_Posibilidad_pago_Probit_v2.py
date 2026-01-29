# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:29:44 2026

@author: Andrés Felipe Arévalo Arévalo
"""
#_________________________________________________________________________________________________________________________
# Modulos
#_________________________________________________________________________________________________________________________
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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

''' Inicialmente, la base de datos se encuentra estructurada a nivel de obligación y no a nivel de cliente. Dado que el objetivo del análisis es determinar si un
    individuo realiza al menos un pago, se unifico la información a nivel de la persona. En el proceso de selección de variables, se excluyeron aquellas que no 
    una relación lógica con la intención de pago, tales como identificadores y números de teléfono. Asimismo, se descartaron variables definidas a nivel de 
    obligación que resultan redundantes una vez consolidada la información por cliente, como los saldos de capital individuales y los días de mora por obligación. 
    Adicionalmente, la variable producto no fue incluida directamente en el modelo a nivel de individuo, dado que un mismo cliente puede estar asociado a múltiples
    productos. Para incorporar esta información sería necesario realizar una clasificación previa de los productos en macrocategorías. Sin embargo, considerando 
    que ya se dispone de la variable TIPO_CLIENTE, se optó por utilizarla como una aproximación de la complejidad financiera y la diversidad de productos del 
    individuo.'''

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
        DIAS_MORA=('Dias Mora', 'max'),  
        RANGO_MORA_CLIENTE=('RANGO_MORA_CLIENTE', 'first'),
        TIPO_CLIENTE=('TIPO_CLIENTE', 'first'),
        ESTADO_ORIGEN=('ESTADO_ORIGEN', 'first')).reset_index())

#_________________________________________________________________________________________________________________________
# Analisis descriptivo
#_________________________________________________________________________________________________________________________
df_cliente['PAGO_REALIZA'].value_counts(normalize=True)

''' Se evidencia que más del 83 % de los registros corresponden a personas que no realizaron ningún pago y que más del 87 % de los individuos presentan más de 540
    días de mora. En consecuencia, el problema se encuentra altamente desbalanceado, por lo que métricas como el accuracy y el uso de un umbral de clasificación 
    fijo de 0.5 no resultan apropiados para la evaluación del modelo.'''
    
prob_condicional_pago_dado_mora = pd.crosstab(df_cliente['RANGO_MORA_CLIENTE'], df_cliente['PAGO_REALIZA'], normalize='index')

''' El código de la línea 74 calcula la probabilidad condicional de que un cliente realice al menos un pago dado su rango de mora. En particular, los valores 
    a la categoría 'MÁS DE 540' indican que un cliente con una mora superior a 540 días realiza un pago únicamente en el 16.6 % de los casos. La relación entre
    pago y mora presenta un comportamiento particular, ya que los clientes con mora extrema exhiben una probabilidad de pago superior a la observada en clientes 
    con moras intermedias. Este patrón puede explicarse por la existencia de procesos especiales de cobranza o acuerdos de pago aplicados a clientes con mora
    elevada, mientras que aquellos con moras intermedias tienden a postergar los pagos. En consecuencia, la relación entre mora y probabilidad de pago no es lineal,
    por lo que un modelo lineal simple resultaría ineficaz para capturar adecuadamente esta dinámica.'''

prob_condicional_pago_dado_tipo_cliente = pd.crosstab(df_cliente['TIPO_CLIENTE'], df_cliente['PAGO_REALIZA'], normalize='index')

''' Al igual que en el caso anterior, se observa que los clientes multiproducto presentan una probabilidad de pago superior a la de los clientes monoproducto, 
    siendo esta más del doble. Este resultado sugiere que, a medida que los clientes poseen un mayor número de productos, su relación con la entidad se fortalece,
    lo que genera mayores incentivos para mantenerse al día con sus obligaciones financieras.'''

prob_condicional_pago_dado_estado = pd.crosstab(df_cliente['ESTADO_ORIGEN'], df_cliente['PAGO_REALIZA'], normalize='index')

''' No se observa una diferencia relevante entre los grupos analizados: tener un acuerdo registrado no modifica, en promedio, la probabilidad de que un cliente 
    realice al menos un pago. En consecuencia, esta variable no discrimina adecuadamente entre clientes pagadores y no pagadores y no parece aportar información
    explicativa significativa. Por lo tanto, se excluye inicialmente del modelo y su eventual incorporación se evaluará posteriormente en función de la mejora que
    pueda generar en el ajuste del modelo.'''
    
#_________________________________________________________________________________________________________________________
# Preparación de datos para modelar
#_________________________________________________________________________________________________________________________

''' En este bloque se prepara la base de datos para la estimación de un modelo de probabilidad de pago a nivel cliente. Inicialmente, se parte de una copia del 
    DataFrame consolidado y se eliminan observaciones con valores faltantes en las variables clave de saldo, mora y tipo de cliente, garantizando consistencia 
    en la muestra de modelación. Posteriormente, se construyen transformaciones y variables derivadas con el objetivo de capturar relaciones no lineales y efectos
    de interacción relevantes desde el punto de vista económico, tales como transformaciones logarítmicas de saldo y días de mora, razones e interacciones entre
    saldo y mora, e indicadores binarios de mora extrema y saldos altos. Luego, las variables categóricas se convierten a formato numérico mediante dummys. 
    Con el conjunto de variables explicativas definido, los datos se dividen en muestras de entrenamiento y prueba de forma estratificada
    para preservar el desbalance de la variable objetivo. Finalmente, las variables se estandarizan para asegurar comparabilidad de escalas, se incorpora el 
    intercepto para mantener la integridad del espacio columna de la matriz de diseño y se calculan ponderaciones para la muestra de entrenamiento, con el fin de
    corregir el desbalance entre individuos que pagan y no pagan durante la estimación.'''
    
df_modelo = df_cliente.copy()
df_modelo = df_modelo.dropna(subset=['SALDO_TOTAL_CLIENTE', 'DIAS_MORA', 'TIPO_CLIENTE'])

df_modelo['LOG_SALDO'] = np.log1p(df_modelo['SALDO_TOTAL_CLIENTE']) # Para capturar relacione no lineales
df_modelo['LOG_DIAS_MORA'] = np.log1p(df_modelo['DIAS_MORA']) # Identificara clientes de alta deuda pero poca mora
df_modelo['RATIO_SALDO_MORA'] = df_modelo['SALDO_TOTAL_CLIENTE'] / (df_modelo['DIAS_MORA'] + 1) 
df_modelo['SALDO_X_MORA'] = df_modelo['SALDO_TOTAL_CLIENTE'] * df_modelo['DIAS_MORA'] # Evaluar interacción entre saldos y mora
df_modelo['LOG_SALDO_X_MORA'] = np.log1p(df_modelo['SALDO_X_MORA'])
df_modelo['MORA_EXTREMA'] = (df_modelo['DIAS_MORA'] > 540).astype(int)
df_modelo['SALDO_ALTO'] = (df_modelo['SALDO_TOTAL_CLIENTE'] > df_modelo['SALDO_TOTAL_CLIENTE'].median()).astype(int)

df_modelo = pd.get_dummies(df_modelo, columns=['TIPO_CLIENTE'], drop_first=True, dtype=float)
vars_modelo = [
    'LOG_SALDO',
    'LOG_DIAS_MORA',
    'RATIO_SALDO_MORA',
    'LOG_SALDO_X_MORA',
    'MORA_EXTREMA',
    'SALDO_ALTO'] + [col for col in df_modelo.columns if col.startswith('TIPO_CLIENTE_')]

X = df_modelo[vars_modelo].copy()
y = df_modelo['PAGO_REALIZA'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

freq_train = y_train.value_counts(normalize=True)
weights_train = y_train.map({0: 1/freq_train[0], 1: 1/freq_train[1]})

#_________________________________________________________________________________________________________________________
# Ajuste del modelo probit
#_________________________________________________________________________________________________________________________
''' Se ajusta el modelo utilizando la muestra de entrenamiento, incorporando ponderaciones por observación con el fin de mitigar el desbalance existente entre 
    clientes pagadores y no pagadores, y permitiendo un mayor número de iteraciones para asegurar la convergencia del algoritmo de máxima verosimilitud. 
    Subsecuentemente, se presenta el resumen estadístico del modelo, que incluye los coeficientes estimados, su significancia y métricas globales de ajuste. 
    Finalmente, el modelo se utiliza para generar probabilidades predichas tanto en la muestra de entrenamiento como en la de prueba, para evaluar
    su capacidad predictiva y comparar el desempeño dentro y fuera de la muestra.'''
    
probit_model = sm.Probit(y_train, X_train_scaled)
probit_results = probit_model.fit(weights=weights_train, disp=True, maxiter=100)
print(probit_results.summary())

y_train_prob = probit_results.predict(X_train_scaled)
y_test_prob = probit_results.predict(X_test_scaled)

#_________________________________________________________________________________________________________________________
# Evaluación
#_________________________________________________________________________________________________________________________
''' Se calcula el estadístico ROC-AUC tanto en la muestra de entrenamiento como en la de prueba, y se compara la diferencia entre ambos valores como criterio
    para detectar sobreajuste; una diferencia pequeña sugiere que el modelo generaliza adecuadamente fuera de la muestra. Posteriormente, se construye la curva 
    precision–recall sobre el conjunto de prueba y se calcula el puntaje F1 para distintos umbrales de decisión, seleccionando aquel que maximiza dicho puntaje
    como umbral óptimo de clasificación. Finalmente, utilizando este umbral, se generan las predicciones binarias, se calcula la matriz de confusión y se presenta
    el reporte de clasificación, lo que permite evaluar de manera integral la capacidad del modelo para identificar correctamente a los clientes que realizan pagos
    frente a aquellos que no lo hacen, priorizando métricas adecuadas para el problema desbalanceado.'''

roc_auc_train = roc_auc_score(y_train, y_train_prob)
roc_auc_test = roc_auc_score(y_test, y_test_prob)
diferencia = roc_auc_train - roc_auc_test # Dado que es un valor inferior a 0.05, se considera que no hay sobreajuste.

# Umbral
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = pr_thresholds[best_threshold_idx] if best_threshold_idx < len(pr_thresholds) else 0.5

# Umbral
y_test_pred = (y_test_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_test_pred)
reporte_clasificacion = classification_report(y_test, y_test_pred, target_names=['No Paga', 'Paga'], zero_division=0)

''' Los resultados del modelo muestran una capacidad discriminatoria moderada, con valores de ROC-AUC de 0.61 tanto en la muestra de entrenamiento como en la de
    prueba, y una diferencia prácticamente nula entre ambos, lo que indica ausencia de sobreajuste y una adecuada generalización fuera de la muestra. El umbral
    óptimo de clasificación, determinado a partir de la maximización del puntaje F1, se sitúa alrededor de 0.14, reflejando el fuerte desbalance de la variable
    objetivo y la necesidad de utilizar un umbral inferior al convencional de 0.5. Bajo este umbral, el modelo logra identificar correctamente el 42 % de los
    clientes que realizan pagos (recall de la clase positiva), aunque con una precisión relativamente baja del 27 %, lo cual es consistente con un enfoque orientado
    a priorizar la detección de pagadores en un contexto de cobranza. La matriz de confusión confirma este compromiso entre precisión y cobertura, mostrando una
    mejora sustancial en la identificación de clientes pagadores a costa de un mayor número de falsos positivos. En conjunto, los resultados sugieren que el modelo
    es útil como una herramienta de priorización y segmentación operativa, más que como un clasificador determinístico, permitiendo focalizar esfuerzos de cobranza
    en clientes con mayor probabilidad de realizar al menos un pago.'''

#_________________________________________________________________________________________________________________________
# VISUALIZACIONES
#_________________________________________________________________________________________________________________________

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Curvas ROC
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

axes[0, 0].plot(fpr_train, tpr_train, label=f'Train (AUC={roc_auc_train:.3f})', linewidth=2.5)
axes[0, 0].plot(fpr_test, tpr_test, label=f'Test (AUC={roc_auc_test:.3f})', linewidth=2.5)
axes[0, 0].plot([0,1], [0,1], 'k--', label='Random', alpha=0.5)
axes[0, 0].set_xlabel('False Positive Rate', fontsize=11)
axes[0, 0].set_ylabel('True Positive Rate', fontsize=11)
axes[0, 0].set_title('Curva ROC', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# 2. Distribución de probabilidades
axes[0, 1].hist(y_test_prob[y_test==0], bins=40, alpha=0.6, label='No Pago', color='salmon', edgecolor='black')
axes[0, 1].hist(y_test_prob[y_test==1], bins=40, alpha=0.6, label='Pago', color='lightgreen', edgecolor='black')
axes[0, 1].axvline(best_threshold, color='red', linestyle='--', linewidth=2.5, label=f'Umbral={best_threshold:.3f}')
axes[0, 1].set_xlabel('Probabilidad Predicha', fontsize=11)
axes[0, 1].set_ylabel('Frecuencia', fontsize=11)
axes[0, 1].set_title('Distribución de Probabilidades (Test)', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# 3. Precision-Recall Curve
axes[1, 0].plot(recall, precision, linewidth=2.5, color='purple')
axes[1, 0].scatter(recall[best_threshold_idx], precision[best_threshold_idx], 
                   color='red', s=100, zorder=5, label=f'Mejor F1 ({f1_scores[best_threshold_idx]:.3f})')
axes[1, 0].set_xlabel('Recall', fontsize=11)
axes[1, 0].set_ylabel('Precision', fontsize=11)
axes[1, 0].set_title('Curva Precision-Recall', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)

# 4. Importancia de variables (coeficientes)
coef_df = pd.DataFrame({
    'Variable': probit_results.params.index,
    'Coeficiente': probit_results.params.values,
    'P-valor': probit_results.pvalues.values
})
coef_df = coef_df[coef_df['Variable'] != 'const'].sort_values('Coeficiente', key=abs, ascending=True)

colors = ['green' if x > 0 else 'red' for x in coef_df['Coeficiente']]
axes[1, 1].barh(coef_df['Variable'], coef_df['Coeficiente'], color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(0, color='black', linewidth=1)
axes[1, 1].set_xlabel('Coeficiente', fontsize=11)
axes[1, 1].set_title('Importancia de Variables', fontsize=13, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

''' Curva ROC.
    La curva ROC muestra un desempeño consistente entre las muestras de entrenamiento y prueba, con valores de AUC cercanos a 0.61 en ambos casos. Esto indica que
    el modelo posee una capacidad discriminatoria moderada, superior al azar pero lejos de una separación perfecta entre clientes pagadores y no pagadores.
    La cercanía entre ambas curvas confirma la ausencia de sobreajuste, lo que sugiere que el modelo generaliza adecuadamente fuera de la muestra. 

    Distribución de probabilidades predichas.
    La distribución de probabilidades en la muestra de prueba evidencia una superposición considerable entre clientes pagadores y no pagadores, lo cual explica el
    AUC moderado observado. No obstante, se aprecia un desplazamiento hacia la derecha en la distribución de los clientes que realizan pagos, lo que indica que el
    modelo asigna, en promedio, probabilidades más altas a este grupo. El umbral óptimo seleccionado (0.14), representado por la línea vertical, es coherente con 
    el fuerte desbalance del problema y permite capturar una mayor proporción de pagadores a costa de aceptar más falsos positivos.

    Curva Precision–Recall.
    La curva Precision–Recall confirma que el modelo enfrenta un trade-off claro entre precisión y cobertura de la clase positiva. El punto marcado corresponde al
    umbral que maximiza el puntaje F1, con un valor cercano a 0.33, reflejando una mejora sustancial frente a una clasificación aleatoria en un entorno altamente
    desbalanceado. Este resultado indica que, si bien la precisión para identificar pagadores es limitada, el modelo logra un nivel de recall relevante, lo cual 
    es permite identificar clientes con mayor probabilidad de pago para acciones de cobranza focalizadas.

    Importancia de variables.
    El gráfico de coeficientes evidencia que la variable con mayor impacto en la probabilidad de pago es TIPO_CLIENTE_MULTIPRODUCTO, lo que refuerza la
    interpretación previa de que los clientes con mayor complejidad financiera y más productos presentan una mayor propensión a pagar. Las transformaciones
    logarítmicas de saldo y mora, así como sus interacciones, aportan información adicional aunque con efectos de menor magnitud. En contraste, variables como
    MORA_EXTREMA y SALDO_ALTO muestran efectos negativos o cercanos a cero, sugiriendo que su contribución marginal es limitada.'''
    
#_________________________________________________________________________________________________________________________
# Análisis de valor 
#_________________________________________________________________________________________________________________________
TN, FP, FN, TP = cm.ravel()
total = TN + FP + FN + TP
costo_contacto = 5000  # COP: costo de contactar a un cliente
ingreso_promedio_pago = df_modelo['SALDO_TOTAL_CLIENTE'].median() * 0.1  # 10% del saldo

print(f'\nSupuestos:')
print(f'  Costo de contactar un cliente: ${costo_contacto:,.0f} COP')
print(f'  Ingreso promedio por pago:     ${ingreso_promedio_pago:,.0f} COP')

valor_TP = ingreso_promedio_pago - costo_contacto  # Contacto exitoso
valor_FP = -costo_contacto  # Contacto sin resultado
valor_TN = 0  # Correcto no contactar
valor_FN = 0  # Oportunidad perdida (pero sin costo directo)

valor_modelo = (TP * valor_TP) + (FP * valor_FP) + (TN * valor_TN) + (FN * valor_FN)

print(f'\n💰 Valor económico del modelo:')
print(f'  Verdaderos Positivos:  {TP:,} × ${valor_TP:,.0f} = ${TP*valor_TP:,.0f}')
print(f'  Falsos Positivos:      {FP:,} × ${valor_FP:,.0f} = ${FP*valor_FP:,.0f}')
print(f'  Valor neto:                                    ${valor_modelo:,.0f}')

clientes_contactados = TP + FP
tasa_exito = TP / clientes_contactados if clientes_contactados > 0 else 0
print(f'\n🎯 Eficiencia operativa:')
print(f'  Clientes a contactar:  {clientes_contactados:,} ({clientes_contactados/total:.1%} del total)')
print(f'  Tasa de éxito:         {tasa_exito:.1%}')
print(f'  Pagadores capturados:  {TP:,} de {TP+FN:,} ({TP/(TP+FN):.1%})')

''' A partir de la matriz de confusión, se construye un ejercicio de valoración económica bajo supuestos: un costo fijo por contacto y un ingreso promedio esperado
    por pago. Con estos supuestos, se calcula el valor económico asociado a cada tipo de decisión (verdaderos y falsos positivos), mostrando que el modelo genera un
    valor neto positivo cercano a 71 mil millones de COP al focalizar los contactos en clientes con mayor probabilidad de pago. En este sentido, el aporte principal
    del modelo no es maximizar el ingreso absoluto, sino mejorar la eficiencia, ya que permite contactar solo al 24.7 % de los clientes, capturando el 41.5 % de los
    pagadores, con una tasa de éxito del 27.2 %. En conjunto, el análisis muestra que el modelo es especialmente útil como herramienta de priorización y optimización
    de recursos, más que como una regla rígida de decisión global.'''
    
#_________________________________________________________________________________________________________________________
# Preguntas
#_________________________________________________________________________________________________________________________

# ¿Por qué un modelo Probir?
''' Se optó por un modelo Probit debido a varias ventajas metodológicas y prácticas. En primer lugar, sus coeficientes admiten una interpretación económica clara en
    términos de efectos sobre la probabilidad de ocurrencia del evento de interés, lo que facilita el análisis y la comunicación de resultados. Adicionalmente, el
    enfoque Probit permite realizar inferencia estadística formal, incluyendo pruebas de significancia, intervalos de confianza y contrastes de hipótesis, aspectos
    fundamentales para validar la robustez del modelo. Asimismo, se trata de un modelo parsimonioso, con un número reducido de parámetros, lo que contribuye a la
    estabilidad de las estimaciones y reduce el riesgo de sobreajuste. Finalmente, el modelo Probit es un estándar en la econometría aplicada y es ampliamente
    utilizado en análisis de riesgo crediticio y estudios de comportamiento financiero, lo que respalda su idoneidad para el problema planteado.'''
    
# ¿Alternativas?
''' LightGBM es una alternativa para problemas binarios desbalanceados, ya que se establecio una line base con el modelo Proit, se desarrollara el LightGBM y se
    evaluar su contriución.'''

#_________________________________________________________________________________________________________________________
# Guardado
#_________________________________________________________________________________________________________________________
modelo_pago = {
    'modelo': probit_results,
    'scaler': scaler,
    'vars_modelo': vars_modelo,
    'threshold': best_threshold,
    'roc_auc_train': roc_auc_train,
    'roc_auc_test': roc_auc_test}

with open(ruta / 'Códigos/Modelo_1_Posibilidad_pago_Probit_v2.pkl', 'wb') as f:
    pickle.dump(modelo_pago, f)