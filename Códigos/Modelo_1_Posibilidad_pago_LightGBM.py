# -*- coding: utf-8 -*-
'''
Created on Wed Jan 28 19:51:55 2026

@author: Andrés Felipe Arévalo Arévalo
'''

#_________________________________________________________________________________________________________________________
# Modulos
#_________________________________________________________________________________________________________________________
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle

# _________________________________________________________________________________________________
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
# Preparación de datos para modelar
#_________________________________________________________________________________________________________________________
df_modelo = df_cliente.copy()
df_modelo = df_modelo.dropna(subset=['SALDO_TOTAL_CLIENTE', 'DIAS_MORA', 'TIPO_CLIENTE'])
df_modelo['LOG_SALDO'] = np.log1p(df_modelo['SALDO_TOTAL_CLIENTE'])
df_modelo['LOG_DIAS_MORA'] = np.log1p(df_modelo['DIAS_MORA'])
df_modelo['RATIO_SALDO_MORA'] = df_modelo['SALDO_TOTAL_CLIENTE'] / (df_modelo['DIAS_MORA'] + 1)
df_modelo['SALDO_X_MORA'] = df_modelo['SALDO_TOTAL_CLIENTE'] * df_modelo['DIAS_MORA']
df_modelo['LOG_SALDO_X_MORA'] = np.log1p(df_modelo['SALDO_X_MORA'])
df_modelo['MORA_EXTREMA'] = (df_modelo['DIAS_MORA'] > 540).astype(int)
df_modelo['SALDO_ALTO'] = (df_modelo['SALDO_TOTAL_CLIENTE'] > df_modelo['SALDO_TOTAL_CLIENTE'].median()).astype(int)
df_modelo['DIAS_MORA_CUADRADO'] = df_modelo['DIAS_MORA'] ** 2
df_modelo['SQRT_DIAS_MORA'] = np.sqrt(df_modelo['DIAS_MORA'])
df_modelo['SALDO_PER_SQRT_MORA'] = df_modelo['SALDO_TOTAL_CLIENTE'] / (np.sqrt(df_modelo['DIAS_MORA']) + 1)

le_tipo = LabelEncoder()
df_modelo['TIPO_CLIENTE_ENCODED'] = le_tipo.fit_transform(df_modelo['TIPO_CLIENTE'])

vars_numericas = [
    'SALDO_TOTAL_CLIENTE',
    'DIAS_MORA',
    'LOG_SALDO',
    'LOG_DIAS_MORA',
    'RATIO_SALDO_MORA',
    'LOG_SALDO_X_MORA',
    'MORA_EXTREMA',
    'SALDO_ALTO',
    'DIAS_MORA_CUADRADO',
    'SQRT_DIAS_MORA',
    'SALDO_PER_SQRT_MORA']
vars_categoricas = ['TIPO_CLIENTE_ENCODED']
vars_modelo = vars_numericas + vars_categoricas
X = df_modelo[vars_modelo].copy()
y = df_modelo['PAGO_REALIZA'].copy()

mask_validos = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))
X = X[mask_validos]
y = y[mask_validos]
print(f'Registros válidos: {len(X)}')
print(f'Distribución: No paga={len(y[y==0])} ({len(y[y==0])/len(y):.1%}) | Paga={len(y[y==1])} ({len(y[y==1])/len(y):.1%})')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#_________________________________________________________________________________________________________________________
# Ajuste del modelo LIGHTGBM 
#_________________________________________________________________________________________________________________________
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'scale_pos_weight': scale_pos_weight,
    'max_depth': 6,
    'verbose': -1,
    'random_state': 42}
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=['TIPO_CLIENTE_ENCODED'])
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=['TIPO_CLIENTE_ENCODED'])

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_test],
    valid_names=['train', 'test'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)    ])

print(f'Mejor iteración: {lgb_model.best_iteration}')
y_train_prob_lgb = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)
y_test_prob_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

# _________________________________________________________________________________________________________________________
# Evaluación
#_________________________________________________________________________________________________________________________

roc_auc_train_lgb = roc_auc_score(y_train, y_train_prob_lgb)
roc_auc_test_lgb = roc_auc_score(y_test, y_test_prob_lgb)

print(f'ROC-AUC Train: {roc_auc_train_lgb:.4f}')
print(f'ROC-AUC Test:  {roc_auc_test_lgb:.4f}')
print(f'Diferencia:    {abs(roc_auc_train_lgb - roc_auc_test_lgb):.4f}')
print(f'MEJORA vs Probit: +{(roc_auc_test_lgb - 0.6125)*100:.2f} puntos porcentuales')

# _________________________________________________________________________________________________________________________
# MODELO 2: RANDOM FOREST
#  _________________________________________________________________________________________________________________________

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0)

rf_model.fit(X_train, y_train)
y_train_prob_rf = rf_model.predict_proba(X_train)[:, 1]
y_test_prob_rf = rf_model.predict_proba(X_test)[:, 1]
roc_auc_train_rf = roc_auc_score(y_train, y_train_prob_rf)
roc_auc_test_rf = roc_auc_score(y_test, y_test_prob_rf)


print(f'ROC-AUC Train: {roc_auc_train_rf:.4f}')
print(f'ROC-AUC Test: {roc_auc_test_rf:.4f}')
print(f'Diferencia: {abs(roc_auc_train_rf - roc_auc_test_rf):.4f}') # <- Sobreajuste

# _________________________________________________________________________________________________________________________
# COMPARACIÓN DE MODELOS
# _________________________________________________________________________________________________________________________

resultados = pd.DataFrame({
    'Modelo': ['Probit', 'LightGBM', 'Random Forest'],
    'ROC-AUC Train': [0.6102, roc_auc_train_lgb, roc_auc_train_rf],
    'ROC-AUC Test': [0.6125, roc_auc_test_lgb, roc_auc_test_rf],
    'Overfitting': [0.0023, abs(roc_auc_train_lgb - roc_auc_test_lgb), abs(roc_auc_train_rf - roc_auc_test_rf)]
})

resultados['Mejora vs Probit'] = (resultados['ROC-AUC Test'] - 0.6125) * 100
resultados = resultados.sort_values('ROC-AUC Test', ascending=False)
print(resultados.to_string(index=False))

# _________________________________________________________________________________________________________________________
# Visualizaciones
# _________________________________________________________________________________________________________________________

# LightGBM
precision_lgb, recall_lgb, pr_thresholds_lgb = precision_recall_curve(y_test, y_test_prob_lgb)
f1_scores_lgb = 2 * (precision_lgb * recall_lgb) / (precision_lgb + recall_lgb + 1e-10)
best_threshold_idx_lgb = np.argmax(f1_scores_lgb)
best_threshold_lgb = pr_thresholds_lgb[best_threshold_idx_lgb] if best_threshold_idx_lgb < len(pr_thresholds_lgb) else 0.5

y_test_pred_lgb = (y_test_prob_lgb >= best_threshold_lgb).astype(int)
cm_lgb = confusion_matrix(y_test, y_test_pred_lgb)
TN_lgb, FP_lgb, FN_lgb, TP_lgb = cm_lgb.ravel()

# Random Forest
precision_rf, recall_rf, pr_thresholds_rf = precision_recall_curve(y_test, y_test_prob_rf)
f1_scores_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf + 1e-10)
best_threshold_idx_rf = np.argmax(f1_scores_rf)
best_threshold_rf = pr_thresholds_rf[best_threshold_idx_rf] if best_threshold_idx_rf < len(pr_thresholds_rf) else 0.5

y_test_pred_rf = (y_test_prob_rf >= best_threshold_rf).astype(int)
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
TN_rf, FP_rf, FN_rf, TP_rf = cm_rf.ravel()

# LightGBM
importance_lgb = lgb_model.feature_importance(importance_type='gain')
feature_importance_lgb = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': importance_lgb
}).sort_values('Importancia', ascending=False)

# Random Forest
feature_importance_rf = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': rf_model.feature_importances_
}).sort_values('Importancia', ascending=False)


fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

#  _________________________________________________________________________________________________________________________
# Curvas ROC comparadas
#  _________________________________________________________________________________________________________________________
ax1 = fig.add_subplot(gs[0, :2])

fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_test_prob_lgb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_prob_rf)

ax1.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC={roc_auc_test_lgb:.4f})', 
         linewidth=3, color='#2E86AB', alpha=0.9)
ax1.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_test_rf:.4f})', 
         linewidth=3, color='#A23B72', alpha=0.9)
ax1.plot([0,1], [0,1], 'k--', label='Random Classifier', alpha=0.4, linewidth=2)

ax1.fill_between(fpr_lgb, tpr_lgb, alpha=0.2, color='#2E86AB')
ax1.fill_between(fpr_rf, tpr_rf, alpha=0.2, color='#A23B72')

ax1.set_xlabel('False Positive Rate (1 - Especificidad)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate (Sensibilidad)', fontsize=12, fontweight='bold')
ax1.set_title('Curvas ROC - Comparación', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='lower right', framealpha=0.95)
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])

#  _________________________________________________________________________________________________________________________
# Comparación
#  _________________________________________________________________________________________________________________________
ax2 = fig.add_subplot(gs[0, 2])

# Top 8 variables de cada modelo
top_lgb = feature_importance_lgb.head(8)
top_rf = feature_importance_rf.head(8)

# Combinar y normalizar
all_vars = list(set(top_lgb['Variable'].tolist() + top_rf['Variable'].tolist()))
comparison_df = pd.DataFrame({'Variable': all_vars})

for var in all_vars:
    lgb_val = feature_importance_lgb[feature_importance_lgb['Variable'] == var]['Importancia'].values
    rf_val = feature_importance_rf[feature_importance_rf['Variable'] == var]['Importancia'].values
    
    comparison_df.loc[comparison_df['Variable'] == var, 'LightGBM'] = lgb_val[0] if len(lgb_val) > 0 else 0
    comparison_df.loc[comparison_df['Variable'] == var, 'RF'] = rf_val[0] if len(rf_val) > 0 else 0

# Normalizar
comparison_df['LightGBM'] = comparison_df['LightGBM'] / comparison_df['LightGBM'].max()
comparison_df['RF'] = comparison_df['RF'] / comparison_df['RF'].max()

# Ordenar por importancia promedio
comparison_df['Avg'] = (comparison_df['LightGBM'] + comparison_df['RF']) / 2
comparison_df = comparison_df.sort_values('Avg', ascending=True).head(10)

y_pos = np.arange(len(comparison_df))
width = 0.35

ax2.barh(y_pos - width/2, comparison_df['LightGBM'], width, 
         label='LightGBM', color='#2E86AB', edgecolor='black', alpha=0.8)
ax2.barh(y_pos + width/2, comparison_df['RF'], width, 
         label='Random Forest', color='#A23B72', edgecolor='black', alpha=0.8)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(comparison_df['Variable'], fontsize=9)
ax2.set_xlabel('Importancia Normalizada', fontsize=10, fontweight='bold')
ax2.set_title('Importancia de Variables (Top 10)', fontsize=11, fontweight='bold', pad=10)
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(alpha=0.3, axis='x', linestyle='--')
ax2.set_xlim([0, 1.1])

#  _________________________________________________________________________________________________________________________
# Distribuciones de probabilidad - LIGHTGBM
#  _________________________________________________________________________________________________________________________
ax3 = fig.add_subplot(gs[1, 0])

ax3.hist(y_test_prob_lgb[y_test==0], bins=50, alpha=0.6, label='No Pago', 
         color='#E63946', edgecolor='black', linewidth=0.5)
ax3.hist(y_test_prob_lgb[y_test==1], bins=50, alpha=0.6, label='Pago', 
         color='#06FFA5', edgecolor='black', linewidth=0.5)
ax3.axvline(best_threshold_lgb, color='#F77F00', linestyle='--', linewidth=3, 
            label=f'Umbral={best_threshold_lgb:.3f}')

ax3.set_xlabel('Probabilidad Predicha', fontsize=10, fontweight='bold')
ax3.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
ax3.set_title('LightGBM - Distribución de Probabilidades', fontsize=11, fontweight='bold', pad=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, linestyle='--')

# _________________________________________________________________________________________________________________________
# Distribuciones de probabilidad  - RANDOM FOREST
# _________________________________________________________________________________________________________________________
ax4 = fig.add_subplot(gs[1, 1])

ax4.hist(y_test_prob_rf[y_test==0], bins=50, alpha=0.6, label='No Pago', 
         color='#E63946', edgecolor='black', linewidth=0.5)
ax4.hist(y_test_prob_rf[y_test==1], bins=50, alpha=0.6, label='Pago', 
         color='#06FFA5', edgecolor='black', linewidth=0.5)
ax4.axvline(best_threshold_rf, color='#F77F00', linestyle='--', linewidth=3, 
            label=f'Umbral={best_threshold_rf:.3f}')

ax4.set_xlabel('Probabilidad Predicha', fontsize=10, fontweight='bold')
ax4.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
ax4.set_title('Random Forest - Distribución de Probabilidades', fontsize=11, fontweight='bold', pad=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3, linestyle='--')

# _________________________________________________________________________________________________________________________
# Curvas RECALL
#_________________________________________________________________________________________________________________________
ax5 = fig.add_subplot(gs[1, 2])

ax5.plot(recall_lgb, precision_lgb, linewidth=3, color='#2E86AB', 
         label=f'LightGBM (AP={average_precision_score(y_test, y_test_prob_lgb):.3f})', alpha=0.9)
ax5.plot(recall_rf, precision_rf, linewidth=3, color='#A23B72', 
         label=f'Random Forest (AP={average_precision_score(y_test, y_test_prob_rf):.3f})', alpha=0.9)

ax5.scatter(recall_lgb[best_threshold_idx_lgb], precision_lgb[best_threshold_idx_lgb], 
           color='#2E86AB', s=200, zorder=5, edgecolors='black', linewidth=2,
           marker='o', label=f'LGB F1={f1_scores_lgb[best_threshold_idx_lgb]:.3f}')
ax5.scatter(recall_rf[best_threshold_idx_rf], precision_rf[best_threshold_idx_rf], 
           color='#A23B72', s=200, zorder=5, edgecolors='black', linewidth=2,
           marker='s', label=f'RF F1={f1_scores_rf[best_threshold_idx_rf]:.3f}')

ax5.set_xlabel('Recall (Sensibilidad)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Precision', fontsize=10, fontweight='bold')
ax5.set_title('Curvas Precision-Recall', fontsize=11, fontweight='bold', pad=10)
ax5.legend(fontsize=8, loc='best')
ax5.grid(alpha=0.3, linestyle='--')
ax5.set_xlim([-0.02, 1.02])
ax5.set_ylim([-0.02, 1.02])

# _________________________________________________________________________________________________________________________
# Matrices de confición - LADO A LADO
# _________________________________________________________________________________________________________________________

# LightGBM
ax6 = fig.add_subplot(gs[2, 0])
sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Blues', ax=ax6, 
            xticklabels=['No Paga', 'Paga'], yticklabels=['No Paga', 'Paga'],
            cbar_kws={'label': 'Frecuencia'}, annot_kws={'size': 11, 'weight': 'bold'})
ax6.set_ylabel('Real', fontsize=11, fontweight='bold')
ax6.set_xlabel('Predicho', fontsize=11, fontweight='bold')
ax6.set_title(f'LightGBM - Matriz de Confusión\n(Umbral={best_threshold_lgb:.3f})', 
              fontsize=11, fontweight='bold', pad=10)

# Random Forest
ax7 = fig.add_subplot(gs[2, 1])
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Purples', ax=ax7, 
            xticklabels=['No Paga', 'Paga'], yticklabels=['No Paga', 'Paga'],
            cbar_kws={'label': 'Frecuencia'}, annot_kws={'size': 11, 'weight': 'bold'})
ax7.set_ylabel('Real', fontsize=11, fontweight='bold')
ax7.set_xlabel('Predicho', fontsize=11, fontweight='bold')
ax7.set_title(f'Random Forest - Matriz de Confusión\n(Umbral={best_threshold_rf:.3f})', 
              fontsize=11, fontweight='bold', pad=10)

# _________________________________________________________________________________________________________________________
# Comparación de métricas
# _________________________________________________________________________________________________________________________
ax8 = fig.add_subplot(gs[2, 2])

# Calcular todas las métricas
metricas_comp = pd.DataFrame({
    'Modelo': ['LightGBM', 'Random Forest'],
    'ROC-AUC': [roc_auc_test_lgb, roc_auc_test_rf],
    'Precision': [TP_lgb/(TP_lgb+FP_lgb), TP_rf/(TP_rf+FP_rf)],
    'Recall': [TP_lgb/(TP_lgb+FN_lgb), TP_rf/(TP_rf+FN_rf)],
    'F1-Score': [f1_scores_lgb[best_threshold_idx_lgb], f1_scores_rf[best_threshold_idx_rf]],
    'Accuracy': [(TP_lgb+TN_lgb)/(TP_lgb+TN_lgb+FP_lgb+FN_lgb), 
                 (TP_rf+TN_rf)/(TP_rf+TN_rf+FP_rf+FN_rf)]})

# Graficar
x = np.arange(len(metricas_comp))
width = 0.13
metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, metric in enumerate(metrics):
    values_lgb = metricas_comp[metricas_comp['Modelo']=='LightGBM'][metric].values[0]
    values_rf = metricas_comp[metricas_comp['Modelo']=='Random Forest'][metric].values[0]
    
    ax8.bar(i - width/2, values_lgb, width*2, label='LightGBM' if i==0 else '', 
            color='#2E86AB', edgecolor='black', alpha=0.8)
    ax8.bar(i + width/2, values_rf, width*2, label='Random Forest' if i==0 else '', 
            color='#A23B72', edgecolor='black', alpha=0.8)
    
    # Añadir valores encima de las barras
    ax8.text(i - width/2, values_lgb + 0.02, f'{values_lgb:.3f}', 
             ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax8.text(i + width/2, values_rf + 0.02, f'{values_rf:.3f}', 
             ha='center', va='bottom', fontsize=8, fontweight='bold')

ax8.set_ylabel('Valor', fontsize=11, fontweight='bold')
ax8.set_title('Comparación de Métricas', fontsize=12, fontweight='bold', pad=10)
ax8.set_xticks(range(len(metrics)))
ax8.set_xticklabels(metrics, rotation=45, ha='right', fontsize=10)
ax8.legend(fontsize=10, loc='lower left', framealpha=0.95)
ax8.grid(alpha=0.3, axis='y', linestyle='--')
ax8.set_ylim([0, 1.1])

plt.suptitle('Análisis Comparativo: LightGBM vs Random Forest', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.show()

# _________________________________________________________________________________________________________________________
# Tablas resumen de métricas
# _________________________________________________________________________________________________________________________

resumen_detallado = pd.DataFrame({
    'Métrica': [
        'ROC-AUC (Train)',
        'ROC-AUC (Test)',
        'Overfitting (AUC)',
        'Precision',
        'Recall (Sensibilidad)',
        'Especificidad',
        'F1-Score',
        'Accuracy',
        'True Positives (TP)',
        'False Positives (FP)',
        'True Negatives (TN)',
        'False Negatives (FN)',
        'Umbral Óptimo',
        'AP Score'
    ],
    'LightGBM': [
        f'{roc_auc_train_lgb:.4f}',
        f'{roc_auc_test_lgb:.4f}',
        f'{abs(roc_auc_train_lgb - roc_auc_test_lgb):.4f}',
        f'{TP_lgb/(TP_lgb+FP_lgb):.4f}',
        f'{TP_lgb/(TP_lgb+FN_lgb):.4f}',
        f'{TN_lgb/(TN_lgb+FP_lgb):.4f}',
        f'{f1_scores_lgb[best_threshold_idx_lgb]:.4f}',
        f'{(TP_lgb+TN_lgb)/(TP_lgb+TN_lgb+FP_lgb+FN_lgb):.4f}',
        f'{TP_lgb:,}',
        f'{FP_lgb:,}',
        f'{TN_lgb:,}',
        f'{FN_lgb:,}',
        f'{best_threshold_lgb:.4f}',
        f'{average_precision_score(y_test, y_test_prob_lgb):.4f}'
    ],
    'Random Forest': [
        f'{roc_auc_train_rf:.4f}',
        f'{roc_auc_test_rf:.4f}',
        f'{abs(roc_auc_train_rf - roc_auc_test_rf):.4f}',
        f'{TP_rf/(TP_rf+FP_rf):.4f}',
        f'{TP_rf/(TP_rf+FN_rf):.4f}',
        f'{TN_rf/(TN_rf+FP_rf):.4f}',
        f'{f1_scores_rf[best_threshold_idx_rf]:.4f}',
        f'{(TP_rf+TN_rf)/(TP_rf+TN_rf+FP_rf+FN_rf):.4f}',
        f'{TP_rf:,}',
        f'{FP_rf:,}',
        f'{TN_rf:,}',
        f'{FN_rf:,}',
        f'{best_threshold_rf:.4f}',
        f'{average_precision_score(y_test, y_test_prob_rf):.4f}'
    ]
})

print(resumen_detallado.to_string(index=False))

print('\n' + '-'*80)
print('DIFERENCIAS CLAVE')
print('-'*80)

diff_auc = roc_auc_test_rf - roc_auc_test_lgb
diff_recall = (TP_rf/(TP_rf+FN_rf)) - (TP_lgb/(TP_lgb+FN_lgb))
diff_precision = (TP_rf/(TP_rf+FP_rf)) - (TP_lgb/(TP_lgb+FP_lgb))
diff_f1 = f1_scores_rf[best_threshold_idx_rf] - f1_scores_lgb[best_threshold_idx_lgb]

print(f'\nRandom Forest vs LightGBM:')
print(f'  • ROC-AUC:   {diff_auc:+.4f} ({"+" if diff_auc > 0 else ""}mejor RF)' if diff_auc > 0 else f'  • ROC-AUC:   {diff_auc:+.4f} (mejor LGB)')
print(f'  • Recall:    {diff_recall:+.4f} ({abs(diff_recall)*100:.2f}% puntos)')
print(f'  • Precision: {diff_precision:+.4f} ({abs(diff_precision)*100:.2f}% puntos)')
print(f'  • F1-Score:  {diff_f1:+.4f}')
print(f'  • Overfitting RF:  {abs(roc_auc_train_rf - roc_auc_test_rf):.4f} {" ALTO" if abs(roc_auc_train_rf - roc_auc_test_rf) > 0.05 else " OK"}')
print(f'  • Overfitting LGB: {abs(roc_auc_train_lgb - roc_auc_test_lgb):.4f} {" ALTO" if abs(roc_auc_train_lgb - roc_auc_test_lgb) > 0.05 else " OK"}')

# Pagadores capturados
pagadores_adicionales = TP_rf - TP_lgb
print(f' Pagadores adicionales capturados por RF: {pagadores_adicionales:+,}')


''' El modelo de Random Forest presenta el mayor ROC-AUC en la muestra de prueba. No obstante, la diferencia entre su desempeño en entrenamiento y prueba es de 
    0.078, lo cual constituye una señal clara de sobreajuste severo. En contraste, tanto LightGBM como el modelo Probit muestran un comportamiento considerablemente 
    más estable. LightGBM presenta una diferencia entre entrenamiento y prueba cercana a 0.01, mientras que el Probit exhibe una brecha prácticamente nula, 
    lo que evidencia una excelente capacidad de generalización. Cabe destacar que, en la muestra de prueba, los tres modelos alcanzan valores de ROC-AUC muy 
    similares, cercanos a 0.61, por lo que la ventaja numérica del Random Forest frente al Probit (aproximadamente 0.002 puntos) resulta estadísticamente 
    insignificante desde una perspectiva práctica.
    
    Ninguno de los modelos logra una capacidad predictiva elevada. Un ROC-AUC cercano a 0.61 indica que las variables disponibles contienen un poder explicativo 
    limitado para predecir el comportamiento de pago. Esta limitación obedece a la ausencia de información crítica que permita capturar adecuadamente los 
    determinantes del pago, como variables comportamentales más ricas o información temporal detallada.
    
    El Probit se perfila como la alternativa más equilibrada, al ofrecer un desempeño en prueba prácticamente idéntico al de los demás modelos, sin incurrir en
    sobreajuste, y permitiendo además una evaluación confiable de la importancia de las variables.'''
    
# _________________________________________________________________________________________________________________________
# Guardar
# _________________________________________________________________________________________________________________________

resultados_modelos = {
    'metadata': {
        'fecha': '2026-01-28',
        'autor': 'Andrés Felipe Arévalo Arévalo',
        'problema': 'Modelo de posibilidad de pago',
        'nivel_analisis': 'Cliente'
    },
    'vars_modelo': vars_modelo,
    'label_encoder_tipo_cliente': le_tipo,

    'modelos': {
        'lightgbm': lgb_model,
        'random_forest': rf_model
    },

    'predicciones': {
        'lgb': {
            'train_prob': y_train_prob_lgb,
            'test_prob': y_test_prob_lgb,
            'threshold': best_threshold_lgb
        },
        'rf': {
            'train_prob': y_train_prob_rf,
            'test_prob': y_test_prob_rf,
            'threshold': best_threshold_rf
        }
    },

    'metricas': {
        'lgb': {
            'roc_auc_train': roc_auc_train_lgb,
            'roc_auc_test': roc_auc_test_lgb,
            'confusion_matrix': cm_lgb,
            'precision': TP_lgb / (TP_lgb + FP_lgb),
            'recall': TP_lgb / (TP_lgb + FN_lgb),
            'f1_score': f1_scores_lgb[best_threshold_idx_lgb],
            'average_precision': average_precision_score(y_test, y_test_prob_lgb)
        },
        'rf': {
            'roc_auc_train': roc_auc_train_rf,
            'roc_auc_test': roc_auc_test_rf,
            'confusion_matrix': cm_rf,
            'precision': TP_rf / (TP_rf + FP_rf),
            'recall': TP_rf / (TP_rf + FN_rf),
            'f1_score': f1_scores_rf[best_threshold_idx_rf],
            'average_precision': average_precision_score(y_test, y_test_prob_rf)
        }
    },

    'tablas': {
        'comparacion_modelos': resultados,
        'resumen_detallado': resumen_detallado,
        'feature_importance_lgb': feature_importance_lgb,
        'feature_importance_rf': feature_importance_rf
    }
}

# Guardar pickle
ruta_pickle = ruta / 'Códigos/Modelo_1_Resultados_LightGBM_RandomForest.pkl'
with open(ruta_pickle, 'wb') as f:
    pickle.dump(resultados_modelos, f)