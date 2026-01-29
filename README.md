# Modelo de posibilidad de pago
En la presente carpeta se encuentran cuatro subcarpetas, organizadas de la siguiente manera:

------------------------------------------------------------------------------------------------------------------------------
# 1. Códigos:

Contiene los scripts desarrollados en Python, tanto para la depuración, estandarización y unión de las bases de datos, como para la implementación de los tres modelos de posibilidad de pago (Probit, LightGBM y Random Forest).
Todos los códigos se encuentran debidamente comentados, con el objetivo de evitar redundancias y dejar explícito el propósito de cada bloque de instrucciones. Las explicaciones no se repiten entre archivos; por lo tanto, en caso de requerir una comprensión progresiva del flujo de trabajo, se recomienda revisar los scripts en el siguiente orden:

	- Manipulación_Bases_Datos.py

	- Modelo_1_Posibilidad_pago_Probit_v2.py

	- Modelo_1_Posibilidad_pago_LightGBM.py

En esta misma carpeta se incluyen los archivos pickle con los objetos y resultados más relevantes de los modelos, lo que permite su reutilización sin necesidad de ejecutar nuevamente los scripts.

------------------------------------------------------------------------------------------------------------------------------
# 2. Datos_Originales
Contiene las bases de datos originales proporcionadas para el desarrollo del ejercicio, sin ningún tipo de manipulación o transformación.

------------------------------------------------------------------------------------------------------------------------------
# 3. Documentación
Incluye los documentos explicativos asociados al desarrollo del ejercicio, así como el diccionario de datos, donde se describen las variables utilizadas y su significado.

------------------------------------------------------------------------------------------------------------------------------
# 4. Resultados
Contiene los principales productos derivados del proceso de depuración, modelamiento y análisis. En particular:

	- df_evolucion_enriquecida.txt: Base de datos final obtenida tras la limpieza, estandarización e integración de las fuentes originales.

	- Manipulación de datos - desarrollo.txt: Documento que describe detalladamente el proceso seguido para construir la base df_evolucion_enriquecida.txt.

	- Análisis y resultados - Modelo 1 - Modelo de posibilidad de pago.txt: Documento que presenta el desarrollo metodológico y el análisis de resultados del modelo de posibilidad de pago. En este archivo se justifica la elección de los modelos Probit, LightGBM y Random Forest, se describen sus resultados individuales y se realiza una comparación entre ellos. Se concluye que el modelo Probit ofrece el mejor balance entre simplicidad, interpretabilidad y desempeño, mostrando resultados comparables a sus alternativas de machine learning con menor complejidad.

	- Modelo_1_resultados_evaluación_LightGBM_Random_Forest.png: Visualizaciones diseñadas para facilitar la interpretación de las métricas de desempeño de los modelos LightGBM y Random Forest.

	- Modelo_1_resultados_evaluación_probit.png: Visualizaciones asociadas a la evaluación del modelo Probit.
	- PRUEBA 4 - PERCEPCIONES DEL RETO.docx: Contiene las respuestas a las preguntas realizadas en el último inciso de la prueba.

<img width="1074" height="937" alt="Modelo_1_resultados_evaluación_LightGBM_Random_Forest" src="https://github.com/user-attachments/assets/4e5245f5-ee7d-466d-8000-70009c6cebdb" />

------------------------------------------------------------------------------------------------------------------------------

# Nota:
Si bien cada script contiene análisis y comentarios sobre los resultados y las decisiones tomadas durante el desarrollo del ejercicio, el documento Análisis y resultados - Modelo 1 - Modelo de posibilidad de pago.txt presenta un resumen claro, conciso y estructurado de todo el proceso, por lo que se recomienda como principal referencia para la comprensión integral del trabajo realizado.
