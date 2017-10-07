## Resumen

   Este es un análisis de el [Bank marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) del repositorio de la Universidad de California en Irvine.

Este consiste en información sobre una campaña de marketing directo de una institución bancaria portuguesa, en la cual se relevo el éxito o fracaso, es decir venta lograda o no, asi como tambien información de las personas contactadas. El objetivo es poder predecir que personas adquirirán el producto ofrecido, basado en los resultados pasados de la mencionada campaña. De esta manera los esfuerzos de la compañia pueden ser dirigidos mas eficientemente.

## Pasos del análisis

### Explorando las variables

Para empezar se chequea la distribución de las clases, en este caso el éxito o no en la venta.


- Se puede observar que es un problema con clases sesgadas. Es mucho mayor en número de casos en los que las personas contactadas no compraron el producto comparado a los casos de éxito. Se muestra en el gráfico de barras a la derecha la distribución de clase en la población total.<img align="right" width="60%" src="https://i.imgur.com/GDMNTZX.png">
<br><br>

- Se cuenta con las siguientes información sobre cada cliente: Edad, Trabajo, Estado civil, Educacion, Si tiene default, Si tiene prestamo para la vivienda, Medio de contacto, Si tiene prestamo personal, Dia contactado, Duración del llamado, Nro contactos en esta campaña, Dias desde ultimo contacto, Nro contactos previos a esta campaña y Éxito previo en campaña anterior.

  Se procede a investigar cada variable, se observa la distribución de los valores que toman estas y tambien se realiza tabulación cruzada con la variable objetivo.

  Por ejemplo esta es la distribución de edad de la población en un histograma.<img align="center" width="90%" src="https://i.imgur.com/mKIK3cm.png">

- De la tabulación cruzada se grafica el porcentaje de ejemplos de clase positiva (caso éxito) para cada posible valor de una variable categórica. Aquí se muestra un ejemplo para la variable trabajo, el color de las barras representa si el porcentaje de casos de éxito para ese valor de la variable, es menor (rojo) parecido (amarillo) o mayor (verde) al de la población general.<img align="center" width="90%" src="https://i.imgur.com/Ur0vFbK.png">

  Se observa por ejemplo que los trabajadores 'blue-collar' no son propensos a adquirir el producto ofrecido mientras que los 'student' si comparado a la población general. Contando con esta información y la distribución de valores de cada variable se transforman estas. Por ejemplo para la edad se agrupa la población en jóvenes, adultos y senior (denotado por 1 , 2 o 3 dependiendo la edad).<img align="right" width="60%" src="https://i.imgur.com/tLqsffX.png">
<br><br>
- Algunos puntos son excluídos por ser valores erráticos que afectan negativamente la performance de las técnicas utilizadas posteriormente. A la derecha se observa un boxplot de la variable con el balance de cada persona, se puede observar que los valores superiores son extraños y no representativos de la población a la cual el análisis apunta.
<br><br><br><br>
- Como resultado de procesar las variables se obtienen 18 variables, a continuación se plotean las correlaciones (Coeficiente de correlación de Spearman) de las variables discretas.<img align="center" width="90%" src="https://i.imgur.com/MIOzG6l.png">

  Es interesante que se observe una ligera correlación entre los subgrupos que fueron creados para las categorías de educación y trabajo. No se observa ninguna dependencia fuerte entre las variables.

  Para las variables nominales se realiza la prueba chi cuadrado de independencia de las variables respecto de la clase. Se observa que la variable Éxito previo está fuertemente relacionada al éxito o no el la campaña actual.

### Balanceo de clases

Al ser un problema de clases sesgadas es muy importante balancear la proporción de las clases. Para lograr esto se utilizan 3 métodos diferentes.

- Repetir ( x3 cada etiqueta positiva) y eliminar negativas de manera aleatoria
- [SMOTE](https://www.jair.org/media/953/live-953-2037-jair.pdf) - Una método para sobremuestrear la minoría
- Combinación de SMOTE y [ENN](https://www.researchgate.net/publication/3115579_Asymptotic_Properties_of_Nearest_Neighbor_Rules_Using_Edited_Data)  - el último un método para submuestrear.

También se prueba sin ningun sobre/sub muestreo para contrastar.

### Predicciones

Para realizar predicciones se utilizan 4 alternativas:

- Regresión logística
- Random forest
- Análisis discriminante lineal
- Una red neuronal

De los datos originales se extraen 5000 ejemplos para hacer de test set y evaluar la performance de los algoritmos. Una vez extraída la data de prueba se procesa la data para entrenar los algoritmos con cada método de sub/sobre muestreo y se entrena y evalúa cada algoritmo en la data de prueba (no adulterada por los métodos de sub/sobre muestreo).

## Resultados

- En el siguiente gráfico se muestra la exactitud o accuracy de los distintos clasificadores entrenados con cada método de sub/sobre muestreo y evaluados en la data de prueba extraída anteriormente.<img align="center" width="90%" src="https://i.imgur.com/t9dqYrB.png">

- Sin embargo la exactitud o accuracy no es un buen indicador de performance en este caso ya que la data es fuertemente sesgada, por lo cual un clasificador que simplemente predice siempre 'no' o 0 obiene alta exactitud. Es por eso que se procede a observar el recall, que proporción de ejemplos de clase positiva el algoritmo clasificó como positivos, para cada clasificador.<img align="center" width="90%" src="https://i.imgur.com/SPVKWa5.png">

- Se puede observar como cuando no se balancean las clases (véase 'false') los clasificadores tienen bajo recall, siendo el caso extremo la red neuronal que no levanta siquiera una predicción positiva.

  Sin embargo el indicador mas relevante para los objetivos de este análisis es la precisión, es decir la proporción de ejemplos que eran de clase positiva dentro de los predichos como positivos por el clasificador. Es el mas relevante ya que si se van a utilizar los clasificadores como herramienta para mejorar la campaña de marketing el objetivo principal es aumentar la probabilidad de dirigirse a un cliente que vaya a adquirir el producto, es decir hacer el targeting mas eficiente. Se muestra la precisión a continuación.

<img align="center" width="90%" src="https://i.imgur.com/qScKZFS.png">

-  La precisión mas alta es del 64.2 % (para regresión logística con SMOTENN). Esto es una mejora significativa, ya que a priori la probabilidad de dar con un cliente comprador era del 11.7%, contactando solamente aquellos clientes que el algoritmo clasifica como positivos la probabilidad pasa a el mencionado 64.2 %. Es decir con el algoritmo se mejora en un factor de 5.5 la posibilidad de encontrar un cliente comprador. Ahora bien el recall de esta opción era bajo, de tan solo 17 %, es decir se están pasando muchos potenciales clientes por alto.
<br><br>
- Una buena opción es el uso de random forest con 'repeat_drop' el cual da una precisión del 41 % y un recall del 46 %, Es decir aumenta significativamente las probabilidades de dar con un caso de éxito sin a su vez dejar de lado demasiados potenciales clientes

## Conclusión

Los algoritmos, asumiendo que los 5000 ejemplos tomados como población de prueba son representativos de la población, generaron resultados útiles. Utilizando su predicción a partir de la información del cliente para targetearlos aumentaría significativamente el porcentaje de éxito de la campaña. Que algoritmo y método de sub/sobre muestreo elegir específicamente depende de cuan dirigida se busque hacer la campaña.
