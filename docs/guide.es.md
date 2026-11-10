# Machine Learning y redes neuronales: guía para el equipo

Esta guía explica lo básico de machine learning y cómo funciona una red neuronal, sin relleno. Al final aterriza en nuestro proyecto: estimar el daño de edificios a partir de imágenes satelitales de antes y después de un desastre.

## 1. Introducción y cultura general

### 1.1 Qué es machine learning

Machine learning es la disciplina que se encarga de **encontrar una función** que dé respuesta a un problema. En lugar de escribir esa función a mano, usamos un **algoritmo de aprendizaje** que la busca por nosotros a partir de datos.

Hay que distinguir dos tipos de valores:

- **Parámetros**: los números de la función que el algoritmo ajusta solo (por ejemplo, los pesos de una red neuronal).
- **Hiperparámetros**: la configuración del algoritmo de aprendizaje, que elegimos nosotros antes de entrenar (por ejemplo, la tasa de aprendizaje, el tamaño del batch o el número de capas).

### 1.2 Áreas de machine learning

- **Aprendizaje supervisado**: los datos vienen con la respuesta correcta y el modelo aprende a reproducirla. Es nuestro caso: cada imagen trae la máscara de daños. Ejemplos: filtrar correos de spam a partir de correos ya etiquetados, estimar el precio de una casa con ventas pasadas, o detectar neumonía en radiografías con diagnósticos confirmados. La respuesta puede ser una categoría (clasificación) o un número (regresión).
- **Aprendizaje no supervisado**: los datos no traen respuesta y el modelo busca estructura por su cuenta, por ejemplo agrupar ejemplos parecidos. Ejemplos: segmentar clientes por su comportamiento de compra, agrupar imágenes satelitales por tipo de paisaje, detectar transacciones atípicas con tarjeta o reducir cientos de variables a unas pocas con PCA.
- **Aprendizaje por refuerzo**: un agente prueba acciones y aprende de las recompensas que recibe. Ejemplos: AlphaGo aprendiendo a jugar Go, un robot aprendiendo a caminar, o un sistema que ajusta semáforos para minimizar el tráfico.

### 1.3 Otros tipos de modelos

Existen muchos modelos además de las redes neuronales. Dos ejemplos clásicos:

- **KNN (k vecinos más cercanos)**: no entrena nada. Para clasificar un punto nuevo, busca los $k$ ejemplos más cercanos y vota por la clase mayoritaria.
- **Árbol de decisión**: hace preguntas de sí o no sobre las características de los datos. En cada paso elige la pregunta que mejor separa las clases y repite el proceso dentro de cada rama.

Nosotros nos enfocamos en redes neuronales porque es lo que usa el proyecto y porque es lo que domina el campo hoy. Quien necesite modelos clásicos en el futuro sabe por dónde entrar.

### 1.4 Tensores

Para nosotros, un tensor es simplemente un **array de arrays de varias dimensiones**. Una imagen en color, por ejemplo, es un tensor de tres dimensiones: alto, ancho y canales (rojo, verde y azul). Los frameworks de machine learning reciben y devuelven tensores.

### 1.5 Frameworks en Python

- **PyTorch**: el más usado hoy en investigación y muy popular en general. Competidor directo de TensorFlow.
- **TensorFlow**: framework de Google para deep learning.
- **Keras**: interfaz de alto nivel construida sobre TensorFlow, pensada para escribir redes con poco código.
- **scikit-learn** (`sklearn`): pensado para modelos clásicos como KNN, árboles de decisión o regresiones. También trae herramientas de validación y métricas. Las redes neuronales no son su fuerte.

Todos nos dan lo mismo en lo esencial: tensores, capas ya implementadas y el cálculo automático del gradiente.

### 1.6 Cómo medimos qué tan bueno es un modelo

Para encontrar la función usamos un conjunto de datos. Lo dividimos en partes:

- **Entrenamiento (train)**: con esto el algoritmo busca la función.
- **Validación**: sirve para tomar decisiones durante el desarrollo (elegir hiperparámetros, decidir cuándo parar, comparar modelos).
- **Prueba (test)**: datos que el modelo nunca vio ni usamos para decidir nada, para medir cómo se comporta de verdad al final.

#### Accuracy

Es el porcentaje de aciertos:

$$\text{accuracy} = \frac{\text{predicciones correctas}}{\text{total de predicciones}}$$

Parece suficiente, pero **engaña cuando las clases están desbalanceadas**. Si el 99% de los píxeles de una imagen son fondo, un modelo que siempre responde "fondo" tiene 99% de accuracy y no sirve para nada.

#### Falsos positivos y falsos negativos

- **Falso positivo**: el modelo dice que sí, pero era no.
- **Falso negativo**: el modelo dice que no, pero era sí.

Para verlo con el mecanismo de cada caso, conviene fijarse en qué significa "sí" en el problema:

- **Conducción autónoma.** El modelo recibe una imagen de la calle y debe decidir "hay un peatón" (sí) o "no hay un peatón" (no). Si hay un peatón delante y el modelo responde "no hay peatón", dijo que no cuando era sí: es un **falso negativo**. El auto no frena y atropella al peatón. Ese es el error peligroso, porque el costo de no detectarlo es un accidente.
- **Filtro de spam.** El modelo debe decidir "es spam" (sí) o "no es spam" (no). Si un correo legítimo de trabajo se marca como spam, el modelo dijo que sí cuando era no: es un **falso positivo**. El correo importante se pierde en la carpeta de spam, y ese es el error costoso; en cambio, dejar pasar un spam (falso negativo) solo molesta.

Cuál duele más depende del contexto y de qué significa "sí" en cada problema: en conducción autónoma el error caro es no detectar al peatón (falso negativo), en un filtro de spam es marcar un correo bueno (falso positivo).

#### Precisión, recall y F1

Accuracy falla con clases desbalanceadas porque cuenta todos los aciertos por igual, y el fondo se come la cuenta. Precisión, recall y F1 miran **solo la clase que nos interesa** (la "positiva"), que en nuestro caso es "este píxel tiene daño", y por eso no se dejan engañar por el fondo. Usamos las mismas nociones de arriba: $VP$ verdaderos positivos, $FP$ falsos positivos y $FN$ falsos negativos.

##### Precisión

De todas las veces que el modelo dijo "sí", ¿cuántas acertó?

$$\text{precisión} = \frac{VP}{VP + FP}$$

Es útil porque **castiga los falsos positivos**: cada vez que el modelo grita "daño" sin que lo haya, la precisión baja. En nuestro proyecto, una precisión baja significa que marcamos como dañados edificios que están bien, y los equipos de rescate pierden tiempo yendo a revisar falsas alarmas. Se usa cuando queremos que, cada vez que el sistema avise, sea confiable.

##### Recall

De todos los casos que de verdad eran "sí", ¿cuántos encontró el modelo?

$$\text{recall} = \frac{VP}{VP + FN}$$

Es útil porque **castiga los falsos negativos**: cada daño real que el modelo deja pasar sin marcar, el recall baja. En nuestro proyecto, un recall bajo significa que hay edificios destruidos que el mapa no muestra, y las personas que necesitan ayuda no son enviadas. Se usa cuando lo que más importa es no dejar pasar nada.

##### F1

Precisión y recall tiran en direcciones opuestas: si bajamos el umbral para no perder daños, sube el recall pero baja la precisión (marcamos de más), y si lo subimos, sube la precisión pero baja el recall (se nos escapan casos). F1 combina las dos en un solo número:

$$F_1 = \frac{2 \cdot \text{precisión} \cdot \text{recall}}{\text{precisión} + \text{recall}}$$

Es útil cuando hay que resumir el balance en una sola cifra y no se quiere elegir entre precisión y recall. Usa la **media armónica** (no la aritmética) a propósito: si una de las dos es baja, F1 se hunde aunque la otra sea perfecta. Así obliga a que ambas sean razonables, y no se puede "hacer trampa" mejorando solo una.

### 1.7 Overfitting

El **overfitting** (sobreajuste) ocurre cuando el modelo memoriza los datos de entrenamiento en lugar de aprender el patrón, y falla con datos nuevos. Se detecta comparando el error de entrenamiento con el de validación: si el primero sigue bajando y el segundo empieza a subir, hay overfitting.

Lo que más ayuda es **variedad en los datos**. Si todos los ejemplos se parecen, el modelo solo aprende eso. Variedad no es solo cantidad: mil fotos de la misma calle con la misma luz enseñan poco. Hacen falta también **ejemplos negativos** (cuándo no hay un carro, no solo cuándo lo hay), porque sin ellos el modelo puede responder "carro" a todo. Cuando faltan datos, se puede recurrir al **aumento de datos** (giros, recortes, cambios de brillo) y a la **regularización**, que penaliza modelos demasiado complicados.

### 1.8 Validación cruzada (K-fold)

La **validación cruzada K-Fold** es un método de remuestreo en aprendizaje automático para estimar qué tan bien generaliza un modelo, es usado para seleccionar un modelo o hiperparámetros para un modelo. El conjunto de datos se divide en **k** pliegues (*folds*) que no se solapan. El modelo se entrena **k** veces: en cada ocasión, un pliegue se reserva para validación y los otros **k-1** pliegues se usan para entrenamiento. Cada punto de datos se usa para validación exactamente una vez.

Los **k** puntajes de validación se resumen, normalmente con su **promedio**:

$$ \bar{x} = \frac{1}{k}\sum_{i=1}^{k} x_i $$

y su **varianza**

$$
s^2 = \frac{1}{k-1}\sum_{i=1}^{k}\left(x_i-\bar{x}\right)^2
$$

para estimar el rendimiento esperado y la estabilidad del modelo.

```python

def k_fold_cross_validation(full_dataset, k, model_builder):

    # Randomly partition the dataset into k non-overlapping folds.
    folds = split_dataset_into_k_non_overlapping_folds(full_dataset, k)

    validation_scores = empty_collection()

    for validation_fold in folds:
        # Build the training set from every fold except validation_fold.
        training_set = combine_all_folds_except(folds, validation_fold)

        # Train a fresh model using only the training set.
        model = build_and_train_model(model_builder, training_set)

        # Evaluate the trained model on the held-out validation fold.
        score = evaluate_model(model, validation_fold)

        # Record this fold's validation score.
        record_score(validation_scores, score)

    average_score = average_of(validation_scores)
    variance_score = variance_of(validation_scores)

    return average_score, variance_score
```

## 2. Redes neuronales

Una red neuronal es **una composición de funciones**. Nada más. No necesitamos analogías biológicas: lo que ocurre realmente es que componemos funciones, y la razón de componerlas es que, si no lo hiciéramos, habría decisiones que no podríamos tomar.

La mejor forma de entenderlo es seguir la historia de por qué cada pieza apareció.

### 2.1 Una neurona

Queremos que una computadora aprenda a multiplicar por 2. Empezamos con la función más simple posible, una sola neurona:

$$f(x) = w \cdot x + b$$

Aquí $w$ (el peso) y $b$ (el sesgo) son los parámetros. El algoritmo de aprendizaje debería encontrar $w = 2$ y $b = 0$. Para este problema, una neurona basta.

### 2.2 El límite de una sola neurona

Una neurona solo puede representar funciones lineales. Hay problemas tan simples como el **XOR** que ninguna función lineal puede resolver: la salida es 1 cuando las dos entradas son distintas y 0 cuando son iguales, y no existe una recta que separe esos casos.

Primer límite encontrado. La solución natural: usar más de una neurona y componerlas.

### 2.3 Componer capas, y por qué hace falta la función de activación

Si componemos dos funciones lineales, el resultado sigue siendo una función lineal:

$$w_2 (w_1 x + b_1) + b_2 = (w_2 w_1)\,x + (w_2 b_1 + b_2)$$

Es decir, apilar capas lineales no añade nada. Por eso entre capa y capa aplicamos una **función de activación** $\sigma$ no lineal, por ejemplo la ReLU:

$$\sigma(z) = \max(0, z)$$

Ahora sí, una red de dos capas tiene la forma:

$$f(x) = W_2 \, \sigma(W_1 x + b_1) + b_2$$

Con esto la red puede representar funciones que una sola neurona no puede, como el XOR. Y añadiendo más capas, funciones cada vez más complejas. Esa es la red neuronal clásica, la del perceptrón multicapa.

> **Aviso de transparencia:** esta forma clásica depende mucho del dominio y casi nadie la usa tal cual hoy. Se enseña porque con ella se entiende la idea. Para imágenes, por ejemplo, se usan otras arquitecturas, como veremos al final.

## 3. Cómo se entrena una red

Ya sabemos la forma del modelo. Falta saber cómo encontrar buenos parámetros.

### 3.1 La función de pérdida

Queremos que el modelo se equivoque lo menos posible. Para medirlo definimos una **función de pérdida** $L$: un número que es pequeño cuando el modelo acierta y grande cuando se equivoca.

La pérdida depende del dominio. Algunos ejemplos:

- **Predecir un número** (precio de una casa): el error cuadrático medio, $L = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$. Penaliza mucho los errores grandes.
- **Clasificar en categorías** (¿qué nivel de daño tiene este edificio?): la entropía cruzada, que castiga fuerte al modelo cuando está seguro de una respuesta equivocada.
- **Clases muy desbalanceadas** (casi todo es fondo): se combina con pérdidas como Dice, o se ponen pesos por clase, para que acertar la clase rara importe.

Entrenar es entonces **minimizar la pérdida**, que es un problema de optimización.

### 3.2 Descenso por gradiente

El gradiente de la pérdida apunta hacia donde la pérdida **sube** más rápido. Así que para bajarla nos movemos en sentido contrario:

$$\theta \leftarrow \theta - \eta \, \nabla L(\theta)$$

Donde $\theta$ son todos los parámetros y $\eta$ es la **tasa de aprendizaje** (learning rate), un hiperparámetro que controla el tamaño del paso.

El gradiente cero es una condición necesaria de mínimo, no suficiente: también lo cumplen los máximos y los **puntos silla**. En redes neuronales la pérdida no es convexa, así que en general solo podemos esperar llegar a un punto donde el gradiente sea cero.

### 3.3 Descenso por gradiente estocástico y batches

Calcular el gradiente sobre todo el dataset en cada paso es caro. En la práctica partimos el dataset en pedazos, los **batches**, y en cada paso usamos uno solo. Esto crea otro hiperparámetro: el **batch size**.

¿Por qué funciona? El gradiente de la pérdida completa es el promedio de los gradientes de cada ejemplo. Un batch aleatorio da una pista **ruidosa** de esa dirección, pero mucho más barata de calcular, así que podemos dar muchos más pasos en el mismo tiempo. Un batch individual puede no ser representativo (si una clase es rara, casi nunca aparecerá en un batch pequeño), pero **muchos pasos con batches distintos, sumados, se parecen a pasos exactos**.

Dos matices importantes:

- **La tasa de aprendizaje.** Con gradientes ruidosos, una tasa constante hace que el modelo rebote alrededor del mínimo sin asentarse. La teoría pide que $\eta$ vaya disminuyendo. En la práctica se usa constante o con un *scheduler* que la va bajando.
- **El batch size no tiene un valor mágico.** Un batch pequeño no rompe la convergencia, solo hace el camino más ruidoso. Se elige probando, equilibrando ruido, velocidad y memoria.

### 3.4 Cuándo parar

Con batches el gradiente siempre es ruidoso, así que mirar si su norma es cercana a cero no sirve bien. En la práctica se para cuando la pérdida **deja de mejorar en el conjunto de validación**.

### 3.5 Backpropagation

Para dar cada paso necesitamos el gradiente de la pérdida respecto a todos los parámetros. Como la red es una composición de funciones, la derivada se expande con la regla de la cadena en una expresión enorme donde muchas subpartes se repiten.

Esas dependencias forman un **grafo dirigido acíclico (DAG)**. Recorriéndolo en orden topológico y guardando los resultados intermedios (programación dinámica), cada subparte se calcula una sola vez. Eso es **backpropagation**. No vamos a deducir las fórmulas: todos los frameworks lo hacen por nosotros, y la idea alcanza para entender qué ocurre.

## 4. Fine tuning

Normalmente una red empieza con pesos aleatorios. En el **fine tuning** empezamos con los pesos de un modelo ya entrenado en otra tarea y nos movemos desde ahí. El modelo ya aprendió representaciones útiles, así que el entrenamiento suele ser más rápido y necesita menos datos. Típicamente se usa una tasa de aprendizaje más pequeña para no destruir lo que ya sabía.

## 5. Nuestro proyecto

### 5.1 El problema

La entrada son dos imágenes satelitales de la misma zona: una **antes** y otra **después** de un desastre. La salida es, para cada píxel, el nivel de daño del edificio que hay ahí. Usamos el dataset **xView2 (xBD)**.

Los datos originales traen polígonos que representan edificios con su nivel de daño. Los convertimos a una **máscara**: una matriz del tamaño de la imagen donde cada posición vale una categoría de 0 a 4 (0 es fondo, sin edificio, y 1 a 4 son los niveles de daño).

### 5.2 La salida de la red

La salida de nuestra red es un tensor de cuatro dimensiones:

1. Índice de la imagen dentro del batch.
2. Categoría (5 en total).
3. Fila del píxel.
4. Columna del píxel.

Cada valor es la probabilidad, entre 0 y 1, de que ese píxel pertenezca a esa categoría. Para la respuesta final nos quedamos con la categoría de mayor probabilidad en cada píxel, y eso nos devuelve la matriz de daños.

Para entrenar, la máscara de etiquetas se compara con las probabilidades usando entropía cruzada, que internamente hace la codificación *one-hot* (un 1 en la categoría correcta y 0 en las demás). No hace falta convertirla nosotros.

### 5.3 Red siamesa con estructura encoder-decoder

Usamos una **red siamesa** con estructura tipo **U-Net**. La idea clave: las dos imágenes entran como entradas separadas y pasan por el mismo encoder (con los mismos pesos), en vez de concatenarse al principio.

- **Encoder**: una composición de capas que reduce el tamaño espacial de la imagen y aumenta la cantidad de canales por posición. En vez de 3 valores por píxel (RGB), podemos tener 64. Intuitivamente, agrupa píxeles cercanos y extrae características de ese grupo.
- **Combinación**: las características de la imagen de antes y de la de después se combinan con una función. Una resta es la intuición más fácil: si una característica como la altura de un edificio pasa de un valor alto a cero, el edificio se cayó. En la práctica la combinación es más flexible que una resta.
- **Decoder**: otra composición de capas que vuelve a expandir hasta el tamaño original y produce la clasificación de daño por píxel. En un U-Net el decoder también recibe atajos del encoder, para recuperar detalle fino.

El entrenamiento no cambia nada de lo visto antes. Encoder, decoder y la función que los une son todos composición de funciones, así que lo único que hacemos es calcular el gradiente de la pérdida y aplicar descenso por gradiente con batches.

En el dataset xView2, un enfoque habitual tiene dos etapas: primero localizar los edificios, y luego clasificar su nivel de daño. Como la mayoría de los píxeles son fondo, el desbalance de clases es un problema real, y por eso se suele combinar la entropía cruzada con otras pérdidas o con pesos por clase.

### 5.4 Plan de implementación

- Versión preliminar local, con imágenes reducidas a 256 por 256 píxeles, para probar todo el pipeline.
- El código debe leerse como esta explicación: sin trucos específicos del lenguaje, de modo que se vean directamente el encoder, el decoder, la pérdida y el descenso por gradiente.
- Cada etapa es un **script independiente** que recibe sus entradas por la entrada estándar, con flags para elegir cuántos archivos usar. Los resultados se guardan en una carpeta del repositorio y, si algo ya se hizo, no se repite.
- Más adelante, si todo funciona, se pueden unir las etapas importando sus funciones desde un único script.
- Usaremos fine tuning a partir de pesos ya entrenados.

## 6. Resumen de ideas clave

- Machine learning es buscar una función con un algoritmo, no escribirla a mano.
- Ningún algoritmo es mejor que otro sin suponer algo del problema (no free lunch): hay que elegir el modelo cuyas suposiciones encajan.
- Se evalúa con datos que el modelo no vio, y la accuracy sola puede engañar.
- Una red neuronal es una composición de funciones; la activación no lineal evita que se colapse en una sola función lineal.
- Entrenar es minimizar una pérdida que depende del dominio, con descenso por gradiente sobre batches.
- Backpropagation es calcular ese gradiente reutilizando subresultados.
- Nuestro modelo es una red siamesa tipo U-Net: un encoder que extrae características de cada imagen, una combinación de antes y después, y un decoder que produce el daño por píxel.