# Machine Learning y redes neuronales

Esta guía explica lo básico de machine learning y cómo funciona una red neuronal.

---

## 1. Conceptos

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

Nosotros nos enfocamos en redes neuronales porque es lo que usa el proyecto y porque es lo que domina el campo hoy.

### 1.4 Tensores

Para nosotros, un tensor es simplemente un **array de arrays de varias dimensiones**. Una imagen en color, por ejemplo, es un tensor de tres dimensiones: alto, ancho y canales (rojo, verde y azul). Los frameworks de machine learning reciben y devuelven tensores.

### 1.5 Frameworks en Python

- **PyTorch**: el más usado hoy en investigación y muy popular en general. Competidor directo de TensorFlow.
- **TensorFlow**: framework de Google para deep learning.
- **Keras**: interfaz de alto nivel construida sobre TensorFlow, pensada para escribir redes con poco código.
- **scikit-learn** (`sklearn`): pensado para modelos clásicos como KNN, árboles de decisión o regresiones. También trae herramientas de validación y métricas. Las redes neuronales no son su fuerte.

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

---

## 2. Redes neuronales

Una red neuronal es en el sentido matemático **una composición de funciones**.

$$ F(x)=f_L\circ f_{L-1}\circ\cdots\circ f_1(x)$$

se suele llamar capas a estas funciones,cada capa suele tener esta forma

$$\qquad f_\ell(x)=g_\ell(W_\ell x+b_\ell)$$

>**Nota**:digo 'suele' porque hay varios tipos de redes neuronales,este ejemplo específico es de una red **feedforward**.

La razón de componerlas es que, si no lo hiciéramos, habría decisiones que no podríamos tomar.La mejor forma de entenderlo es seguir la historia de por qué cada pieza apareció.

### 2.1 Una neurona

Queremos que una computadora aprenda a multiplicar por 2. Empezamos con la función más simple posible, una sola neurona:

$$f(x) = w \cdot x + b$$

Aquí $w$ (el peso) y $b$ (el sesgo) son los parámetros. El algoritmo de aprendizaje debería encontrar $w = 2$ y $b = 0$. Para este problema, una neurona basta.

### 2.2 El límite de una sola neurona

Empecemos por una función muy simple que **no** es afín: el valor absoluto.

$$
f(x)=|x|.
$$

> Nota: Las funciones afines son las que tienen la forma $y= Wx+B$, la $B$ es lo que la diferencia de una lineal

Una sola neurona afín tiene la forma

$$
\hat f(x)=wx+b.
$$

Supongamos que existen $w$ y $b$ tales que $\hat f(x)=|x|$ para todo $x$. Evaluando en $x=1$ y en $x=-1$:

$$
w+b=1,\qquad -w+b=1.
$$

Sumando ambas ecuaciones: $2b=2$, luego $b=1$. Restándolas: $2w=0$, luego $w=0$. Pero entonces

$$
\hat f(0)=w\cdot 0+b=1\ne |0|=0.
$$

Contradicción. Por tanto, **no existen $w$ y $b$** que representen $|x|$ con una sola neurona afín.

### 2.3 Componer capas, y por qué hace falta la función de activación

Si componemos dos funciones afines , el resultado sigue siendo una función afín:

$$w_2 (w_1 x + b_1) + b_2 = (w_2 w_1)\,x + (w_2 b_1 + b_2)$$

Es decir, apilar capas lineales no añade nada. Por eso entre capa y capa aplicamos una **función de activación** $\sigma$ no lineal, por ejemplo la ReLU:

$$\sigma(z) = \max(0, z)$$

Ahora sí, una red de dos capas tiene la forma:

$$f(x) = W_2 \, \sigma(W_1 x + b_1) + b_2$$

Con esto la red puede representar funciones que una sola neurona no puede, como el XOR. Y añadiendo más capas, funciones cada vez más complejas. Esa es la red neuronal clásica, la del perceptrón multicapa.

> **Nota:** esta forma clásica depende mucho del dominio y casi nadie la usa tal cual hoy. Se enseña porque con ella se entiende la idea. Para imágenes, por ejemplo, se usan otras arquitecturas, como veremos al final.

---

## 3. Cómo se entrena una red

Ya sabemos la forma del modelo: una función $f$ que, dado un ejemplo $x$, produce una predicción

$$
\hat{y}=f(x;\theta),
$$

donde $\theta$ son los pesos o parámetros del modelo.

Falta saber cómo encontrar buenos valores de $\theta$.

### 3.1 Los parámetros como un vector

Aunque una red tenga muchas matrices de pesos y muchos vectores de sesgo (*bias*), conviene pensar en **todos** sus parámetros como un único vector.

Si la red tiene $L$ capas con matrices

$$
W_1,\dots,W_L
$$

y sesgos

$$
b_1,\dots,b_L,
$$

definimos

$$
\begin{aligned}
\theta
=
\big(
&\mathrm{vec}(W_1),\, b_1,\\
&\mathrm{vec}(W_2),\, b_2,\\
&\dots,\\
&\mathrm{vec}(W_L),\, b_L
\big)
\in \mathbb{R}^{p},
\end{aligned}
$$

donde $\mathrm{vec}(W)$ significa **apilar las columnas** de $W$ una debajo de otra.

Es decir, $\theta$ es simplemente la concatenación de:

- las columnas de cada matriz de pesos $W_\ell$, y
- cada vector de sesgo $b_\ell$.

Todo eso, junto, forma un vector de dimensión $p$, el número total de parámetros de la red.

Así, aunque $f$ sea una función complicada (convoluciones, activaciones, *pooling*, etc.), el modelo queda completamente determinado por un vector

$$
\theta \in \mathbb{R}^p.
$$

Entrenar es, entonces, encontrar ese vector.

### 3.2 La función de pérdida

Para poder encontrar este vector necesitamos una forma de medir “qué tan mal” lo está haciendo el modelo.

Esa medida es la **función de pérdida**.

#### Definición general

Sea

$$
D=\{(x_i,y_i)\}_{i=1}^n
$$

el dataset de entrenamiento.

La función de pérdida es una función de los pesos del modelo:

$$
L(\theta;D)
=
\frac{1}{n}
\sum_{i=1}^n
\ell\big(y_i, f(x_i;\theta)\big),
$$

donde $\ell$ es la pérdida de un solo ejemplo.

Aquí conviene distinguir dos piezas:

- $D=\{(x_i,y_i)\}$ es el **dataset**.
  - Durante el entrenamiento lo tratamos como **constante**.

- $\theta \in \mathbb{R}^p$ es el **vector de pesos del modelo**.
  - Son las **variables** que queremos ajustar.
  - La pérdida cambia cuando cambia $\theta$.

- $L(\theta;D)$ es entonces una función que, para un dataset fijo, nos dice cuánto se equivoca el modelo con unos pesos $\theta$.

Entrenar es entonces **minimizar la pérdida con respecto a los pesos**:

$$
\theta^\star
=
\arg\min_{\theta \in \mathbb{R}^p}
L(\theta;D).
$$

Es decir, buscamos el vector $\theta$ que hace que el modelo se equivoque lo menos posible sobre el dataset de entrenamiento.

La forma concreta de $\ell$ depende del tipo de problema.

#### 3.2.1 Predecir un número

Se usa el error cuadrático medio:

$$
\ell(y,\hat{y})=(y-\hat{y})^2.
$$

Entonces,

$$
L(\theta;D)
=
\frac{1}{n}
\sum_{i=1}^n
\big(y_i-\hat{y}_i\big)^2,
$$

donde

$$
\hat{y}_i=f(x_i;\theta).
$$

Penaliza mucho los errores grandes.

#### 3.2.2 Segmentación donde casi todo es fondo

Aquí destaca la **pérdida Dice**.

Si $A$ es el conjunto de píxeles predichos como  especiales y $B$ el conjunto real de estos píxeles, se define

$$
\mathrm{Dice}(A,B)
=
\frac{2\,|A\cap B|}{|A|+|B|},
$$

y

$$
\ell_{\text{Dice}}
=
1-\mathrm{Dice}(A,B).
$$

La función mide directamente el solapamiento entre predicción y verdad, normalizado por el tamaño de ambos conjuntos.

Si el conjunto ocupa poquísimos píxeles, el fondo (que es enorme) no domina la pérdida:

- el denominador $|A|+|B|$ incluye solo la región de interés,
- no todos los píxeles.

Por eso se usa mucho en segmentación médica, de edificios dañados, etc.

### 3.3 Descenso por gradiente

El gradiente de la pérdida apunta hacia donde la pérdida **sube** más rápido. Así que para bajarla nos movemos en sentido contrario:

$$\theta \leftarrow \theta - \eta \, \nabla L(\theta)$$

Donde $\theta$ son todos los parámetros y $\eta$ es la **tasa de aprendizaje** (learning rate), un hiperparámetro que controla el tamaño del paso.

El gradiente cero es una condición necesaria de mínimo, no suficiente: también lo cumplen los máximos y los **puntos silla**. En redes neuronales la pérdida no es convexa, así que en general solo podemos esperar llegar a un punto donde el gradiente sea cero.

### 3.4 Descenso por gradiente estocástico por mini-lotes (batches)

Este algoritmo es el más usado en la práctica, realiza actualizaciones de parámetros usando el gradiente promedio calculado sobre un subconjunto pequeño de los datos de entrenamiento, muestreado aleatoriamente, llamado **mini-lote** (a menudo denotado como $\mathcal{B}_t$ de tamaño $b$). El tamaño de este mini-lote normalmente varía de 2 a unos pocos cientos, aunque puede ser mayor, a menudo limitado por la memoria de la GPU.

La regla de actualización en la iteración $t$ es:

$$
\theta_{t+1} \leftarrow \theta_t - \eta_t \cdot \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \nabla \ell_i(\theta_t)
$$

donde $\eta_t$ es la tasa de aprendizaje (o tamaño de paso) en la iteración $t$, y $\nabla \ell_i(\theta_t)$ es el gradiente de la pérdida para un único punto de datos $i$ evaluado en $\theta_t$.

La idea es que el gradiente del mini-lote sirve como una **estimación insesgada** del gradiente de lote completo, es decir, $\mathbb{E}[\nabla L_{\mathcal{B}_t}(\theta_t)] = \nabla L(\theta_t)$. Esto significa que, *en promedio*, la actualización apunta en la dirección del gradiente verdadero, lo que permite la convergencia mientras es mucho más eficiente por iteración que el GD de lote completo.

> *Nota*: en el entrenamiento estándar basado en épocas (una época  significa recorrer una vez todo el conjunto de entrenamiento), los mini-lotes **no** se reutilizan como los mismos grupos fijos. En cada época, el conjunto de entrenamiento se vuelve a mezclar y se reparte en nuevos mini-lotes $\mathcal{B}_1, \dots, \mathcal{B}_K$, donde $K \approx \lceil N/b \rceil$ para un conjunto de datos de tamaño $N$ y tamaño de lote $b$. Así, los ejemplos se reutilizan en cada época, pero su agrupación en lotes cambia. Si **no** se mezclan, entonces efectivamente se rota por los mismos lotes fijos en cada época; los gradientes de lotes individuales entonces no son estimaciones insesgadas del gradiente completo, aunque el promedio durante una época completa es igual al gradiente completo. Algunas implementaciones, en cambio, muestrean un mini-lote aleatorio nuevo **con reemplazo** en cada paso, por lo que no hay ninguna lista fija de lotes. En la notación anterior, $\mathcal{B}_t$ es, por lo tanto, generalmente un subconjunto aleatorio nuevo en cada iteración, no un lote fijo que simplemente cicla.

### 3.5 Cuándo parar

Con mini-lotes, el gradiente que usamos en cada iteración es ruidoso:

$$
\nabla L_{\mathcal{B}_t}(\theta_t)
=
\frac{1}{|\mathcal{B}_t|}
\sum_{i\in \mathcal{B}_t}
\nabla \ell_i(\theta_t),
$$

y aunque estemos cerca de un mínimo, su norma puede seguir siendo grande. Por eso **no** conviene parar mirando si $\|\nabla L_{\mathcal{B}_t}(\theta_t)\|$ es cercana a cero.

En la práctica se usa un **conjunto de validación** distinto del de entrenamiento. Sea

$$
D_{\text{val}}=\{(x_i^{\text{val}},y_i^{\text{val}})\}_{i=1}^{n_{\text{val}}}
$$

un conjunto disjunto de $D$. Definimos su pérdida como

$$
L_{\text{val}}(\theta;D_{\text{val}})
=
\frac{1}{n_{\text{val}}}
\sum_{i=1}^{n_{\text{val}}}
\ell\big(y_i^{\text{val}}, f(x_i^{\text{val}};\theta)\big).
$$

Esta pérdida **no se usa para actualizar** $\theta$; solo se usa para decidir cuándo parar y qué pesos conservar.

Al final de cada época $e$, calculamos $L_{\text{val}}(\theta_e)$, donde $\theta_e$ son los pesos después de recorrer una vez el conjunto de entrenamiento $D$. La idea de “deja de mejorar” se formaliza así:

- Se eligen una **paciencia** $P\in\mathbb{N}$ y una **mejora mínima** $\delta\ge 0$.
- Se guarda el mejor valor de validación visto hasta ahora, $L_{\text{val}}^{\text{best}}$, y los pesos correspondientes $\theta^\star$.
- Después de cada época:
  - si $L_{\text{val}}(\theta_e) < L_{\text{val}}^{\text{best}}-\delta$, hay mejora significativa: actualizamos $L_{\text{val}}^{\text{best}}$ y $\theta^\star$, y reiniciamos el contador;
  - si no, aumentamos el contador de épocas sin mejora.
- Se para cuando el contador llega a $P$, o cuando se alcanza un número máximo de épocas.

Es decir, **“deja de mejorar” significa que durante $P$ épocas consecutivas la pérdida de validación no ha bajado al menos $\delta$ respecto del mejor valor anterior**. Al final se devuelven los mejores pesos $\theta^\star$, no necesariamente los últimos.

Si $\delta=0$, cualquier mejora estricta cuenta. La paciencia $P$ permite tolerar el ruido de los mini-lotes y no parar por una fluctuación pequeña. Además, conviene fijar un máximo de épocas para evitar entrenamientos infinitos.

### 3.6 Backpropagation

Para dar cada paso necesitamos el gradiente de la pérdida respecto a todos los parámetros, $\nabla L(\theta;D)$,los frameworks se encargan de esto por nosotros pero es bueno saber a alto nivel la idea que usan. La red es una composición de operaciones: entradas, pesos, sesgos, preactivaciones, activaciones, predicción y pérdida. Cada una de esas cantidades se puede pensar como un **nodo** de un grafo, y ponemos una arista $a\to b$ si $b$ se calcula usando $a$.

Por ejemplo, en una red simple:

$$
x,W_1 \longrightarrow z_1=W_1x+b_1
\longrightarrow h_1=\sigma(z_1)
\longrightarrow z_2=W_2h_1+b_2
\longrightarrow \hat{y}
\longrightarrow \ell(y,\hat{y}).
$$

> **Nota**: ayuda ver este ejemplo de derecha a izquierda al principio, y después de izquierda a derecha

Todas las flechas van desde algo que se usa para calcular otra cosa hacia su resultado. Nunca se vuelve a un nodo anterior. Por eso el grafo es **dirigido acíclico**: un **DAG**. Si hubiera un ciclo, una variable dependería de sí misma y no podríamos ordenar el cálculo hacia adelante ni hacia atrás de forma limpia.

La pérdida $L$ es una composición de funciones a lo largo de los caminos que van desde $\theta$ hasta $\ell$. Para saber cuánto afecta un nodo a la pérdida hay que multiplicar derivadas locales a lo largo de esos caminos: esa es la regla de la cadena. Si lo hiciéramos ingenuamente, muchas subexpresiones se repetirían.

**Backpropagation** recorre el DAG en orden inverso: empieza en la pérdida y va propagando hacia atrás “cuánto cambia $L$ si cambia cada nodo”. En cada nodo reutiliza los valores ya calculados de sus consumidores y las derivadas locales, y guarda los resultados intermedios del forward pass. Así, cada subparte se calcula una sola vez: eso es programación dinámica sobre el DAG.

---

## 4. Fine tuning

Normalmente una red empieza con pesos aleatorios. En el **full fine tuning** empezamos con los pesos de un modelo ya entrenado en otra tarea y nos movemos desde ahí. El modelo ya aprendió representaciones útiles, así que el entrenamiento suele ser más rápido y necesita menos datos. Típicamente se usa una tasa de aprendizaje más pequeña para no destruir lo que ya sabía.