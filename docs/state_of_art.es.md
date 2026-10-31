---

### Reporte de Estado del Arte: Evaluación Automatizada de Daños en Edificaciones mediante Imágenes Satelitales Pre y Post-Desastre

**1. Introducción y Relevancia del Problema**

La evaluación rápida y precisa de daños en edificaciones tras un desastre natural es una tarea crítica para la respuesta de emergencia y la recuperación. Los métodos tradicionales de inspección en campo son lentos y costosos, mientras que la interpretación manual de imágenes satelitales es laboriosa y propensa a errores. En este contexto, el aprendizaje profundo aplicado a la teledetección ha emergido como una solución escalable y automatizada. La literatura reciente confirma que el uso de la inteligencia artificial (IA) ha avanzado significativamente en la evaluación de daños por desastres, permitiendo un análisis más rápido y preciso que las inspecciones tradicionales.

El proyecto propuesto se alinea con la frontera de esta área al abordar la **segmentación y clasificación de daños en edificaciones** a partir de pares de imágenes satelitales (antes/después), una formulación que la comunidad científica considera el estándar para este problema.

**2. Datasets de Referencia: El Estandar de la Industria**

El estado del arte en este dominio está intrínsecamente ligado a la disponibilidad de datasets públicos a gran escala y anotados con rigor.

*   **xBD (xView2):** Es el *benchmark* predominante y el estándar de facto para la evaluación de daños en edificaciones. El dataset xBD, liberado como parte del desafío xView2, proporciona pares de imágenes satelitales de muy alta resolución (VHR) de múltiples eventos de desastre (huracanes, terremotos, inundaciones, etc.) junto con anotaciones a nivel de edificio para cuatro niveles de daño: sin daño, menor, mayor y destruido. La existencia de este *benchmark* ha acelerado la investigación en detección automatizada de daños.

*   **Datasets Complementarios para Inundaciones:** Para el alcance opcional de generalización a otros tipos de desastres, la literatura ofrece alternativas. **Sen1Floods11** es un dataset geo-referenciado diseñado para el entrenamiento y prueba de algoritmos de detección de inundaciones utilizando datos de radar de apertura sintética (SAR) de Sentinel-1. Por otro lado, **SpaceNet 8** se centra en la detección de carreteras y edificios inundados, proporcionando imágenes antes/después de eventos de inundación.

**3. Arquitecturas y Enfoques Metodológicos Dominantes**

La literatura reciente muestra una evolución desde modelos puramente convolucionales hacia arquitecturas más sofisticadas que capturan mejor la información temporal y espacial.

*   **Redes Siamesas (Siamese Networks):** Son el enfoque fundamental para procesar pares de imágenes. Una red siamesa utiliza dos ramas idénticas (con pesos compartidos) para extraer características de la imagen pre y post-desastre por separado. Luego, estas características se comparan o fusionan para detectar cambios. La combinación de una arquitectura **U-Net** (eficaz para segmentación) con un diseño siamés es una estrategia ampliamente validada. Por ejemplo, un estudio reciente propone una red siamesa basada en UNet que logró una precisión general del 95.5% en la evaluación de daños por el terremoto de Marruecos de 2023, utilizando el dataset xBD.

*   **Arquitecturas con Transformers:** Más recientemente, los *Transformers* se han integrado para superar las limitaciones de las CNN en la captura de dependencias de largo alcance. Un trabajo notable propone una arquitectura **UNet jerárquica con transformadores** que utiliza la diferencia entre las características codificadas por transformadores de las imágenes pre y post-desastre. Este método, que logra resultados de vanguardia en el dataset xBD, demuestra la efectividad de los transformadores para modelar cambios complejos. Otras propuestas, como **DDNet**, emplean un módulo de atención conjunta para extraer mejor las características temporales, alcanzando un F1-score total del 79.56% en xBD.

*   **Enfoques Híbridos (Espacial + Frecuencia):** La investigación también explora representaciones complementarias. Un estudio comparativo reciente evaluó modelos que operan en el dominio espacial (como las CNN), en el de frecuencia y en un dominio dual. Los resultados muestran que los modelos de dominio dual, que combinan ambas representaciones, ofrecen mejoras medibles sobre los enfoques de un solo dominio, aunque todos los modelos luchan por detectar daños sutiles debido al desbalance de clases.

**4. Desafíos Centrales del Estado del Arte: Desbalance de Clases y Evaluación**

La literatura identifica el **severo desbalance de clases** como el principal obstáculo técnico. La inmensa mayoría de los edificios en una escena no presentan daño, lo que sesga a los modelos hacia la clase mayoritaria. Un modelo ingenuo puede alcanzar una alta precisión general (accuracy) simplemente prediciendo "sin daño" para casi todos los edificios, lo cual es inútil en la práctica. Por ello, la comunidad ha desarrollado varias estrategias:

*   **Funciones de Pérdida Especializadas:** El uso de **funciones de pérdida ponderadas** es fundamental. La combinación de **Focal Loss y Dice Loss** ha demostrado ser particularmente efectiva para abordar el desbalance y mejorar la precisión de los límites de los edificios. Asimismo, se han propuesto pérdidas personalizadas como la **Ordinal Class Distance Penalty Loss (OCDPL)** , que penaliza los errores de clasificación según la distancia ordinal entre las clases de daño, mejorando los F1-scores de las clases minoritarias.

*   **Métricas de Evaluación Adecuadas:** Para reflejar el rendimiento real en las clases minoritarias, el **F1-score por clase (o macro-F1)** es la métrica estándar, en lugar de la precisión global (accuracy). Los estudios reportan tanto el F1-score de localización de edificios como el de clasificación de daño, siendo este último el más desafiante.

**5. Limitaciones y Direcciones Futuras: Generalización y Ruido en Etiquetas**

El estado del arte reconoce dos limitaciones importantes que definen las líneas de investigación futuras y que son relevantes para el realismo de tu proyecto:

*   **Generalización a Eventos No Vistos:** Muchos modelos con arquitecturas complejas logran un rendimiento excelente en la división de datos original de xBD, pero **ven degradado su rendimiento cuando se evalúan en eventos de desastre completamente nuevos** (e.g., un huracán no visto durante el entrenamiento). Esto ha motivado enfoques "centrados en los datos" (*data-centric*). Un ejemplo es **DisasterAdaptiveNet**, que incorpora información sobre el tipo de desastre como una variable de entrada, mejorando significativamente su capacidad de generalización a eventos no vistos en comparación con métodos puramente centrados en el modelo.

*   **Ruido Inherente en las Etiquetas:** Las etiquetas de daño son asignadas por analistas humanos y conllevan un grado de subjetividad, especialmente al distinguir entre "daño menor" y "daño mayor". Este ruido en las etiquetas establece un techo práctico de precisión para los modelos. Un estudio sobre métodos basados en *Transformers* para clasificación de parches confirma que tanto el ruido en las etiquetas como el severo desbalance de clases siguen siendo desafíos abiertos.

**Conclusión: Posicionamiento del Proyecto**

El proyecto propuesto no es un ejercicio aislado, sino una implementación de complejidad media que se sitúa directamente sobre las prácticas y desafíos actuales del estado del arte. La elección del dataset xBD, el uso de una arquitectura de segmentación (U-Net o similar) con un enfoque siamés para procesar el par de imágenes, y la evaluación mediante F1-score por clase, son decisiones metodológicas que la literatura respalda como el camino correcto.

El proyecto se alinea con la frontera del conocimiento al reconocer explícitamente los riesgos (desbalance, cómputo) y al proponer mitigaciones estándar en la industria (pesos por clase, funciones de pérdida especializadas). Además, la idea de un **"mapa de afectación por celdas"** como entregable final, aunque es un desarrollo de posprocesamiento, se conecta con la necesidad práctica de traducir las predicciones del modelo en información accionable para la respuesta al desastre. La etapa de posprocesamiento es, por tanto, un área donde el equipo puede aportar valor y justificar sus propias decisiones de diseño.

En resumen, el estado del arte valida la propuesta: los datasets, arquitecturas y métricas que planeas utilizar son los que la comunidad científica emplea para avanzar en este campo. El éxito del proyecto dependerá de la ejecución rigurosa para manejar los desafíos conocidos, especialmente el desbalance de clases y la limitación de recursos computacionales, tal como lo anticipa tu planificación.
