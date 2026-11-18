# Plan de implementación

Está diseñado para que 5 personas trabajen en paralelo sin bloquearse, con pruebas independientes (Mocks), estructuras flexibles y reglas claras de integración.

---

### 📜 1. El Contrato de Comunicación (Interfaces)
Para que todos trabajen en paralelo, definan estas "formas" de datos usando **\(N\)** como variable dinámica (potencia de 2, ej. 64, 128, 256):
*   **Entrada (Imágenes):** `[B, 2, 3, N, N]` (B=lote, 2=antes/después, 3=RGB).
*   **Entrada (Máscaras):** `[B, N, N]` (Valores: `0`=fondo, `1-4`=daño, `255`=ignorar).
*   **Salida del Modelo:** `[B, 5, N, N]` (Logits por clase).

**Regla de oro:** Nadie escribe `N=512` en el código. Se usa la variable `N` o se infiere de la forma del tensor (`x.shape[-1]`).

---

### 👥 2. Roles, Entregables y Pruebas Independientes

#### 🗂️ Persona 1: Ingeniero de Datos
*   **Carpeta:** `data_pipeline/`
*   **Entregable:** Un `DataLoader` que lea desde `/kaggle/input/` (o desde una carpeta local `test_data/` con solo 4 pares de imágenes), rasterice los GeoJSON con `cv2.fillPoly` y devuelva las tuplas `(imagenes, mascaras)`.
*   **Prueba (Test Independiente):** 
    *   Correr el loader local con las 4 imágenes.
    *   Imprimir formas: `imagenes.shape == [B, 2, 3, N, N]` y `mascaras.shape == [B, N, N]`.
    *   Verificar que las máscaras tengan valores entre 0-4 y 255.
    *   **Visual:** Guardar un PNG con la máscara superpuesta a la imagen "después" para confirmar que los polígonos se pintaron bien.

#### 🧠 Persona 2: Arquitecto del Modelo
*   **Carpeta:** `models/`
*   **Entregable:** Una clase `ModeloSiames(nn.Module)` usando `segmentation_models_pytorch` (encoder preentrenado + decoder U-Net). Debe recibir `[B, 2, 3, N, N]` y devolver `[B, 5, N, N]`.
*   **Prueba (Test Independiente):**
    *   Generar tensores falsos con `torch.randn(2, 2, 3, 64, 64)` y `torch.randn(2, 2, 3, 128, 128)`.
    *   Pasar por el modelo y verificar que las salidas son `[2, 5, 64, 64]` y `[2, 5, 128, 128]`.
    *   **Nota:** Al instanciar el modelo por primera vez, descargará los pesos de ImageNet. Hacerlo en un script aparte.

#### 🏋️ Persona 3: Entrenador
*   **Carpeta:** `training/`
*   **Entregable:** Un `train.py` con el bucle de entrenamiento, optimizador y **la función de pérdida** (CrossEntropy con `ignore_index=255` + Dice Loss o Focal Loss, con pesos por clase para compensar el fondo).
*   **Prueba (Test Independiente con Mini-Siamés):**
    *   **No uses el modelo real todavía.** Crea un `MiniModeloSiamés` de 2 capas convolucionales para cada rama, concatenación y una capa final.
    *   Crea un `MockDataLoader` que devuelva `[2, 2, 3, 64, 64]` y `[2, 64, 64]` (con algunos 255).
    *   Corre 2 épocas. Verifica que la pérdida baja, el optimizador actualiza pesos y se guarda un `test_model.pth`.
    *   *Cuando pase, solo cambias el Mini-Siamés por el modelo de la Persona 2.*

#### 📊 Persona 4: Evaluador y Desarrollador de la Demo
*   **Carpeta:** `evaluation_demo/`
*   **Entregable:** `evaluate.py` (IoU, F1 por clase) y `app.py` con **Gradio** (recibe 2 imágenes, devuelve mapa de colores o matriz de área de daño).
*   **Prueba (Test Independiente):**
    *   **Evaluador:** Usar el `test_model.pth` de la Persona 3 para calcular métricas (aunque salgan malas, verificar que el código no falle).
    *   **Demo:** Hacer que la interfaz reciba imágenes, haga un `resize` dinámico a \(N \times N\), pase un modelo falso (que devuelva ruido) y pinte un mapa de colores. Verificar que la interfaz no se rompa.

#### 📝 Persona 5: Coordinador / Documentador
*   **Carpeta:** Raíz del proyecto.
*   **Entregable:** `README.md`, `requirements.txt` y el informe final.
*   **Prueba (Test Independiente):**
    *   Crear un entorno virtual limpio, instalar `requirements.txt` y correr los scripts `test_model.py` y `test_train.py`.
    *   Asegurar que todos suban su código a ramas separadas en GitHub y hacer el "Merge" sin conflictos.

---

### 🗓️ 3. Cronograma (1 Mes)

*   **Semana 1: Desarrollo en Paralelo (Mocks).** Todos construyen sus piezas y pasan sus pruebas independientes. Persona 5 configura el repo.
*   **Semana 2: Integración y Overfit Test.** Persona 3 junta el DataLoader real y el Modelo real. Corren el "Overfit Test" (entrenar con 4 imágenes hasta que la pérdida sea casi 0). Si falla, se depura en equipo.
*   **Semana 3: Entrenamiento Completo.** Persona 3 lanza el entrenamiento en Kaggle. Persona 4 conecta el modelo real a la demo y calcula métricas.
*   **Semana 4: Demo, Ajustes y Entrega.** Pruebas de estrés de la demo, redacción del informe y preparación de la presentación.

---

### 🚦 4. Reglas de Oro del Equipo

1.  **Cero dependencias cruzadas en la Semana 1:** Nadie espera a que el otro termine. Se prueban con datos y modelos falsos.
2.  **La pérdida es del Entrenador:** Persona 3 decide la combinación exacta (ej. CE ponderada + Dice) y la implementa. Persona 1 le informa sobre el desbalance de clases.
3.  **GitHub:** Trabajen en ramas (`rama-datos`, `rama-modelo`, etc.). No suban a `main` sin que su código pase su prueba independiente.
4.  **Reuniones diarias (15 min):** Qué probé ayer, qué haré hoy, tengo un bloqueo en...
5.  **Manejo del tamaño:** Todo el código debe aceptar cualquier \(N\) potencia de 2. En la demo, si el usuario sube una imagen grande, el código la recorta o redimensiona a la \(N\) elegida antes de pasarla por el modelo.

