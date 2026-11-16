# Guía de PyTorch

PyTorch es una capa delgada de Python sobre un runtime de tensores y autograd en C++, construida a partir de unos pocos patrones clásicos: Composite para los modelos, Strategy para las pérdidas y los optimizadores, Iterator para los datos, Memento para los checkpoints y Command para el grafo de autograd. Las matemáticas de la guía describen funciones puras sobre números reales exactos; el código ejecuta objetos con estado, muta memoria y calcula en float32.

> **Descargo de responsabilidad**: Aprender PyTorch en profundidad requiere muchos años de experiencia; está lleno de decisiones y cosas poco intuitivas, y lo normal es entender el panorama de alto nivel en lugar del bagaje histórico y las decisiones específicas que resultan muy poco intuitivas hasta que se leen los docs. Este documento no pretende sustituir eso.

## Patrones de diseño

Cada componente de PyTorch es un objeto Python simple que puedes subclassificar, intercambiar o llamar, lo que se deriva de los cuatro principios de su propio artículo: ser Pythonic, poner a los investigadores primero, ofrecer rendimiento pragmático y que «worse is better». Los nombres de patrones a continuación son nuestra lectura del diseño, no terminología de PyTorch.

| Patrón | Dónde en PyTorch | Qué aporta |
| --- | --- | --- |
| Composite | `nn.Module` contiene módulos hijos, `nn.Sequential` los encadena; `parameters()`, `.to()` y `.train()` recorren recursivamente el árbol | El $F = f_L \circ \ldots \circ f_1$ de la guía es literalmente un árbol, y un modelo completo se usa como una sola capa |
| Strategy | Las funciones de pérdida, los optimizadores, los planificadores de LR, los `Sampler` y `collate_fn` comparten una interfaz | Cambiar SGD por Adam sin tocar el bucle de entrenamiento |
| Iterator | `DataLoader` es un iterable: el sampler produce índices, `dataset[i]` obtiene muestras, `collate_fn` las apila en un lote | El mezclado, el agrupamiento en lotes y los procesos trabajadores quedan ocultos detrás de un bucle `for` |
| Command | Cada nodo del grafo de autograd es un objeto `Function` que conoce su propio paso backward; la pasada backward los reproduce en orden inverso | Cualquier programa de Python es diferenciable, y el grafo se reconstruye en cada pasada forward |
| Memento | `state_dict()` captura parámetros y búferes persistentes como un dict, `load_state_dict()` los restaura | Puntos de control y parada temprana con los mejores pesos; el dict contiene referencias, así que usa `copy.deepcopy` para tomar una instantánea |
| Despacho múltiple | Cada llamada a un operador pasa por un despachador indexado por dispositivo, layout y estado de autograd, que elige el kernel | Un único `a + b` se ejecuta en CPU, en CUDA, con o sin seguimiento de gradientes |
| Vistas sobre almacenamiento compartido | Un tensor son tamaños, strides y un desplazamiento sobre un `Storage`; el slicing y la transposición comparten memoria | Vistas O(1) y `torch.from_numpy` sin copia, pero una escritura a través de una vista se ve en todas |
| Estado con ámbito (gestor de contexto) | `torch.no_grad()` y `torch.inference_mode()` desactivan el registro para un bloque y lo restauran al salir; también funcionan como decoradores | Validación sin construir un grafo, y sin forma de olvidarse de volver a activarlo |

Las vistas comparten almacenamiento: una escritura a través de una vista es visible en las demás; llama a `.clone()` para romper la compartición.

Estas son convenciones, no tipos impuestos: un dataset es cualquier objeto que se comporta como una lista posiblemente perezosa, y una capa es cualquier callable que contiene parámetros. Por eso los dos modelos y los dos optimizadores de una GAN caben en una docena de líneas de Python simple.

## El bucle de entrenamiento, anotado

Cada línea del bucle de entrenamiento es uno de los patrones anteriores o uno de los huecos de la tabla; los comentarios nombran las filas de la tabla.

```python
import copy
import torch
from torch import nn

model = MLP(d_in, d_hidden, d_out)                    # Composite: un árbol de módulos, theta = model.parameters()
loss_fn = nn.BCEWithLogitsLoss()                      # Strategy: cualquier pérdida con la misma firma de llamada
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Strategy: cambia a SGD y el bucle no cambia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dispatch múltiple: elige el kernel
model.to(device)

n_val = len(val_loader.dataset)
best, best_state, bad, patience, delta = float('inf'), None, 0, 10, 0.0

for epoch in range(max_epochs):
    model.train()                                     # flag de modo
    for x, y in train_loader:                         # Iterator: sampler, luego dataset[i], luego collate_fn
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)                   # forward: grafo Command, un nodo por operador
        opt.zero_grad(set_to_none=True)               # .grad se acumula; set_to_none=True lo libera
        loss.backward()                               # Command: reproduce los nodos backward en orden inverso
        opt.step()                                    # actualización in-place de theta

    model.eval()                                      # eval() cambia dropout/batchnorm; no desactiva autograd
    with torch.no_grad():                             # Estado con scope: no registrar ningún grafo
        val = 0.0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val += loss_fn(model(x), y).item() * len(x)
        val /= n_val                                  # media ponderada

    if val < best - delta:                            # paciencia y mejora mínima
        best, bad = val, 0
        best_state = copy.deepcopy(model.state_dict())    # Memento: state_dict contiene referencias; deepcopy toma una instantánea
    else:
        bad += 1
        if bad >= patience:
            break

model.load_state_dict(best_state)                     # Restauración Memento: mejores pesos, no los últimos
```

Para guardar un punto de control, prefiere usar detach y mover a CPU: `{k: v.detach().cpu() for k, v in model.state_dict().items()}`.

Nada en el bucle es específico de segmentación: cambia a BCE más Dice y un cargador de imágenes y máscaras, y el bucle sigue igual, que es el patrón Strategy haciendo su trabajo.
