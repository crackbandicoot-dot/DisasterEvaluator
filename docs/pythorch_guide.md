# PyTorch Guide

PyTorch is a thin Python layer over a C++ tensor and autograd runtime, built from a few classic patterns: Composite for models, Strategy for losses and optimizers, Iterator for data, Memento for checkpoints and Command for the autograd graph. The math in the guide describes pure functions on exact real numbers; the code runs stateful objects, mutates memory and computes in float32.

> **Disclaimer**: Learning PyTorch deeply requires a lot of years of experience; it is full of non-intuitive decisions and things, and the normal thing to do is understand the high-level picture rather than historical baggage and specific decisions that are very unintuitive until you read the docs. This document does not pretend to be a substitute for that.

## Design patterns

Every PyTorch component is a plain Python object you can subclass, swap or call, which follows from the four principles in its own paper: be Pythonic, put researchers first, provide pragmatic performance, and worse is better. The pattern names below are our reading of the design, not PyTorch terminology.

| Pattern | Where in PyTorch | What it buys |
| --- | --- | --- |
| Composite | `nn.Module` holds child modules, `nn.Sequential` chains them; `parameters()`, `.to()` and `.train()` recurse down the tree | The guide's $F = f_L \circ \ldots \circ f_1$ is literally a tree, and a whole model is used like a single layer |
| Strategy | Losses, optimizers, LR schedulers, `Sampler`s and `collate_fn` each share one interface | Swap SGD for Adam without touching the training loop |
| Iterator | `DataLoader` is an iterable: sampler yields indices, `dataset[i]` fetches samples, `collate_fn` stacks them into a batch | Shuffling, batching and worker processes hide behind a `for` loop |
| Command | Each node of the autograd graph is a `Function` object that knows its own backward step; the backward pass replays them in reverse | Any Python program is differentiable, and the graph is rebuilt on every forward pass |
| Memento | `state_dict()` captures parameters and persistent buffers as a dict, `load_state_dict()` restores them | Checkpoints and best-weights early stopping; the dict holds references, so `copy.deepcopy` it to snapshot |
| Multiple dispatch | Every operator call goes through a dispatcher keyed on device, layout and autograd state, which picks the kernel | One `a + b` runs on CPU, on CUDA, with or without gradient tracking |
| Views over shared storage | A tensor is sizes, strides and an offset over a `Storage`; slicing and transposing share memory | O(1) views and `torch.from_numpy` without a copy, but a write through one view shows in all |
| Scoped state (context manager) | `torch.no_grad()` and `torch.inference_mode()` turn recording off for a block and restore it on exit; they also work as decorators | Validation without building a graph, and no way to forget to turn it back on |

Views share storage: a write through one view is visible in the others; call `.clone()` to break sharing.

These are conventions, not enforced types: a dataset is any object that behaves like a possibly lazy list, and a layer is any callable that holds parameters. That is why a GAN's two models and two optimizers fit in a dozen lines of plain Python.

## The training loop, annotated

Every line of the training loop is either one of the patterns above or one of the gaps in the table; the comments name the table rows.

```python
import copy
import torch
from torch import nn

model = MLP(d_in, d_hidden, d_out)                    # Composite: a tree of modules, theta = model.parameters()
loss_fn = nn.BCEWithLogitsLoss()                      # Strategy: any loss with the same call signature
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Strategy: swap in SGD and the loop is unchanged

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Multiple dispatch: picks the kernel
model.to(device)

n_val = len(val_loader.dataset)
best, best_state, bad, patience, delta = float('inf'), None, 0, 10, 0.0

for epoch in range(max_epochs):
    model.train()                                     # mode flag
    for x, y in train_loader:                         # Iterator: sampler, then dataset[i], then collate_fn
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)                   # forward: Command graph, one node per operator
        opt.zero_grad(set_to_none=True)               # .grad accumulates; set_to_none=True frees it
        loss.backward()                               # Command: replay backward nodes in reverse
        opt.step()                                    # in-place update of theta

    model.eval()                                      # eval() switches dropout/batchnorm; it does not disable autograd
    with torch.no_grad():                             # Scoped state: record no graph
        val = 0.0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val += loss_fn(model(x), y).item() * len(x)
        val /= n_val                                  # weighted mean

    if val < best - delta:                            # patience and minimum improvement
        best, bad = val, 0
        best_state = copy.deepcopy(model.state_dict())    # Memento: state_dict holds references; deepcopy snapshots
    else:
        bad += 1
        if bad >= patience:
            break

model.load_state_dict(best_state)                     # Memento restore: best weights, not the last ones
```

For saving a checkpoint, prefer detaching and moving to CPU: `{k: v.detach().cpu() for k, v in model.state_dict().items()}`.

Nothing in the loop is specific to segmentation: swap in BCE plus Dice and a loader of images and masks, and the loop stays the same, which is the Strategy pattern doing its job.