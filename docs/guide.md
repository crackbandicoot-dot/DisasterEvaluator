# Machine Learning and neural networks

This guide explains the basics of machine learning and how a neural network works.

---

## 1. Concepts

### 1.1 What machine learning is

Machine learning is the discipline responsible for **finding a function** that answers a problem. Instead of writing that function by hand, we use a **learning algorithm** that searches for it for us from data.

Two types of values must be distinguished:

- **Parameters**: the numbers of the function that the algorithm adjusts by itself (for example, the weights of a neural network).
- **Hyperparameters**: the configuration of the learning algorithm, which we choose before training (for example, the learning rate, the batch size, or the number of layers).

### 1.2 Areas of machine learning

- **Supervised learning**: the data come with the correct answer and the model learns to reproduce it. This is our case: each image comes with the damage mask. Examples: filtering spam emails from already labeled emails, estimating the price of a house from past sales, or detecting pneumonia in radiographs with confirmed diagnoses. The answer can be a category (classification) or a number (regression).
- **Unsupervised learning**: the data do not come with an answer and the model looks for structure on its own, for example grouping similar examples. Examples: segmenting customers by their purchasing behavior, grouping satellite images by landscape type, detecting unusual card transactions, or reducing hundreds of variables to a few with PCA.
- **Reinforcement learning**: an agent tries actions and learns from the rewards it receives. Examples: AlphaGo learning to play Go, a robot learning to walk, or a system that adjusts traffic lights to minimize traffic.

### 1.3 Other types of models

There are many models besides neural networks. Two classic examples:

- **KNN (k nearest neighbors)**: it does not train anything. To classify a new point, it looks for the $k$ closest examples and votes for the majority class.
- **Decision tree**: it asks yes-or-no questions about the features of the data. At each step it chooses the question that best separates the classes and repeats the process inside each branch.

We focus on neural networks because that is what the project uses and because it is what dominates the field today.

### 1.4 Tensors

For us, a tensor is simply a **multi-dimensional array of arrays**. A color image, for example, is a three-dimensional tensor: height, width, and channels (red, green, and blue). Machine learning frameworks receive and return tensors.

### 1.5 Frameworks in Python

- **PyTorch**: the most used today in research and very popular in general. A direct competitor to TensorFlow.
- **TensorFlow**: Google's framework for deep learning.
- **Keras**: a high-level interface built on top of TensorFlow, designed to write networks with little code.
- **scikit-learn** (`sklearn`): designed for classic models such as KNN, decision trees, or regressions. It also brings validation tools and metrics. Neural networks are not its strength.

### 1.6 How we measure how good a model is

To find the function we use a dataset. We divide it into parts:

- **Training (train)**: with this the algorithm searches for the function.
- **Validation**: used to make decisions during development (choosing hyperparameters, deciding when to stop, comparing models).
- **Test**: data that the model never saw and that we did not use to decide anything, to measure how it really behaves at the end.

#### Accuracy

It is the percentage of correct predictions:

$$\text{accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$$

It seems sufficient, but **it is misleading when classes are imbalanced**. If 99% of the pixels in an image are background, a model that always answers "background" has 99% accuracy and is useless.

#### False positives and false negatives

- **False positive**: the model says yes, but it was no.
- **False negative**: the model says no, but it was yes.

To see it with the mechanism of each case, it helps to look at what "yes" means in the problem:

- **Autonomous driving.** The model receives an image of the street and must decide "there is a pedestrian" (yes) or "there is no pedestrian" (no). If there is a pedestrian ahead and the model answers "there is no pedestrian," it said no when it was yes: it is a **false negative**. The car does not brake and runs over the pedestrian. That is the dangerous error, because the cost of not detecting it is an accident.
- **Spam filter.** The model must decide "it is spam" (yes) or "it is not spam" (no). If a legitimate work email is marked as spam, the model said yes when it was no: it is a **false positive**. The important email is lost in the spam folder, and that is the costly error; in contrast, letting spam through (a false negative) is only annoying.

Which one hurts more depends on the context and on what "yes" means in each problem: in autonomous driving the costly error is not detecting the pedestrian (false negative), in a spam filter it is marking a good email (false positive).

#### Precision, recall, and F1

Accuracy fails with imbalanced classes because it counts all correct predictions equally, and the background dominates the count. Precision, recall, and F1 look **only at the class we care about** (the "positive" class), which in our case is "this pixel has damage," and therefore they are not fooled by the background. We use the same notions as above: $TP$ true positives, $FP$ false positives, and $FN$ false negatives.

##### Precision

Of all the times the model said "yes," how many did it get right?

$$\text{precision} = \frac{TP}{TP + FP}$$

It is useful because **it penalizes false positives**: every time the model shouts "damage" when there is none, precision drops. In our project, low precision means we mark buildings that are fine as damaged, and rescue teams waste time going to check false alarms. It is used when we want the system to be reliable every time it warns.

##### Recall

Of all the cases that were truly "yes," how many did the model find?

$$\text{recall} = \frac{TP}{TP + FN}$$

It is useful because **it penalizes false negatives**: every real damage the model lets through without marking lowers recall. In our project, low recall means there are destroyed buildings that the map does not show, and people who need help are not sent. It is used when what matters most is not letting anything through.

##### F1

Precision and recall pull in opposite directions: if we lower the threshold to avoid missing damages, recall goes up but precision goes down (we mark too much), and if we raise it, precision goes up but recall goes down (we miss cases). F1 combines the two into a single number:

$$F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

It is useful when the balance must be summarized in a single figure and one does not want to choose between precision and recall. It uses the **harmonic mean** (not the arithmetic mean) on purpose: if one of the two is low, F1 collapses even if the other is perfect. This forces both to be reasonable, and one cannot "cheat" by improving only one.

### 1.7 Overfitting

**Overfitting** occurs when the model memorizes the training data instead of learning the pattern, and fails on new data. It is detected by comparing training error with validation error: if the former keeps decreasing and the latter starts increasing, there is overfitting.

What helps most is **variety in the data**. If all examples look alike, the model learns only that. Variety is not just quantity: a thousand photos of the same street with the same lighting teach little. **Negative examples** are also needed (when there is no car, not only when there is one), because without them the model can answer "car" to everything. When data is scarce, one can resort to **data augmentation** (rotations, crops, brightness changes) and to **regularization**, which penalizes overly complicated models.

### 1.8 Cross-validation (K-fold)

**K-Fold cross-validation** is a resampling method in machine learning for estimating how well a model generalizes; it is used to select a model or hyperparameters for a model. The dataset is divided into **k** non-overlapping folds. The model is trained **k** times: on each occasion, one fold is held out for validation and the other **k-1** folds are used for training. Each data point is used for validation exactly once.

The **k** validation scores are summarized, usually with their **average**:

$$ \bar{x} = \frac{1}{k}\sum_{i=1}^{k} x_i $$

and their **variance**

$$
s^2 = \frac{1}{k-1}\sum_{i=1}^{k}\left(x_i-\bar{x}\right)^2
$$

to estimate the expected performance and stability of the model.

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

## 2. Neural networks

A neural network is, in the mathematical sense, **a composition of functions**.

$$ F(x)=f_L\circ f_{L-1}\circ\cdots\circ f_1(x)$$

these functions are usually called layers; each layer usually has this form

$$\qquad f_\ell(x)=g_\ell(W_\ell x+b_\ell)$$

>**Note**: I say "usually" because there are several types of neural networks; this specific example is of a **feedforward** network.

The reason for composing them is that, if we did not, there would be decisions we could not make. The best way to understand it is to follow the story of why each piece appeared.

### 2.1 A neuron

We want a computer to learn to multiply by 2. We start with the simplest possible function, a single neuron:

$$f(x) = w \cdot x + b$$

Here $w$ (the weight) and $b$ (the bias) are the parameters. The learning algorithm should find $w = 2$ and $b = 0$. For this problem, one neuron is enough.

### 2.2 The limit of a single neuron

Let us start with a very simple function that **is not** affine: the absolute value.

$$
f(x)=|x|.
$$

> Note: Affine functions are those of the form $y= Wx+B$; $B$ is what distinguishes them from a linear one

A single affine neuron has the form

$$
\hat f(x)=wx+b.
$$

Suppose there exist $w$ and $b$ such that $\hat f(x)=|x|$ for all $x$. Evaluating at $x=1$ and at $x=-1$:

$$
w+b=1,\qquad -w+b=1.
$$

Adding both equations: $2b=2$, so $b=1$. Subtracting them: $2w=0$, so $w=0$. But then

$$
\hat f(0)=w\cdot 0+b=1\ne |0|=0.
$$

Contradiction. Therefore, **there are no $w$ and $b$** that represent $|x|$ with a single affine neuron.

### 2.3 Composing layers, and why the activation function is needed

If we compose two affine functions, the result is still an affine function:

$$w_2 (w_1 x + b_1) + b_2 = (w_2 w_1)\,x + (w_2 b_1 + b_2)$$

That is, stacking linear layers adds nothing. That is why between layer and layer we apply a **nonlinear activation function** $\sigma$, for example ReLU:

$$\sigma(z) = \max(0, z)$$

Now, a two-layer network has the form:

$$f(x) = W_2 \, \sigma(W_1 x + b_1) + b_2$$

With this, the network can represent functions that a single neuron cannot, such as XOR. And by adding more layers, increasingly complex functions. That is the classic neural network, the multilayer perceptron.

> **Note:** this classic form depends heavily on the domain and almost no one uses it as is today. It is taught because it helps one understand the idea. For images, for example, other architectures are used, as we will see at the end.

---

## 3. How a network is trained

We already know the form of the model: a function $f$ that, given an example $x$, produces a prediction

$$
\hat{y}=f(x;\theta),
$$

where $\theta$ are the weights or parameters of the model.

What remains is to know how to find good values of $\theta$.

### 3.1 The parameters as a vector

Although a network may have many weight matrices and many bias vectors, it is convenient to think of **all** its parameters as a single vector.

If the network has $L$ layers with matrices

$$
W_1,\dots,W_L
$$

and biases

$$
b_1,\dots,b_L,
$$

we define

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

where $\mathrm{vec}(W)$ means **stacking the columns** of $W$ one below another.

That is, $\theta$ is simply the concatenation of:

- the columns of each weight matrix $W_\ell$, and
- each bias vector $b_\ell$.

All of that together forms a vector of dimension $p$, the total number of parameters in the network.

Thus, even if $f$ is a complicated function (convolutions, activations, pooling, etc.), the model is completely determined by a vector

$$
\theta \in \mathbb{R}^p.
$$

Training, then, is finding that vector.

### 3.2 The loss function

To be able to find this vector, we need a way to measure "how badly" the model is doing.

That measure is the **loss function**.

#### General definition

Let

$$
D=\{(x_i,y_i)\}_{i=1}^n
$$

be the training dataset.

The loss function is a function of the model's weights:

$$
L(\theta;D)
=
\frac{1}{n}
\sum_{i=1}^n
\ell\big(y_i, f(x_i;\theta)\big),
$$

where $\ell$ is the loss of a single example.

Here it is useful to distinguish two pieces:

- $D=\{(x_i,y_i)\}$ is the **dataset**.
  - During training we treat it as **constant**.

- $\theta \in \mathbb{R}^p$ is the **model's weight vector**.
  - These are the **variables** we want to adjust.
  - The loss changes when $\theta$ changes.

- $L(\theta;D)$ is then a function that, for a fixed dataset, tells us how much the model is wrong with some weights $\theta$.

Training is then **minimizing the loss with respect to the weights**:

$$
\theta^\star
=
\arg\min_{\theta \in \mathbb{R}^p}
L(\theta;D).
$$

That is, we look for the vector $\theta$ that makes the model make as few mistakes as possible on the training dataset.

The concrete form of $\ell$ depends on the type of problem.

#### 3.2.1 Predicting a number

Mean squared error is used:

$$
\ell(y,\hat{y})=(y-\hat{y})^2.
$$

Then,

$$
L(\theta;D)
=
\frac{1}{n}
\sum_{i=1}^n
\big(y_i-\hat{y}_i\big)^2,
$$

where

$$
\hat{y}_i=f(x_i;\theta).
$$

It heavily penalizes large errors.

#### 3.2.2 Segmentation where almost everything is background

Here the **Dice loss** stands out.

If $A$ is the set of pixels predicted as special and $B$ is the real set of these pixels, we define

$$
\mathrm{Dice}(A,B)
=
\frac{2\,|A\cap B|}{|A|+|B|},
$$

and

$$
\ell_{\text{Dice}}
=
1-\mathrm{Dice}(A,B).
$$

The function directly measures the overlap between prediction and truth, normalized by the size of both sets.

If the set occupies very few pixels, the background (which is enormous) does not dominate the loss:

- the denominator $|A|+|B|$ includes only the region of interest,
- not all pixels.

That is why it is widely used in medical segmentation, damaged buildings, etc.

### 3.3 Gradient descent

The gradient of the loss points toward where the loss **increases** fastest. So to lower it we move in the opposite direction:

$$\theta \leftarrow \theta - \eta \, \nabla L(\theta)$$

Where $\theta$ are all the parameters and $\eta$ is the **learning rate**, a hyperparameter that controls the step size.

Zero gradient is a necessary but not sufficient condition for a minimum: maxima and **saddle points** also satisfy it. In neural networks the loss is not convex, so in general we can only hope to reach a point where the gradient is zero.

### 3.4 Stochastic gradient descent by mini-batches

This algorithm is the most used in practice; it updates parameters using the average gradient computed over a small subset of the training data, randomly sampled, called a **mini-batch** (often denoted as $\mathcal{B}_t$ of size $b$). The size of this mini-batch normally ranges from 2 to a few hundred, although it can be larger, often limited by GPU memory.

The update rule at iteration $t$ is:

$$
\theta_{t+1} \leftarrow \theta_t - \eta_t \cdot \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \nabla \ell_i(\theta_t)
$$

where $\eta_t$ is the learning rate (or step size) at iteration $t$, and $\nabla \ell_i(\theta_t)$ is the gradient of the loss for a single data point $i$ evaluated at $\theta_t$.

The idea is that the mini-batch gradient serves as an **unbiased estimate** of the full-batch gradient, that is, $\mathbb{E}[\nabla L_{\mathcal{B}_t}(\theta_t)] = \nabla L(\theta_t)$. This means that, *on average*, the update points in the direction of the true gradient, which allows convergence while being much more efficient per iteration than full-batch GD.

> *Note*: in standard epoch-based training (an epoch means going through the entire training set once), mini-batches **are not** reused as the same fixed groups. In each epoch, the training set is shuffled again and split into new mini-batches $\mathcal{B}_1, \dots, \mathcal{B}_K$, where $K \approx \lceil N/b \rceil$ for a dataset of size $N$ and batch size $b$. Thus, the examples are reused in each epoch, but their grouping into batches changes. If they are **not** shuffled, then one effectively rotates through the same fixed batches in each epoch; the gradients of individual batches are then not unbiased estimates of the full gradient, although the average over a complete epoch equals the full gradient. Some implementations, instead, sample a new random mini-batch **with replacement** at each step, so there is no fixed list of batches. In the previous notation, $\mathcal{B}_t$ is therefore generally a new random subset at each iteration, not a fixed batch that simply cycles.

### 3.5 When to stop

With mini-batches, the gradient we use at each iteration is noisy:

$$
\nabla L_{\mathcal{B}_t}(\theta_t)
=
\frac{1}{|\mathcal{B}_t|}
\sum_{i\in \mathcal{B}_t}
\nabla \ell_i(\theta_t),
$$

and even if we are close to a minimum, its norm can still be large. That is why it is **not** advisable to stop by checking whether $\|\nabla L_{\mathcal{B}_t}(\theta_t)\|$ is close to zero.

In practice, a **validation set** different from the training set is used. Let

$$
D_{\text{val}}=\{(x_i^{\text{val}},y_i^{\text{val}})\}_{i=1}^{n_{\text{val}}}
$$

be a set disjoint from $D$. We define its loss as

$$
L_{\text{val}}(\theta;D_{\text{val}})
=
\frac{1}{n_{\text{val}}}
\sum_{i=1}^{n_{\text{val}}}
\ell\big(y_i^{\text{val}}, f(x_i^{\text{val}};\theta)\big).
$$

This loss **is not used to update** $\theta$; it is only used to decide when to stop and which weights to keep.

At the end of each epoch $e$, we compute $L_{\text{val}}(\theta_e)$, where $\theta_e$ are the weights after going through the training set $D$ once. The idea of "stops improving" is formalized as follows:

- Choose a **patience** $P\in\mathbb{N}$ and a **minimum improvement** $\delta\ge 0$.
- Store the best validation value seen so far, $L_{\text{val}}^{\text{best}}$, and the corresponding weights $\theta^\star$.
- After each epoch:
  - if $L_{\text{val}}(\theta_e) < L_{\text{val}}^{\text{best}}-\delta$, there is significant improvement: we update $L_{\text{val}}^{\text{best}}$ and $\theta^\star$, and reset the counter;
  - otherwise, we increase the counter of epochs without improvement.
- Stop when the counter reaches $P$, or when a maximum number of epochs is reached.

That is, **"stops improving" means that for $P$ consecutive epochs the validation loss has not decreased by at least $\delta$ with respect to the previous best value**. At the end, the best weights $\theta^\star$ are returned, not necessarily the last ones.

If $\delta=0$, any strict improvement counts. Patience $P$ allows tolerating mini-batch noise and not stopping because of a small fluctuation. In addition, it is advisable to set a maximum number of epochs to avoid infinite training.

### 3.6 Backpropagation

To take each step we need the gradient of the loss with respect to all parameters, $\nabla L(\theta;D)$; frameworks take care of this for us, but it is good to know at a high level the idea they use. The network is a composition of operations: inputs, weights, biases, preactivations, activations, prediction, and loss. Each of these quantities can be thought of as a **node** in a graph, and we put an edge $a\to b$ if $b$ is computed using $a$.

For example, in a simple network:

$$
x,W_1 \longrightarrow z_1=W_1x+b_1
\longrightarrow h_1=\sigma(z_1)
\longrightarrow z_2=W_2h_1+b_2
\longrightarrow \hat{y}
\longrightarrow \ell(y,\hat{y}).
$$

> **Note**: it helps to view this example from right to left at first, and then from left to right

All arrows go from something that is used to compute something else toward its result. One never returns to an earlier node. That is why the graph is a **directed acyclic graph**: a **DAG**. If there were a cycle, a variable would depend on itself and we could not order the computation forward or backward cleanly.

The loss $L$ is a composition of functions along the paths that go from $\theta$ to $\ell$. To know how much a node affects the loss, one must multiply local derivatives along those paths: that is the chain rule. If we did it naively, many subexpressions would be repeated.

**Backpropagation** traverses the DAG in reverse order: it starts at the loss and propagates backward "how much $L$ changes if each node changes." At each node it reuses the already computed values from its consumers and the local derivatives, and stores the intermediate results of the forward pass. Thus, each subpart is computed only once: that is dynamic programming over the DAG.

---

## 4. Fine tuning

Normally a network starts with random weights. In **full fine tuning** we start with the weights of a model already trained on another task and move from there. The model has already learned useful representations, so training is usually faster and needs less data. Typically a smaller learning rate is used so as not to destroy what it already knew.