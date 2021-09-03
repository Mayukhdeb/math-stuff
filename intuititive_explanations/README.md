# Intuitive explanations

Explaining a bunch of random stuff that I've learned so far in case I do accidentally wipe my brain someday and want to revert.

The order is really really random, but most of the stuff here is related to deep learning

<!-- 
- chain rule -done
- backprop -done 
- why do we need activation functions ? 
    - activation functions and where to use each:
        - sigmoid
        - relu
        - leaky relu
        - tanh
- dropout layers
- weight decay 
- learning rate
- lr schedulers 
- batchnorm
- vanishing gradients
- overfitting
- dying relu -> leaky relu
- residual layers 
- loss functions when and why to use each 
    - MSE loss
    - cross entropy loss
    - BCELoss
    - Dice score/jaccard index

- why normalizing data is important, explain `transforms.Normalize`
- what is label smoothing and why is is used sometimes 
- GANS:
    - Generator loss 
    - discriminator loss 
    - nash equilibrium 
 -->

## Chain rule 

If a small change in x causes a small change in y, and if a small change in y causes a small change in p. Then it is possible to also calculate what happens to p given a small change in x. 

<img src = "https://github.com/Mayukhdeb/math-stuff/raw/main/backprop/images/chain_rule.jpg" width = "30% ">

## Backpropagation

Its an algorithm for calculating the gradient of a loss function with respect to variables of a model. Backprop asks the question: *By how much does each learnable parameter affect the loss ?*


```python

class SomeLayer:
    def __init__(self, w0, b0):
        """
        init weights and biases 
        """
        self.w = w0
        self.b = b0
    def forward(self, x):
        """
        local grads:
            d(out)/d(self.w) = x
            d(out)/d(self.b) = 1
        
        d(out)/d(x) = self.w
        """
        out = self.w*x + self.b
        return out 

class Model:
    def __init__(self):
        self.layer1 = SomeLayer()
        self.layer2 = SomeLayer()
        self.layer3 = SomeLayer()

    def forward(self, x):
        """
        x -> layer 1 -> x1 -> layer2 -> x2
        """
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return x2

model = Model()
y = model(x)

loss = loss_function(y, label)
loss.backward()
```

When `loss.backward()` is called, the following things happen:
> "it" refers to the autograd engine
1. It first calculates `d(loss)/d(x2)` (how much does the loss vary given a small change in x2)

2. Once we know `d(loss)/d(x2)`, we can use it to calculate the following 3 things:

    * *How much do the weights of layer2 affect the loss ?* `d(loss)/d(layer2.w)`

    * *How much do the biases of layer2 affect the loss ?* `d(loss)/d(layer2.b)` 

    * *How much does the input of layer2 (x1) affect loss ?* `d(x2)/d(x1)` -- this will be needed when we want to "chain" the gradients from layer1 to the loss.

    ```python
    d(loss)/d(layer2.w) = d(x2)/d(layer2.w) * d(loss)/d(x2) ## where d(x2)/d(layer2.w) is known from the local gradients 
    ```

    ```python
    d(loss)/d(layer2.b) = d(x2)/d(layer2.b) * d(loss)/d(x2) ## where d(x2)/d(layer2.b) is known from the local gradients 
    ```

3. Now that we know `d(x2)/d(x1)`, we can calculate the final 2 gradients that are needed:

    * *How much do the weights of layer2 affect the loss ?* `d(loss)/d(layer1.w)`

    * *How much do the biases of layer2 affect the loss ?* `d(loss)/d(layer1.b)`  

    ```python
    d(loss)/d(layer1.w) = d(x1)/d(layer1.w) * d(x2)/d(x1) * d(loss)/d(x2) ## where d(x1)/d(layer1.w) is known from the local gradients 
    ```

     ```python
    d(loss)/d(layer1.b) = d(x1)/d(layer1.b) * d(x2)/d(x1) * d(loss)/d(x2) ## where d(x1)/d(layer1.w) is known from the local gradients 
    ```

A more serious explanation [can be found here](https://github.com/Mayukhdeb/math-stuff/tree/main/backprop).

## Activation functions

**Why do we even need them ?**

It helps the network learn complex data, compute and learn almost any function representing a question, and provide accurate predictions.

Without non linear activations, a model with 2 layers might look like: 

```python
y = w2(w1*x + b1) + b2
```

which can be boiled down to a single linear transformation:

```python
y = w1*w2*x + (w2*b1 + b2)
```

A linear transformation is limited in its capacity to solve complex problems and hold less power to learn complex functional mappings from data.

**Which ones should I look out for ?**

* ReLU: This one simply acts like an identity function for all values > 0. 
> One common problem we face with this one is the dying relu problem. Which gets fixed by using a leaky relu. 

* Leaky ReLU: This one fixes the problem that relu had by not completely killing the gradients for negative values to zero.

* Sigmoid: Whatever be the input, the outputs of a sigmoid get bound between 0 and 1. 

> This one has a problem too: for very high or very low values of inputs, there is almost no change to the prediction, causing a vanishing gradient problem.

> ok theres another problem: let us consider the follwing situation: 

"""
out = self.w*x + self.b
"""
where x is the output of a sigmoid layer i.e its always >= 0
            
* Tanh

TODO
