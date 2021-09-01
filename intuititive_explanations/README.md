# Intuitive explanations

Explaining a bunch of random stuff that I've learned so far in case I do accidentally wipe my brain someday and want to revert.

The order is really really random, but most of the stuff here is related to deep learning

<!-- 
- chain rule 
- backprop
- why do we need activation functions ?
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
- activation functions and where to use each:
    - sigmoid
    - relu
    - leaky relu
    - tanh
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

Given a model M with 3 layers a, b, c: if the calculated loss is E, then we can calculate the gradients of each layer while "walking backwards" though each layer.

```python
y = c(b(a(x)))  ## a -> b -> c
loss = loss_function(y, label)
loss.backward()

"""
when loss.backward is calle, the following things happen in sequence:

1. Calculate dloss/dy
## todo
"""
```

A more serious explanation is shown below:

<img src = "https://github.com/Mayukhdeb/math-stuff/raw/main/backprop/images/backprop_main.jpg" width = "100% ">