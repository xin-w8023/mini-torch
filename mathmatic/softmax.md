# Softmax

## Definition

Here only provide equations, for more detail definition, please refer to [wiki-Softmax](https://en.wikipedia.org/wiki/Softmax_function).

Let $X = [x_1, x_2, ..., x_n]$, then $Softmax(X) = [\frac{e^{x_i}}{\sum_{j}^{n}{e^{x_j}}}, \frac{e^{x_2}}{\sum_{j}^{n}{e^{x_j}}}, ..., \frac{e^{x_n}}{\sum_{j}^{n}{e^{x_j}}}]$

## Derivative

For each $x_i$, we could write a partial derivative equation like this

$$
\frac{\partial Softmax(X)}{\partial x_i} = \frac{\partial \frac{e^{x_1}}{\sum_{j}^{n}{e^{x_j}}}}{\partial x_i} + \frac{\partial \frac{e^{x_2}}{\sum_{j}^{n}{e^{x_j}}}}{\partial x_i} + ... \frac{\partial \frac{e^{x_n}}{\sum_{j}^{n}{e^{x_j}}}}{\partial x_i}
$$

For $i$ th item

$$
\frac{\partial \frac{e^{x_i}}{\sum_j^n{e^{x_j}}}}{\partial x_i} = \frac{e^{x_i}}{\sum_{j}^{n}{e^{x_j}}} - \frac{e^{x_i} * e^{x_i}}{[\sum_{j}^{n}{e^{x_j}}]^2}
                                                                = \frac{e^{x_i}}{\sum_{j}^{n}{e^{x_j}}} * (1 - \frac{e^{x_i}}{\sum_{j}^{n}{e^{x_j}}})
                                                                = Softmax(x_i) * (1 - Softmax(x_i))
                                                                = Softmax(x_i) - [Softmax(x_i)]^2
$$

For not $i$ th item, denote by $z$

$$
\frac{\partial \frac{e^{x_z}}{\sum_j^n{e^{x_j}}}}{\partial x_i} = 0 - \frac{e^{x_z} * e^{x_i}}{[\sum_{j}^{n}{e^{x_j}}]^2}
                                                                = - Softmax(x_z) * Softmax(x_i)
$$

Then overall

$$
\frac{\partial Softmax(X)}{\partial x_i} = Softmax(x_i) - [Softmax(x_i)]^2 + \sum_{j!=i}^{n}{-Softmax(x_z) * Softmax(x_i)}
                                         = Softmax(x_i) - \sum_{j}^{n}{Softmax(x_z) * Softmax(x_i)}
$$

### Example code

```python
import numpy as np

N = 10
# softmax computation
x = np.arange(N)
exp = np.exp(x)
softmax = exp / np.sum(exp)

# derivative
grad = np.zeros_like(x)
for i in range(N):
    grad[i] = softmax[i] - np.sum(softmax[i] * softmax)
```
