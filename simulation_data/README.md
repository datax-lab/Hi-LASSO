## Simulation Data

### Setting
- Ground Truth(beta):

$$
\begin{cases}
    \beta_j \sim N(0, 4)              & \text{for } j = 1, \dots, \text{number of nonzero features} \\
    \beta_j = 0              & \text{for } j = 51, \dots, p \\
\end{cases}
$$


- $\sigma = 3$

### Generation

- $\mathbf{X} \sim N(\mathbf{0}, \sum)$


- $\boldsymbol{\epsilon} \sim N(0, \sigma^2)$


- $\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}$
