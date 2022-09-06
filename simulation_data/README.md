## Simulation Data

### Setting
- Ground Truth($\beta$):    

$$
\begin{cases}
    \beta_j \sim N(0, 4)              & \text{for } j = 1, \dots, \text{number of nonzero features} \\
    \beta_j = 0              & \text{for } j = 51, \dots, p \\
\end{cases}
$$


- $\sigma = 3$


- Covariance matrix($\sum$):    
$$
\sigma^2 \times
\begin{bmatrix}
\sum^{15}_{0.9} & 0 & 0 & 0\\
0 & \sum^{15}_{0.9} & \mathbf{J}_{0.3} & 0\\
0 & \mathbf{J}^{T}_{0.3} & \sum^{20}_{0.9} & 0\\
0 & 0 & 0 & \mathbf{I}^{p - \text{number of nonzero features}}\\
\end{bmatrix}
$$

### Generation

- $\mathbf{X} \sim N(\mathbf{0}, \sum)$


- $\boldsymbol{\epsilon} \sim N(0, \sigma^2)$


- $\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}$
