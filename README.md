# Fixed Point Analysis
This is the implementation of fixed point analysis for Recurrent Neural Network by PyTorch.

﻿Sussillo, D., & Barak, O. (2013). [Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks.](https://doi.org/10.1162/NECO_a_00409)  

﻿Niru Maheswaranathan. et al. (2019) [Universality and individuality in neural dynamics across large populations of recurrent networks.](https://papers.nips.cc/paper/9694-universality-and-individuality-in-neural-dynamics-across-large-populations-of-recurrent-networks)

This repository contains the code for the analysis on the canonical task **Frequency-cued sine wave**, which is studied on these papers.


# Experiments

First, train your model by `train.py`. 

## Trajectories and topology of fixed points

- Plot trajectories and fixed points
```bash
$ python plot_trajectories.py --activation relu
```

![trajectory_relu](https://user-images.githubusercontent.com/24406002/71605599-7164f600-2bad-11ea-8fb1-5ffccb8b3f42.png)


- Different points in the same trajectory correspond to one fixed point, 
and different trajectories correspond to different fixed point.

```bash
$ python compare_fixed_point.py --activation relu
```


```
distance between 2 fixed point start from different IC; different time of same trajectory.
2.2076301320339553e-07
distance between 2 fixed point start from different IC; same time of different trajectories.
0.13503964245319366
```

## Eigenvalue decomposition of Jacobian around fixed points.

```bash
$ python linear_approximation.py --activation relu
```

- Distribution of eigenvalues

![relu_eigenvalues](https://user-images.githubusercontent.com/24406002/71605806-48ddfb80-2baf-11ea-8f33-62a9c10355eb.png)


- There is the correlation with the frequencies of trajectories and the values of 
the imaginary part of the maximum eigenvalue of Jacobians.

![freq_relu](https://user-images.githubusercontent.com/24406002/71605816-54c9bd80-2baf-11ea-8310-fd92b3aff1eb.png)
