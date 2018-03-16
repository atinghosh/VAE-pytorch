# Variational-Autoencoder
Contains code to learn variational autoencoder model on MNIST dataset using pytorch.





Gaussian loss is given by

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{1}{N}\sum_{i=1}^{N}\left[\frac{1}{L}\sum_{l=1}^{L}\left\{ \frac{1}{2}\sum_{j=1}^{784}\log(\sigma_{ij}^{(l)})^2 + \frac{1}{2}\sum_{j=1}^{784}\left(\frac{x_{ij}-\mu_{ij}^{(l)}}   {\sigma_{ij}^{(l)}}\right)^2 \right\} \right ]  - \frac{1}{N}\sum_{i=1}^{N}\left[ \sum_{j=1}^{J}\frac{1}{2}\left(1+\log(\sigma_j^{\prime(i)})^2-(\mu_j^{\prime(i)})^2 -(\sigma_j^{\prime(i)})^2\right )\right ]"/>


#<br />
BCE Loss is given by

<img src="https://latex.codecogs.com/svg.latex? \frac{1}{N}\sum_{i=1}^{N}\left[\frac{1}{L}\sum_{l=1}^{L}\left\{x_{ij}\log p_{ij}^{(l)} + (1-x_{ij})\log(1-\log p_{ij}^{(l)}) \right\} \right ]  - \frac{1}{N}\sum_{i=1}^{N}\left[ \sum_{j=1}^{J}\frac{1}{2}\left(1+\log(\sigma_j^{\prime(i)})^2-(\mu_j^{\prime(i)})^2 -(\sigma_j^{\prime(i)})^2\right )\right ]"/>

