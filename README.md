## 🤖 Unsupervised-learning
Note:  This work summarizes my learnings in the domain of unsupervised learning as part of the **"EECS 545: Machine Learning"** course, conducted from January to April 2023. Here I provide implementations of topics such as Image compression, Face image dimension reduction, Audio separation, and Handwritten digits generation.

<p align="center">
  <img src="assets/gmm_gif.gif" alt="Simulation 1" width="400" />
  <img src="assets/kmeans.png" alt="Simulation 2" width="400" />
  <img src="assets/eigenfaces.png" alt="Simulation 3" width="400" />
  <img src="assets/mnist_data_generation.png" alt="Simulation 4" width="400" />
</p>

### 🎯 Goal
The goal of this project is to provide a useful resource for anyone seeking to understand and implement some of the fundamental unsupervised learning algorithms. For a brief overview, this repo contains the following implementations:
- K-means based image compression
- Gaussian Mixture Model with Expectation Maximization based image compression
- Principal component analysis for Eigenfaces generation
- Independent component analysis for Audio separation
- Conditional variational autoencoder based MNIST data generation

## 🛠️ Test/Demo
- Image compression
    - Launch the 'kmeans_gmm.ipynb 'jupyter notebook.
- Eigenfaces generation
    - Launch the 'pca.ipynb' jupyter notebook.
- Audio separation
    - Launch the 'ica.ipynb' jupyter notebook.
- MNIST data generation
    - Launch the 'cvae.ipynb' jupyter notebook.

## 📊 Results
### 📈 Image compression
- K-means
<p align="center">
  <img src="assets/kmeans_gif.gif" alt="Simulation 1" width="400" />
  <img src="assets/kmeans.png" alt="Simulation 2" width="400" />
</p>

- Gaussian mixture model with EM
<p align="center">
  <img src="assets/gmm_gif.gif" alt="Simulation 1" width="400" />
  <img src="assets/gmm.png" alt="Simulation 2" width="400" />
</p>


### 📈 Eigenfaces generation
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/eigenfaces.png">
</div>


### 📈 Audio separation
- Mixed audio
[Listen](https://drive.google.com/file/d/1gOYO4-QU3ReCSVI3TinCQFmMjONq-bD8/view?usp=sharing)

- Unmixed source 0
[Listen](https://drive.google.com/file/d/1xQ4vs7CvwScQ5QS4LHhoJQanjK0OMaB5/view?usp=sharing)

- Unmixed source 1
[Listen](https://drive.google.com/file/d/1lfzsv0j28lr2PC86Lgj3iBY58azmYXwa/view?usp=sharing)

- Unmixed source 2
[Listen](https://drive.google.com/file/d/1a3zTCa5YXunwjSQldRtemogKCgY-JJVt/view?usp=sharing)

- Unmixed source 3
[Listen](https://drive.google.com/file/d/1-9npjq_X4xtR0RQfvYCXTo7lqB6W-kvR/view?usp=sharing)

- Unmixed source 4
[Listen](https://drive.google.com/file/d/1hKuIJq2vw9h3Nt_6lVXO1wabILtcbdae/view?usp=sharing)

### 📈 MNIST Data generation
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/mnist_data_generation.png">
</div>