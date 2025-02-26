# GLocal-K with Temporal Information and Factorization

This repository contains code which enhancing the original model introduced in [GLocal-K: Global and Local Kernels for Recommender Systems](https://arxiv.org/pdf/2108.12184.pdf).

### <div align="center"> Yoav Zelinger, Yehonatan Kidushim (2025, February). <br> GLocal-K with Temporal Information and Factorization <br> Enhancing Kernel-Based Recommendation with Matrix Factorization and Temporal Modeling </div>

![Enhanced_GLocal_K_overview](https://github.com/[yoavzelinger]/[Glocal_K]/blob/[main]/new_pipeline.png?raw=true)

## 1. Introduction
This repository provides an enhanced implementation of **GLocal-K**, a matrix completion framework based on global and local kernels, with additional matrix factorization and temporal modeling components.  

GLocal-K operates in two main stages:  
1. **Pre-training** an autoencoder using a local kernelized weight matrix.  
2. **Fine-tuning** the pre-trained autoencoder with a rating matrix produced by a global convolutional kernel.  

### Enhancements in This Implementation  
This version extends the original GLocal-K with the following improvements:  
- **Matrix Factorization (MF) Integration** – Incorporated at the pre-training stage to better capture linear dependencies and improve rating reconstruction.  
- **Session-Based Bias Modeling** – Dynamically adjusts ratings based on short-term user behavior within sessions.  
- **Time-Decay Function** – Ensures recent interactions have a stronger influence on predictions while preserving long-term trends.  

### Supported Datasets  
The implementation supports evaluation on the following benchmark datasets:  
- [x] **MovieLens-100K (ML-100K)**  
- [x] **MovieLens-1M (ML-1M)**  
- [x] **Douban** (Original GLocal-K only, due to missing timestamps)

## 2. Setup
Download this repository. As the code format is .ipynb, there are no settings but the Jupyter notebook with GPU.

## 3. Requirements
* numpy
* scipy
* tensorflow (converted to version 1.x automatically in the main code, for the Matrix Factorization code use 2.x)

## 4. Run
1. Insert the path of a data directory on the main code by yourself (e.g., '/content/.../data').
2. Write down a dataset correctly among 'ML-1M', 'ML-100K', and 'Douban' (Douban only for the original model) on the main code. If you want to run the Matrix Factorization, enter a latent dimension for the matrices either.
3. There are no other things to do anymore, just try running the code and see it.

## Results  
We evaluate our enhancements on the ML-100K and ML-1M datasets using **Root Mean Squared Error (RMSE)**.  

### Overall Performance  
The hybrid **GLocal-K + MF** model improves upon the baseline GLocal-K, particularly in **denser datasets** like ML-100K, where it achieves lower RMSE with fewer latent dimensions.  

| Model                         | ML-100K RMSE | ML-1M RMSE |  
|-------------------------------|--------------|------------|  
| **GLocal-K (Baseline)**       | 1.0457       | 0.8951     |  
| **Matrix Factorization (MF)** | 0.9692       | **0.8916** |  
| **GLocal-K + MF (Ours)**      | **0.9691**   | 0.8943     |  

### Effect of Latent Dimensions  

#### ML-100K  
| Latent Dimensions (𝑑) | MF RMSE     | GLocal-K + MF RMSE |  
|-----------------------|-------------|--------------------|  
| 25                    | 0.9772      | 0.9984             |  
| 50                    | 0.9810      | 0.9739             |  
| 75                    | 0.9728      | **0.9691**         |  
| 100                   | **0.9692**  | 0.9713             |  

#### ML-1M  
| Latent Dimensions (𝑑) | MF RMSE     | GLocal-K + MF RMSE  |  
|-----------------------|-------------|---------------------|  
| 25                    | 0.8927      | 0.8979              |  
| 50                    | 0.8942      | **0.8943**          |  
| 75                    | 0.8922      | 0.8954              |  
| 100                   | **0.8916**  | 0.8954              |  

## 6. Data References
1. Harper, F. M., & Konstan, J. A. (2015). The movielens datasets: History and context. *Acm transactions on interactive intelligent systems (tiis)*, 5(4), 1-19.
2. Monti, F., Bronstein, M. M., & Bresson, X. (2017, December). Geometric matrix completion with recurrent multi-graph neural networks. In *Proceedings of the 31st International Conference on Neural Information Processing Systems* (pp. 3700-3710).
