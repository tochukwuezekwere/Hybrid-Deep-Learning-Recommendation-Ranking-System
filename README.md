# Hybrid Deep Learning Recommendation System

## Overview
This project is an end-to-end machine learning recommendation system that predicts and ranks personalized movie recommendations using collaborative filtering and deep learning techniques built in PyTorch.

The system simulates real-world recommendation pipelines similar to those used in platforms like Netflix, YouTube, and Amazon.

---

## Problem Statement
Traditional recommendation systems struggle with sparsity, cold-start issues, and nonlinear user-item interactions.

This project addresses these challenges by:
- Learning latent user and item representations (embeddings)
- Modeling nonlinear interactions using neural networks
- Generating ranked top-K recommendations

---

## Dataset
- MovieLens dataset (ml-latest-small)
- Provided by GroupLens Research
- Contains:
  - user ratings
  - movie metadata
  - movie genres

---

## Tech Stack
- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- FastAPI (for inference API)
- Google Colab

---

## System Architecture

1. Data Preprocessing
   - Encode users and movies
   - Clean and structure interaction data

2. Model Training
   - Neural Collaborative Filtering (NCF)
   - Embedding layers for users and items
   - Multi-layer perceptron for rating prediction

3. Recommendation Engine
   - Predict scores for unseen items
   - Rank items by predicted relevance
   - Return Top-K recommendations

---

## Model Architecture
- User Embedding Layer
- Movie Embedding Layer
- Concatenation Layer
- Fully Connected Neural Network
- Output: Predicted rating

---

## Training Process
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Framework: PyTorch
- GPU accelerated training (Google Colab)

---

## Evaluation Metrics
- Mean Squared Error (MSE)
- Precision@K
- Recall@K

---

## Key Features
- Neural Collaborative Filtering model
- Embedding-based representation learning
- Top-K recommendation system
- Cold-start awareness (via metadata exploration)
- Inference pipeline for real-time recommendations

---

## How to Run

1. Clone the repository:
