# 🧠 Self-Pruning Neural Network (AI Engineering Case Study)

## 📌 Overview

This project implements a **self-pruning neural network** that dynamically learns which weights to remove during training.
Unlike traditional pruning (post-training), this approach integrates pruning directly into the training process using **learnable gating mechanisms**.

The goal is to achieve:

* High model sparsity (efficient network)
* Competitive classification accuracy
* A clear sparsity–accuracy trade-off

---

## ⚙️ Methodology

### 🔹 Prunable Linear Layer

A custom linear layer was implemented where each weight is associated with a learnable **gate parameter**.

* Gate values are computed using a sigmoid function
* A **Straight-Through Estimator (STE)** is used to approximate hard pruning:

  * Forward pass uses binary gates (0 or 1)
  * Backward pass allows gradient flow

This enables the network to **effectively prune connections during training**.

---

### 🔹 Sparsity Regularization

The loss function is defined as:

Total Loss = Classification Loss + λ × Sparsity Loss

* **Classification Loss**: Cross-Entropy Loss
* **Sparsity Loss**: Mean of gate activations across layers
* λ controls the trade-off between accuracy and sparsity

---

### 🔹 Model Architecture

* Fully Connected Network (MLP)
* Input: CIFAR-10 images (flattened)
* Layers:

  * PrunableLinear (3072 → 512) + BatchNorm + ReLU
  * PrunableLinear (512 → 256) + BatchNorm + ReLU
  * PrunableLinear (256 → 10)

---

### 🔹 Training Improvements

To improve performance:

* **Batch Normalization** → stabilizes training
* **Data Augmentation** → improves generalization
* **Learning Rate Scheduler** → better convergence

---

## 📊 Results

### 🔹 Experiment 1: Strong Pruning Behavior

| Lambda | Test Accuracy | Sparsity |
| ------ | ------------- | -------- |
| 0.001  | 50.12%        | 53.43%   |
| 0.01   | 51.24%        | 64.32%   |
| 0.1    | 50.52%        | 79.59%   |

**Observation:**

* Increasing λ significantly increases sparsity
* Demonstrates a clear **sparsity–accuracy trade-off**

---

### 🔹 Experiment 2: Improved Accuracy

| Lambda | Test Accuracy | Sparsity |
| ------ | ------------- | -------- |
| 0.001  | 55.84%        | 51.71%   |
| 0.01   | 56.46%        | 52.10%   |
| 0.1    | 55.67%        | 53.83%   |

**Observation:**

* Architectural improvements improved accuracy
* Sparsity became more stable across λ values

---

## 📈 Key Insights

* L1-style regularization on gates encourages sparsity by pushing values toward zero
* Straight-Through Estimation enables **hard pruning with gradient flow**
* There exists a trade-off:

  * Higher λ → more pruning → possible accuracy drop
* Improved architectures can **reduce sparsity sensitivity but increase performance**

---

## 🧠 Conclusion

This project demonstrates a practical implementation of **self-pruning neural networks** with:

* Dynamic weight pruning during training
* Effective sparsity control
* Competitive performance on CIFAR-10

It highlights the balance between:

> Model efficiency (sparsity) and predictive performance (accuracy)

---

## 🚀 How to Run

1. Open the notebook in Google Colab
2. Enable GPU (optional but recommended)
3. Run all cells sequentially

---

## 📂 Project Structure

```
├── train.ipynb        # Main training notebook
├── README.md          # Project report

```

---

## 📌 Future Improvements

* Use convolutional architectures (CNNs) instead of MLP
* Explore structured pruning (neurons instead of weights)
* Apply advanced techniques like Hard-Concrete distributions
* Evaluate inference speed improvements after pruning

---

## 🙌 Acknowledgment

This project was developed as part of an AI Engineering case study to demonstrate:

* Deep learning fundamentals
* Custom model design
* Optimization and experimentation skills

---
