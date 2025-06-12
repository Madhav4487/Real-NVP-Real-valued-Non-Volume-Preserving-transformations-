# 🔄 RealNVP for MNIST Digit-Pair Image Generation

This project implements a **Real-valued Non-Volume Preserving (RealNVP)** generative model using PyTorch to learn and generate combined MNIST digit pairs. The model leverages normalizing flows to map between a latent space and image space while maintaining exact likelihood computation.

---

## 📦 Dataset Preparation

- **Base dataset**: [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/)
- **Customization**:
  - Pairs of MNIST digit images are combined **horizontally** into a `28x56` grayscale image
  - Targets are encoded as **two-digit integers** (e.g., digits 3 and 7 become label `37`)
- **Custom PyTorch Dataset**:
  - A subclass of `torchvision.datasets.MNIST` is used to combine pairs of images into a single one-channel tensor

---

## 🧠 Model Architecture: RealNVP

- RealNVP is a type of **normalizing flow** that allows exact density estimation and efficient sampling
- Key components:
  - **Coupling layers** with scale and translation networks
  - **Alternating binary masks** for invertible transformations
- Implementation based on PyTorch modules with 6 coupling layers and 512 hidden units

---

## 🚀 Training Workflow

1. Flattened images of shape `28x56` → vector of size `1568`
2. Used **Negative Log-Likelihood (NLL)** as the training loss
3. Optimized using **Adam** optimizer
4. Training time: ~2.5 minutes per epoch on GPU
5. **Training loss improved steadily**, showing model convergence

## 🧪 Image Generation

- After training, the model generates new digit-pair images by:
  - Sampling from a **standard normal distribution** in the latent space
  - Using the **inverse RealNVP transformation** to map latent vectors back to the image space
## 🔮 Future Work

1. Add conditional RealNVP to generate specific digit-pair combinations
2. Integrate evaluation metrics (e.g., FID score) to assess image quality
3. Extend to color image datasets (e.g., CIFAR-10)
4. Compare performance with Variational Autoencoders (VAEs) and GANs

## 📌 Key Learnings

1. How to implement a normalizing flow model from scratch
2. Dataset manipulation for creative input designs (image pair concatenation)
3. Latent space sampling and invertible neural transformations
4. Training and visualizing generative models using PyTorch

## 📬 Contact

**Madhav Dahal**  
📧 madhavdahal16@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/madhav-dahal-ms-9a1147b0)  
🔗 [GitHub](https://github.com/Madhav4487)
