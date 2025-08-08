Title: 🖼️ Image Classification using Custom CNN, VGG16 & ResNet with Streamlit Deployment

Description:
This project implements an Image Classification System using three deep learning approaches:

Custom CNN (trained from scratch)

VGG16 (transfer learning)

ResNet (transfer learning)

The trained models were deployed with an interactive Streamlit web application, allowing users to upload images and get instant classification results.

Key Features:

Dataset: CIFAR-10 (with selected classes: airplane, automobile, bird, cat, deer)

Preprocessing: Image resizing, normalization, and augmentation

Models:

Custom Convolutional Neural Network (CNN)

VGG16 pretrained model (fine-tuned)

ResNet pretrained model (fine-tuned)

Evaluation Metrics: Training accuracy per epoch, final accuracy comparison between models

Deployment: Streamlit app for real-time predictions

Storage: Models saved in .pth format for easy reusability

Technologies Used:

Python

PyTorch

Torchvision

NumPy, Pandas

PIL (Python Imaging Library)

Streamlit
