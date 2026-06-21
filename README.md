# Deep Learning Applications: Image Classification and Character-Level Language Modeling

## Overview

This project contains two deep learning applications implemented using TensorFlow:

1. Food image classification using a Convolutional Neural Network (CNN)
2. Character-level language modeling and text generation using Recurrent Neural Networks (RNN)

The goal of this project was to gain hands-on experience building deep learning models from scratch, implementing complete training pipelines, and applying neural networks to both computer vision and natural language processing tasks.

---

# Part 1 — Food Image Classification (Food11)

## Objective

Train a Convolutional Neural Network (CNN) to classify food images into 11 categories from the Food11 dataset.

## Dataset

The Food11 dataset contains images from the following categories:

- Bread
- Dairy Product
- Dessert
- Egg
- Fried Food
- Meat
- Noodles / Pasta
- Rice
- Seafood
- Soup
- Vegetable / Fruit

Images were resized to **128×128 RGB** before training.

---

## Model Architecture

```text
Input (128×128×3)

Conv(32)
→ ReLU
→ BatchNorm
→ MaxPool

Conv(64)
→ ReLU
→ BatchNorm

Conv(64)
→ ReLU
→ BatchNorm
→ MaxPool

Conv(128)
→ ReLU
→ BatchNorm

Conv(128)
→ ReLU
→ BatchNorm
→ MaxPool

Flatten

Dense(1024)
→ Sigmoid
→ BatchNorm
→ Dropout

Dense(11)
→ Softmax
```

---

## Techniques Used

- Convolutional Neural Networks (CNN)
- Batch Normalization
- Dropout Regularization
- Adam Optimizer
- TensorBoard Visualization
- Model Checkpoint Saving
- Mini-batch Training
- Multi-class Classification

---

## Training Pipeline

- Custom data loading and preprocessing
- Mini-batch training
- Validation monitoring
- Automatic checkpoint saving
- TensorBoard histogram visualization
- Accuracy and loss tracking

---

## Outcome

The model successfully learned visual features for food recognition and demonstrated stable convergence during training. This project provided practical experience in designing CNN architectures and training image classification models using TensorFlow.

---

# Part 2 — Character-Level Language Modeling

## Objective

Build a character-level language model capable of learning writing patterns from Shakespeare's works and generating new text sequences.

## Dataset

Training data consists of Shakespeare text corpus.

The model learns to predict the next character given a sequence of previous characters.

---

## Model Architecture

```text
Character Input
↓
Embedding Layer
↓
Multi-layer RNN
↓
Dropout
↓
Fully Connected Layer
↓
Softmax
↓
Next Character Prediction
```

### Network Configuration

- Embedding Layer
- 2-layer Recurrent Neural Network
- Hidden Size: 128
- Sequence Length: 50
- Batch Size: 50
- Adam Optimizer
- Gradient Clipping

---

## Techniques Used

- Character-Level Language Modeling
- Sequence Prediction
- Word/Character Embedding
- Multi-layer RNN
- Gradient Clipping
- Learning Rate Decay
- Checkpoint Recovery
- Text Sampling and Generation
- TensorBoard Logging

---

## Training Pipeline

1. Preprocess raw text into character sequences
2. Convert characters into indexed vocabulary representations
3. Train a multi-layer RNN using teacher forcing
4. Evaluate validation loss periodically
5. Save checkpoints during training
6. Generate text samples from the trained model

---

## Example Capabilities

After training, the model can generate Shakespeare-style text by recursively predicting the next character in a sequence.

Example workflow:

```text
Input:
"To be, or not to be"

Model predicts:
"To be, or not to be that is the..."
```

---

## Outcome

This project provided practical experience with:

- Sequential data modeling
- Recurrent Neural Networks
- Language Modeling
- Text Generation
- Training stateful neural networks
- Managing long-term dependencies in sequence data

---

# Skills Demonstrated

## Deep Learning

- Neural Network Fundamentals
- Forward and Backpropagation
- Optimization Techniques
- Hyperparameter Tuning
- Model Evaluation

## Computer Vision

- CNN Architecture Design
- Image Classification
- Feature Extraction
- Data Preprocessing

## Natural Language Processing

- Language Modeling
- Sequence Learning
- Character Embedding
- Text Generation

## Software Engineering

- TensorFlow 1.x Development
- Training Pipeline Implementation
- Checkpoint Management
- TensorBoard Visualization
- Experiment Tracking

---

# Technologies

- Python
- TensorFlow 1.x
- NumPy
- OpenCV
- TensorBoard

---

# What I Learned

Through this project, I gained hands-on experience building deep learning systems from scratch for both computer vision and natural language processing tasks. The work involved designing neural network architectures, implementing training workflows, debugging optimization issues, and evaluating model performance on real-world datasets.

These projects strengthened my understanding of how deep learning techniques can be applied across different domains, from image recognition to sequence generation.
