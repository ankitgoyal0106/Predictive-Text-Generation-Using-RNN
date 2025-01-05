# Predictive-Text-Generation-Using-RNN

This project involves building a character-level text autocomplete model using a Recurrent Neural Network (RNN) implemented in PyTorch. The model is trained on datasets ranging from simple repetitive sequences to complex text (*War and Peace*). The project demonstrates the ability of RNNs to model sequential data and generate text predictions, simulating autocomplete functionality.

---

## Objectives

1. Understand the fundamentals of RNNs and their application in sequence modeling.
2. Implement an RNN from scratch using PyTorch.
3. Train the model on datasets of varying complexity for text generation tasks.
4. Optimize hyperparameters to improve training efficiency and model performance.
5. Explore temperature scaling to balance coherence and creativity in text predictions.

---

## Features

- **Text Autocomplete**: The model predicts and completes a sequence of characters based on input.
- **Character-Level RNN**: Processes data at the character level for fine-grained text generation.
- **Temperature Scaling**: Adjusts randomness in predictions to control diversity and coherence.
- **Hyperparameter Optimization**: Includes tuning of embedding dimensions, hidden state size, learning rate, and batch size.
- **Training and Testing Pipelines**: End-to-end processing with clear training and evaluation loops.

---

## Datasets

1. **Simple Sequence Dataset**:
   - Input: Repeated alphabet sequence ("abcdefghijklmnopqrstuvwxyz" * 100).
   - Purpose: To verify the basic functionality of the RNN.
   - Training Loss: ~0.027 after 5 epochs.

2. **Complex Dataset (*War and Peace*)**:
   - Input: Text from *War and Peace*.
   - Purpose: To test the RNN's ability to model complex dependencies in text.
   - Final Loss: 1.35; Accuracy: 54.87% after 3 epochs.

---

## Project Workflow

### 1. Data Preprocessing
- Convert text to lowercase and retain only valid characters (letters and punctuation).
- Map characters to numerical indices and vice versa.

### 2. Dataset Creation
- Custom `CharDataset` class to generate overlapping input-target pairs for training.

### 3. Model Architecture
- **Embedding Layer**: Converts character indices to dense vectors.
- **Recurrent Layer**: Captures sequential dependencies using RNN equations.
- **Fully Connected Layer**: Maps hidden states to output vocabulary.

### 4. Training
- Train the model using CrossEntropyLoss and Adam optimizer.
- Monitor accuracy and loss over epochs.

### 5. Text Generation
- Generate text character-by-character using the trained model.
- Incorporate temperature scaling for randomness control.

---

## How to Run

1. **Setup Environment**
   - Install Python 3.x and PyTorch.
   - Install additional dependencies: `tqdm`.

2. **Prepare Data**
   - Place the training data (e.g., *War and Peace*) in the working directory.

3. **Run the Code**
   - Train on the simple sequence: Modify the `sequence` variable.
   - Train on *War and Peace*: Use the provided `read_file` function.

4. **Generate Text**
   - Input a starting sequence and desired length.
   - Adjust the temperature parameter for diversity.

---

## Key Insights

1. **RNN Strengths**:
   - Effectively learns short-term patterns, as shown with the alphabet sequence.

2. **Challenges**:
   - Struggles with long-term dependencies and complex grammar in large texts.

3. **Temperature Tuning**:
   - Lower values (e.g., 0.5) improve coherence.
   - Higher values (e.g., 1.5) encourage creativity but reduce grammatical accuracy.

---

