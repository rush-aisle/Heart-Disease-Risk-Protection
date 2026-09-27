# Heart Disease Risk Prediction: Neural Network Built From Scratch

A fully connected deep neural network implemented **entirely from scratch using NumPy**, with no TensorFlow, PyTorch, or Keras, applied to real-world public health survey data to detect heart disease risk.

## Why this project

I wanted to understand the theory and math of neural networks rather than js colling `model.fit()` in a framework. It uses forward propagation, backpropagation, gradient descent, and a full object-oriented layer/model architecture manually. Then, I applied it to a public health dataset to conduct binary classification.

## The dataset

[Heart Disease Health Indicators (BRFSS 2015)](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset): 253,680 survey responses from the CDC's Behavioral Risk Factor Surveillance System, with 21 lifestyle and health features (BMI, blood pressure, physical activity, smoking status, etc.) predicting whether a respondent has had heart disease or a heart attack.

This is an important problem in society to identify individuals at-risk from survey data. 

## Architecture

```
Input (21 features)
   → Dense(16, ReLU)
   → Dense(8, ReLU)
   → Dense(1, Sigmoid)
```

- **He initialization** on all weights, chosen to keep activation variance stable through ReLU layers.
- **Binary cross-entropy loss**, optimized via full-batch gradient descent.
- Every component — activation functions and their derivatives, forward propagation, backpropagation, parameter updates — is implemented manually in `dnn_framework.py`.

## The core challenge: class imbalance

Only **9.4%** of respondents in this dataset report heart disease or a heart attack.

**First attempt (unweighted loss):** the model reached 90.6% test accuracy. This initially seemed like a really accurate model, until the confusion matrix produced:

```
[[45968     0]
 [ 4768     0]]
```

**Zero true positives.** Even though this model is correct ~90% of the time, the model just decided to predict that everyone has "no heart disease." This provides zero usefulness, since there is no positive diagnosis.

**Fix — weighted binary cross-entropy loss:** I modified the deep neural network in order to predict some positive cases. I did this by changing the loss function and its gradient to weight positive-class errors by the inverse class ratio (~9.6x). This ratio was computed from the training data. Thus, the model will have to learn how to distinguish the smaller class rather than just ignoring it.

**Result after reweighting:**

```
[[31757 14211]
 [  959  3809]]
```

| Metric | Unweighted | Weighted |
|---|---|---|
| Test accuracy | 90.6% | 70.1% |
| Recall (class 1) | 0% | **80%** |
| Precision (class 1) | 0% | 21% |

The overall accuracy *dropped*, but now the model is able to identify **80% of actual heart disease cases** instead of none. This is a positive trade-off in the health-screening context, since a false negative is way costlier than a false alarm. This highlights that precision is much better than raw accuracy in analyzing medical data. 

## What this project demonstrates

- Manual implementation of forward/backward propagation and gradient descent, instead of using a pre-built framework
- Understanding of weight initialization strategy (He initialization) and why it matters for ReLU networks
- Recognizing a misleading evaluation metric (accuracy) on imbalanced data, figuring out the cause, and implementing a solution (weighted loss)
- End-to-end ML workflow: data preprocessing, train/test methodology without leakage, model training, and honest evaluation

## Project structure

```
DeepNeuralNetwork.ipynb   # Full notebook: from-scratch framework, data pipeline, training, evaluation
heart_disease.csv         # Heart Disease Health Indicators dataset (CSV)
```

## Running it

```bash
pip install numpy pandas scikit-learn matplotlib
```

Open `DeepNeuralNetwork.ipynb` and run all cells. Data is loaded from `heart_disease.csv`, which is provided in the repository.

## Possible extensions

- Sweep `pos_weight` values and plot a full precision-recall trade-off curve
- Add L2 regularization to reduce the false-positive rate
- Compare against a logistic regression baseline to quantify what the extra network depth actually buys
- Try SMOTE (synthetic minority oversampling) as an alternative to loss weighting
