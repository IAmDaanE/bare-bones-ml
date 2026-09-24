# Custom Neural Network and Machine Learning Library

My own homemade neural network and machine learning library written in python using just NumPy.

---

<img width="1199" height="774" alt="image" src="https://github.com/user-attachments/assets/35d24991-2286-46e1-a87f-78278507bdae" />

*the visualization window*

## About the Project

It contains Loss functions, Activation functions, Learning Rate decay functions and multiple ways to initialize weights when creating the network. It also has a nice visualization function made with pygame where you can see the neural network and its weights changing as training progresses.

## Getting Started

### Installation

**Requires:**: Python 3.9 - 3.13  
Run this command (preferably using a virtual environment):

```
pip install git+https://github.com/IAmDaanE/bare-bones-ml.git@master
```

### Usage

Importing
```python
import barebones_ml as bbml
```
Creating a neural network
```python
network = bbml.Network(bbml.losses.mse)
network.add(bbml.Layer(1, 16, bbml.Activations.ReLu))
```
Doing a forward pass
```python
outputs = network.forward(inputs)
```
Doing a backward pass
```Python
network.backward(network.loss)
```

## License

This project is open-source and available under the MIT License.
