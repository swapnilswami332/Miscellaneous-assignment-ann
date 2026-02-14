# Miscellaneous-assignment-ann
ANN assignment 


Artificial Neural Network (ANN) – Experiments & Documentation

1.	Evolution of ANN from Perceptron (with derivation)

Perceptron (Single-layer)

Developed by Frank Rosenblatt (1958).

Model:

Y = f(w^Tx + b)

Where

 = input vector

 = weights

 = bias

 = step activation


Learning rule derivation: Error:

E = y_{true} – y_{pred}

Weight update:

W_{new} = w + \eta e x

Bias update:

B_{new} = b + \eta e

Limitation:
Cannot solve non-linear problems (e.g., XOR).



Multilayer Perceptron (MLP)

Introduced to solve non-linear separability.

Forward pass:

A^{(l)} = f(W^{(l)} a^{(l-1)} + b^{(l)})

Loss:

L = \frac{1}{2}(y-\hat y)^2

Backpropagation:

\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat y}
\cdot \frac{\partial \hat y}{\partial z}
\cdot \frac{\partial z}{\partial w}

Weight update:

W = w - \eta \frac{\partial L}{\partial w}



2.	Problems in Training ANN & Solutions

Problem	Cause	Solution	Derivation idea

Vanishing gradient	Sigmoid/tanh saturation	ReLU	gradient → 0 when slope small
Exploding gradient	Large weights	Gradient clipping	
Overfitting	Too many parameters	Dropout/L2	add penalty to loss
Slow convergence	Poor initialization	Xavier/He init	variance preservation
Local minima	Non-convex loss	Adam optimizer	momentum updates




3.	Activation Functions Comparison

Function	Formula	Range	Pros	Cons

Sigmoid		0–1	smooth	vanishing gradient
Tanh		-1–1	zero centered	still vanishing
ReLU		0–∞	fast	dying ReLU
Leaky ReLU	 if x>0 else 0.01x	-∞–∞	avoids dead neurons	small negative slope
Softmax	exp(x)/sum(exp(x))	probability	multi-class	expensive




4.	Weight Initialization Techniques

Method	Formula	Use

Random	small random values	basic
Xavier		tanh/sigmoid
He		ReLU
Zero init	all zeros	❌ not used


Reason: maintain variance across layers.



5.	Batch Normalization Results

Before BN

Slow training

Internal covariate shift

Unstable gradients


After BN

Faster convergence

Higher accuracy

Stable training


Formula:

\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}

Y = \gamma \hat x + \beta 



6.	Transfer Learning Results

Using pretrained model (e.g., ImageNet weights)

Metric	Before	After

Training time	high	low
Accuracy	medium	high
Data needed	large	small




7.	Early Stopping & Checkpointing

Early stopping condition: Stop when validation loss increases:

L_{val}^{t} > L_{val}^{t-1}

Checkpointing:
Save best weights:

If val_loss improves → save model

Benefits:

Prevents overfitting

Saves best model




8.	Optimizers Comparison (with derivation)

Gradient Descent

W = w - \eta \nabla L

Momentum

V = \beta v + \eta \nabla L

W = w – v 

RMSProp

S = \beta s + (1-\beta)g^2

W = w - \eta \frac{g}{\sqrt{s}} 

Adam

M = \beta_1 m + (1-\beta_1)g

V = \beta_2 v + (1-\beta_2)g^2 

W = w - \eta \frac{m}{\sqrt{v}}



9.	Loss Functions

Loss	Formula	Use

MSE		regression
Binary cross entropy		binary
Categorical cross entropy		multi-class
Hinge loss	max(0,1-yx)	SVM




10.	Regularization Techniques

L1

L = loss + \lambda |w|

L2

L = loss + \lambda w^2

Dropout

Random neuron removal:

Y = w(x \cdot mask)

Results

Technique	Before	After

Accuracy	overfit	better generalization
Loss	low train, high val	balanced




Conclusion

ANN evolved from simple perceptron to deep networks using:

Backpropagation

Advanced optimizers

Normalization

Regularization

Transfer learning


These techniques improve:

Accuracy

Training speed

Generalization








