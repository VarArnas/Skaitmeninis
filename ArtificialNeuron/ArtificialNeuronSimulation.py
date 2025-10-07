import numpy as np
import matplotlib.pyplot as plt

# Fix random seed for reproducibility
np.random.seed(5)

def GenerateInputs():
    # Generate 10 random 2D points for class 0 in range [0, 10]
    classZero = np.random.uniform(0.0, 10.0, (10, 2))

    # Generate 10 random 2D points for class 1 in range [10, 20]
    classOne = np.random.uniform(10.0, 20.0, (10, 2))

    # Print the generated points for inspection
    for (x0, y0), (x1, y1) in zip(classZero, classOne):
        print(f"Class zero x: {x0}, y: {y0}")
        print(f"Class one x: {x1}, y: {y1}")
    return classZero, classOne


def NeuralNeuron(input, activation, params):
    w1, w2, b = params
    preds = []

    for x, y in input:
        # Compute the linear combination (weighted sum) for each input point
        a = x*w1 + y*w2 + b

        # Classify point using threshold (step) or sigmoid activation
        if activation == 'threshold':
            preds.append(1 if a >= 0 else 0)
        elif activation == 'sigmoid':
            preds.append(1 if sigmoid(a) >= 0.5 else 0)
        else:
            raise ValueError("activation function not valid")

    # Return neuron outputs
    return preds

def sigmoid(logit):
    # Sigmoid activation function maps any value into (0,1)
    return 1 / (1 + np.exp(-logit))

def IsPredictionCorrect(allPoints, groundTruths, activationFunc, params):    
    # Check if all predictions match the ground-truth labels
    preds = NeuralNeuron(allPoints, activationFunc, params)
    return all(p == gt for p, gt in zip(preds, groundTruths))

def DrawPoints(axes, classZero, classOne):
    # Plot points of both classes
    axes.scatter(classZero[:,0], classZero[:,1], color="blue", label="Class 0", s=15)
    axes.scatter(classOne[:,0], classOne[:,1], color="red", label="Class 1", s=15)

    # Add grid, axis lines, and formatting
    axes.grid(True, zorder=0)
    axes.axhline(0, color='black', linewidth=2)
    axes.axvline(0, color='black', linewidth=2)
    axes.set_xlim(-2, 22)
    axes.set_ylim(-2, 22)
    axes.set_aspect('equal') 
    axes.legend(bbox_to_anchor=(-0.2, 1.05), loc='upper left', borderaxespad=0.)
    axes.set_title('Class points')

def DrawBoundries(axes, classZero, classOne, weights):
    # X-axis values for plotting decision boundaries
    xPoints = np.linspace(-2, 22, 100)
    colors = ['yellow', 'purple', 'pink']
    for i, weight in enumerate(weights):
        w1, w2, b = weight

        # If w2 != 0, plot line y = -(w1/w2)*x - (b/w2)
        if w2 != 0:
            yPoints = -(w1 / w2) * xPoints - (b / w2)
            x0 = 10
            y0 = -(w1 /w2) * x0 - (b / w2)
            axes.plot(xPoints, yPoints, color=colors[i], label=f'Decision boundary {i + 1}')

        # If w2 == 0, decision boundary is a vertical line
        else:
            xIntercept = -b / w1
            x0 = xIntercept
            y0 = 10
            axes.axvline(x=xIntercept, color=colors[i], label=f'Decision boundary {i + 1}')

        # Draw the weight vector showing the orientation of the boundary
        axes.quiver(x0, y0, w1, w2, angles='xy', scale_units='xy', scale=0.35, color=colors[i], label=f'Vector {i + 1}')

    # Replot the data points for context
    DrawPoints(axes, classZero, classOne)
    axes.set_title('Decision boundries and weight vectors')

# === Main execution ===

# Generate training data (two classes of points)
classZero, classOne = GenerateInputs()
sigWeights = []
threshWeights = []

# Ground truth labels: 0 for classZero, 1 for classOne
groundTruths = [0] * len(classZero) + [1] * len(classOne)

# Combine all data points into one array
allPoints = np.vstack((classZero, classOne))

# Randomly try weights until we find 3 that perfectly separate the classes
while len(sigWeights) < 3 and len(threshWeights) < 3:
    # Randomly initialize weights (w1, w2) and bias (b) for the neuron
    params = np.random.uniform(-10, 10, 3)

    # Check for perfect classification with threshold activation
    if IsPredictionCorrect(allPoints, groundTruths, 'threshold', params) and len(threshWeights) < 3:
        threshWeights.append(params)

    # Check for perfect classification with sigmoid activation
    if IsPredictionCorrect(allPoints, groundTruths, 'sigmoid', params) and len(sigWeights) < 3:
        sigWeights.append(params)

# Print out the weights and biases that achieved perfect classification
for (w01, w02, b0), (w11, w12, b1) in zip(threshWeights, sigWeights):
    print(f"Threshold values: {w01}, {w02}, {b0}")
    print(f"Sigmoid values: {w11}, {w12}, {b1}")

# Draw results: points and decision boundaries found
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
DrawPoints(axs[0], classZero, classOne)
DrawBoundries(axs[1], classZero, classOne, threshWeights)
plt.show()