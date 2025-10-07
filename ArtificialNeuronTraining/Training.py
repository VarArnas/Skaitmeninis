import random
import time
import numpy as np
from Inference import perform_inference

def sigmoid_activation(a):
    # Sigmoid activation function - maps any input to range (0,1)
    return 1 / (1 + np.exp(-a))

def sgd(training_data, lr, epochs, target_loss, validation_data):
    # Stochastic Gradient Descent - updates weights after each training example
    start = time.time()
    # Initialize weights randomly in small range to avoid saturation
    weights = np.random.uniform(-0.1, 0.1, 10)

    # History tracking for plotting/analysis
    avg_loss_hist = []
    acc_hist = []
    val_avg_loss_hist = []
    val_acc_hist = []

    # Initialize loss to ensure while loop starts
    avg_loss = target_loss + 1
    curr_epoch = 0

    # Train until target loss reached or max epochs exceeded
    while (avg_loss > target_loss) and (curr_epoch < epochs):
        # Shuffle data each epoch to prevent cyclical patterns
        random.shuffle(training_data)
        avg_loss = 0.0
        avg_acc = 0
        
        # Process each training example individually
        for row in training_data:
            # Add bias term (1) and extract features, ground truth
            features = [1] + row[:-1]  # [bias, feature1, feature2, ...]
            ground_truth = row[-1]     # target value (0 or 1)
            
            # Calculate weighted sum and apply sigmoid activation
            a = sum(x * w for x, w in zip(features, weights))
            y = sigmoid_activation(a)
            # Clip predictions to avoid log(0) errors
            y = np.clip(y, 1e-8, 1 - 1e-8)

            # Calculate cross-entropy loss and accuracy for this example
            avg_loss += -np.log(y) if ground_truth == 1 else -np.log(1 - y)
            avg_acc += 1 if (y >= 0.5) == ground_truth else 0
            
            # SGD: Update weights immediately after each example
            for i, x in enumerate(features):
                # Gradient descent: w = w - lr * ∂L/∂w
                # For cross-entropy + sigmoid: ∂L/∂w = (y - t) * x
                weights[i] -= lr * (y - ground_truth) * x

        # Calculate average loss and accuracy for this epoch
        avg_loss /= len(training_data)
        avg_acc /= len(training_data)

        # Store training metrics
        avg_loss_hist.append(avg_loss)
        acc_hist.append(avg_acc)

        curr_epoch += 1

        # Evaluate on validation set
        val_loss, val_acc, _ = perform_inference(validation_data, weights)
        val_avg_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        # Progress reporting
        print(f"Epoch {curr_epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    training_length = time.time() - start
    return weights, curr_epoch, training_length, avg_loss_hist, acc_hist, val_avg_loss_hist, val_acc_hist


def batch(training_data, lr, epochs, target_loss, validation_data):
    #Batch Gradient Descent - updates weights once per epoch using all examples
    start = time.time()
    # Initialize weights randomly
    weights = np.random.uniform(-0.1, 0.1, 10)

    # History tracking
    avg_loss_hist = []
    acc_hist = []
    val_avg_loss_hist = []
    val_acc_hist = []

    avg_loss = target_loss + 1
    curr_epoch = 0

    while (avg_loss > target_loss) and (curr_epoch < epochs):
        avg_loss = 0.0
        avg_acc = 0
        # Initialize gradients accumulator for all weights
        gradients = [0.0] * len(weights)

        # First pass: accumulate gradients from all examples
        for row in training_data:
            features = [1] + row[:-1]  # [bias, feature1, feature2, ...]
            ground_truth = row[-1]     # target value

            # Forward pass: prediction
            a = sum(x * w for x, w in zip(features, weights))
            y = sigmoid_activation(a)
            y = np.clip(y, 1e-8, 1 - 1e-8)

            # Calculate loss and accuracy
            avg_loss += -np.log(y) if ground_truth == 1 else -np.log(1 - y)
            avg_acc += 1 if (y >= 0.5) == ground_truth else 0

            # Accumulate gradients (don't update weights yet)
            for i, x in enumerate(features):
                gradients[i] += (y - ground_truth) * x
        
        # Second pass: update all weights using average gradients
        for i in range(len(weights)):
            # Average gradient over all examples and apply learning rate
            weights[i] -= lr * gradients[i] / len(training_data)

        # Calculate epoch averages
        avg_loss /= len(training_data)
        avg_acc /= len(training_data)
        avg_loss_hist.append(avg_loss)
        acc_hist.append(avg_acc)

        curr_epoch += 1
        print(f"Epoch {curr_epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

        # Validation
        val_loss, val_acc, _ = perform_inference(validation_data, weights)
        val_avg_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    training_length = time.time() - start
    return weights, curr_epoch, training_length, avg_loss_hist, acc_hist, val_avg_loss_hist, val_acc_hist