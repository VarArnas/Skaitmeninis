import numpy as np

def sigmoid_activation(a):
    # Sigmoid activation function - maps any input to (0,1) range.
    return 1 / (1 + np.exp(-a))

def perform_inference(data_rows, weights):
    # Perform inference using trained weights on given data.

    avg_loss = 0.0
    avg_acc = 0
    results = []

    for row in data_rows:
        features = [1] + row[:-1]  # Add bias term
        ground_truth = row[-1]
                
        # Forward pass through the model
        a = sum(x * w for x, w in zip(features, weights))
        y = sigmoid_activation(a)
        y = np.clip(y, 1e-8, 1 - 1e-8)  # Prevent log(0)

        prediction = 1 if y >= 0.5 else 0

        # Calculate loss and accuracy
        avg_loss += -np.log(y) if ground_truth == 1 else -np.log(1 - y)
        avg_acc += 1 if prediction == ground_truth else 0

        # Store results: original features + true label + prediction
        results.append(row[:-1] + [ground_truth, prediction])

    # Calculate averages
    avg_loss /= len(data_rows)
    avg_acc /= len(data_rows)
    return avg_loss, avg_acc, results