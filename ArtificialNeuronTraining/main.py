import numpy as np
from RetrieveDatasets import get_dataframes
from Training import sgd, batch
from Inference import perform_inference
from Graphs import train_val_graph, sgd_batch_graph
from CreateExcel import create_excel

LR_SGD = 0.01
LR_BATCH = 0.5
EPOCHS = 1000
TARGET_LOSS = 0.01 # set to 0.01 so that would train until 1000 epochs, if want to train until target loss is reached, change to 0.1


train_df, val_df, test_df = get_dataframes()

sgd_weights, sgd_epochs, sgd_time, sgd_train_loss, sgd_train_acc, sgd_val_loss, sgd_val_acc = sgd(train_df, LR_SGD, EPOCHS, TARGET_LOSS, val_df)
sgd_test_loss, sgd_test_acc, test_results = perform_inference(test_df, sgd_weights)
create_excel(len(test_df[1]) - 1, 'sgd_results.xlsx', test_results)

batch_weights, batch_epochs, batch_time, batch_train_loss, batch_train_acc, batch_val_loss, batch_val_acc = batch(train_df, LR_BATCH, EPOCHS, TARGET_LOSS, val_df)
test_batch_err, test_batch_acc, test_results = perform_inference(test_df, batch_weights)
create_excel(len(test_df[1]) - 1, 'batch_results.xlsx', test_results)

print("\nSGD")
print(f"Weights: {np.round(sgd_weights[1:], 4)}, Bias: {sgd_weights[0]:.4f}")
print(f"Train loss: {sgd_train_loss[-1]:.4f}, Train accuracy: {sgd_train_acc[-1]:.4f}")
print(f"Val loss: {sgd_val_loss[-1]:.4f}, Val accuracy: {sgd_val_acc[-1]:.4f}")
print(f"Test loss: {sgd_test_loss:.4f}, Test accuracy: {sgd_test_acc:.4f}")
print(f"Epochs: {sgd_epochs}, Total time (in seconds): {sgd_time:.4f}")

print("\nBATCH")
print(f"Weights: {np.round(batch_weights[1:], 4)}, Bias: {batch_weights[0]:.4f}")
print(f"Train loss: {batch_train_loss[-1]:.4f}, Train accuracy: {batch_train_acc[-1]:.4f}")
print(f"Val loss: {batch_val_loss[-1]:.4f}, Val accuracy: {batch_val_acc[-1]:.4f}")
print(f"Test loss: {test_batch_err:.4f}, Test accuracy: {test_batch_acc:.4f}")
print(f"Epochs: {batch_epochs}, Total time (in seconds): {batch_time:.4f}")

train_val_graph(sgd_train_loss, sgd_train_acc, sgd_val_loss, sgd_val_acc)
train_val_graph(batch_train_loss, batch_train_acc, batch_val_loss, batch_val_acc)
sgd_batch_graph(sgd_val_loss, sgd_val_acc, batch_val_loss, batch_val_acc) 