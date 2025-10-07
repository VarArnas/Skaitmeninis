import matplotlib.pyplot as plt

def train_val_graph(train_loss, train_acc, val_loss, val_acc):
    plt.figure(figsize=(15, 5))
    
    # Draw training and validation loss based on loss hist
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, 'purple', label='Training')
    plt.plot(val_loss, 'orange', label='Validation')
    plt.title('Training/validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss function avg')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # Draw training and validation accuracy based on acc hist
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, 'blue', label='Training')
    plt.plot(val_acc, 'red', label='Validation')
    plt.title('Training/validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy avg')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def sgd_batch_graph(sgd_loss, sgd_acc, batch_loss, batch_acc):
    plt.figure(figsize=(15, 5))
    
    # Draw SGD and BGD loss based on their hist
    plt.subplot(1, 2, 1)
    plt.plot(sgd_loss, 'purple', label='Stochastic')
    plt.plot(batch_loss, 'orange', label='Batch')
    plt.title('SGN/Batch validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss function avg')
    plt.legend()
    plt.ylim(0, 2.2)
    plt.grid(True, alpha=0.5)
    
    # Draw SGD and BGD accuracy based on their hist
    plt.subplot(1, 2, 2)
    plt.plot(sgd_acc, 'blue', label='Stochastic')
    plt.plot(batch_acc, 'red', label='Batch')
    plt.title('SGN/Batch validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy avg')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.show()