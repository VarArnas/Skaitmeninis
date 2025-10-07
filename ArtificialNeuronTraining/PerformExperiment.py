import matplotlib.pyplot as plt
from RetrieveDatasets import get_dataframes
from Training import sgd, batch
from dataclasses import dataclass

EPOCHS = 1000
TARGET_LOSS = 0.1
LEARNING_RATES = [0.001, 0.01, 0.1, 0.5, 1]

@dataclass
class RunResult:
    lr: float
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    histories: tuple 
    time: float
    epochs: int

def experiment(train_data, val_data, lrs, is_stochastic=True):
    # Container for experiment results
    results = []
    
    for lr in lrs:
        # Choose gradient descent type
        if is_stochastic:
            _, epoch_num, train_time, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = sgd(train_data, lr, EPOCHS, TARGET_LOSS, val_data)
        else:
            _, epoch_num, train_time, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = batch(train_data, lr, EPOCHS, TARGET_LOSS, val_data)
        
        result = RunResult(
            lr=lr,
            train_loss=train_loss_hist[-1],
            val_loss=val_loss_hist[-1],
            train_acc=train_acc_hist[-1],
            val_acc=val_acc_hist[-1],
            histories=(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist),
            time=train_time,
            epochs=epoch_num
        )
        results.append(result)
    
    return results

def lr_diff_graph(results):
    # Plot comparisons of performance metrics for different learning rates.
    plt.figure(figsize=(20, 5))

    lrs = [r.lr for r in results]
    train_losses = [r.train_loss for r in results]
    val_losses = [r.val_loss for r in results]
    train_accs = [r.train_acc for r in results]
    val_accs = [r.val_acc for r in results]
    
    # Training loss vs epochs for each LR 
    plt.subplot(1, 3, 1)
    for r in results:
        train_loss_hist = r.histories[0]
        plt.plot(train_loss_hist, label=f'LR={r.lr:.4f}', alpha=0.6)

    plt.title('Training loss over epochs with different LRs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss avg')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.5)
    plt.ylim(0, 2.2)
    
    # Final loss vs learning rate
    plt.subplot(1, 3, 2)
    plt.semilogx(lrs, train_losses, 'purple', label='Training loss')
    plt.semilogx(lrs, val_losses, 'orange', label='Validation loss')
    idx = lrs.index(0.5)
    plt.annotate('0.5', (lrs[idx], train_losses[idx]))
    plt.title('Loss/LR')
    plt.xlabel('LR')
    plt.ylabel('Loss avg')
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Final accuracy vs learning rate
    plt.subplot(1, 3, 3)
    plt.semilogx(lrs, train_accs, 'blue', label='Training accuracy')
    plt.semilogx(lrs, val_accs, 'red', label='Validation accuracy')
    idx = lrs.index(0.5)
    plt.annotate('0.5', (lrs[idx], train_accs[idx]))
    plt.title('Accuracy/LR')
    plt.xlabel('LR')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.show()

train_df, val_df, test_df = get_dataframes()

# Run experiments for stochastic gradient descent
sgd_results = experiment(train_df, val_df, LEARNING_RATES)
lr_diff_graph(sgd_results)

# Run experiments for batch gradient descent
batch_results = experiment(train_df, val_df, LEARNING_RATES, is_stochastic=False)
lr_diff_graph(batch_results)