import matplotlib.pyplot as plt
import os

def plot_training_history(history, save_dir='plots', model_name='model'):
    """
    Plots validation vs training loss and accuracy.
    Saves the figure to the 'plots' directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['valid_loss'], 'r-', label='Validation Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    ax2.plot(epochs, history['valid_acc'], 'r-', label='Validation Acc')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Save file
    plot_path = os.path.join(save_dir, f'{model_name}_history.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()