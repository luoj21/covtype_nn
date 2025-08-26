import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score

def plot_accuracy_loss(epochs, training_loss, val_loss, training_acc, val_acc, output_path: str):
    """Plots accuracy and loss plots for model training/validation"""
    plt.clf()

    fig, axs = plt.subplots(2,1, figsize = [10,10])

    axs[0].plot(range(0,epochs), training_loss, c='g')
    axs[0].plot(range(0,epochs), val_loss, c='r')
    axs[0].set_title('Training and Validation Loss', fontsize = 10)
    axs[0].legend(['Training Loss', 'Validation Loss'])

    axs[1].plot(range(0,epochs), training_acc, c='g')
    axs[1].plot(range(0,epochs), val_acc, c='r')
    axs[1].set_title('Training and Validation Accuracy' , fontsize = 10)
    axs[1].legend(['Training Accuracy', 'Validation Accuracy'])

    plt.savefig(output_path, dpi = 200)


def plot_confusion_matrix(y_pred, y_test, output_path: str):
    """Plots confusion matrix and saves to file"""
    plt.clf()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot = True)
    plt.title(f'Confusion Matrix with accuracy {accuracy_score(y_test, y_pred)}')

    plt.savefig(output_path, dpi = 200)
