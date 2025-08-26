import torch
import torch.nn
import pandas as pd
import torch.optim as optim
import os

from tqdm import tqdm
from extract.extract import fetch_transform
from nnet.model import BasicNN
from torch.utils.data import DataLoader, random_split
from nnet.data_gen import CoverTypeDataset
from plots.plot_utils import plot_accuracy_loss, plot_confusion_matrix

BATCH_SIZE = 300
NUM_EPOCHS = 25
model_name = f'CovType_NN_{BATCH_SIZE}_{NUM_EPOCHS}'
only_train = False
train_losses, val_losses = [], []
train_accs, val_accs= [], []
y_preds, y_test = [], []
epochs = []

if __name__ == "__main__":
   
    data_path =  f'{os.getcwd()}/data'
    X, y = fetch_transform(data_path=data_path)

    model = BasicNN(input_dim = X.shape[1], num_classes=7)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    full_data = CoverTypeDataset(file_path="/Users/jasonluo/Documents/Neural_Net_Stuff/NN1/data/transformed/transformed_data.csv")
    train_size = int(len(full_data) * 0.8)
    val_size = int(len(full_data) * 0.1)
    test_size = len(full_data) - (train_size + val_size)

    train_data, val_data, test_data = random_split(full_data, [train_size, val_size, test_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


    for epoch in tqdm(range(NUM_EPOCHS), desc = "### Training ### ", leave = False):
        epochs.append(epoch)
        tqdm.write(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        running_loss = 0.0
        total_correct = 0

        for features, labels in tqdm(train_loader):
            optimizer.zero_grad() # clear gradients

            outputs = model(features)
            preds_values, preds_class = torch.max(outputs, 1)
            loss = criterion(outputs, labels) 

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
            total_correct += (preds_class == labels).sum().item()


            
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        train_acc = total_correct / len(train_loader.dataset)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        total_correct = 0
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc =  "### Validation ###"):
                outputs = model(features)
                preds_values, preds_class = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * features.size(0)
                total_correct += (preds_class == labels).sum().item()

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        val_acc = total_correct / len(val_loader.dataset)
        val_accs.append(val_acc)

        print(f"Epoch: {(epoch+1)} / {NUM_EPOCHS}. Train acc: {train_acc}. Val acc: {val_acc}. Train loss: {train_loss}. Val loss: {val_loss}. LR: {optimizer.param_groups[0]['lr']} \n")


     # Create subfolder for that particular model
    os.makedirs(f"{os.getcwd()}/saved_models/{model_name}", exist_ok=True)
    torch.save(model, f'{os.getcwd()}/saved_models/{model_name}/{model_name}.pt')


    plot_accuracy_loss(epochs=NUM_EPOCHS, training_loss=train_losses,val_loss=val_losses,
                       training_acc=train_accs, val_acc=val_accs, output_path=f'{os.getcwd()}/plots/{model_name}_acc_loss.png')
    

    # Evaluate model on test set
    if only_train is False:
        model_path = f'{os.getcwd()}/saved_models/{model_name}/{model_name}.pt'
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()

        total_correct = 0
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc =  "### Testing ###"):
                outputs = model(features)
                preds_values, preds_class = torch.max(outputs, 1)
                total_correct += (preds_class == labels).sum().item()

                y_preds.extend(preds_class.tolist())
                y_test.extend(labels.tolist())

        print(f"Testing Accuracy: {total_correct / len(test_loader.dataset)}")

        plot_confusion_matrix(y_pred = y_preds, y_test = y_test, output_path=f'{os.getcwd()}/plots/{model_name}_CM.png')