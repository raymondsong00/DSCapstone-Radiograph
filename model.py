import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(filename='log.txt',
        filemode='a',
        level=logging.INFO,
        datefmt='%H:%M:%S',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')

def get_model(model_name):
    """
    Takes model name and replaces the last layer
    
    Parameters:
    model_name: Model Name either 'resnet' or 'vgg'
    
    Returns:
    Returns selected model
    """
    model = None
    if model_name == 'resnet':
        model = models.resnet152(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif model_name == 'vgg':
        model = models.vgg19_bn(weights='DEFAULT')
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 1)
    else:
        raise("Invalid model name")
    return model

def prepare_training(config):
    """
    Prepares all training needs based on config.json
    
    Parameters:
    config: Training configuration from config.json
    
    Returns:
    model: Neural Network
    criterion: Loss Function
    optimizer: Weight update optimizer
    Scheduler: Learning Rate Scheduler
    Transforms: 
    """
    if config['training']['criterion'] == 'MAE':
        criterion = torch.nn.L1Loss()
    elif config['training']['criterion'] == 'MSE':
        criterion = torch.nn.MSELoss()
        
    # Select Model and transforms
    if config['model'] == 'resnet':
        model = get_model('resnet')
        transforms = models.ResNet152_Weights.IMAGENET1K_V2.transforms()
    elif config['model'] == 'vgg':
        model = get_model('vgg')
        transforms = models.VGG19_BN_Weights.IMAGENET1K_V1.transforms()
    elif config['model'] == 'resnet50':
        model = get_model('resnet50')
        transforms = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    else:
        raise('invalid model')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay'])
    if config['training']['use_scheduler']:
        scheduler=StepLR(optimizer,
                         config['training']['scheduler_step_size'],\
                         gamma=config['training']['lr_decay_rate'])
        
    return model, criterion, optimizer, scheduler, transforms

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, config):
    """
    Trains a CNN model with the arguments passed in
    
    Parameters:
    model: Pytorch Model
    criterion: Loss Function eg torch.nn.MSELoss()
    optimizer: Weight updater
    scheduler: Learning rate Scheduler
    dataloaders: Dictionary of data loaders
    device: Device to use for training
    
    Returns:
    model: Pytorch Model with weights from lowest validation loss
    train_loss: Training loss over all epochs
    val_loss: Validation loss over all epochs
    """
    best_loss = float('inf')
    best_loss_epoch = 0
    
    early_stop_count = 0
    
    best_model_params_path = config['filepaths']['saved_weights_path']
    
    train_loss = []
    val_loss = []

    for epoch in tqdm(range(config['training']['start_epoch'], config['training']['end_epoch'])):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            dataset_size = 0.0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                dataset_size += inputs.size(0)
                
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            logging.info(f'Epoch: {epoch}, {phase} Loss: {epoch_loss:.4f}')
            
            if phase == 'train':
                train_loss.append(epoch_loss)
            elif phase == 'val':
                val_loss.append(epoch_loss)

            # Deep copy model if validation loss is better
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_loss_epoch = epoch
                early_stop_count = 0
                torch.save(model.state_dict(), best_model_params_path)
            elif phase == 'val' and epoch_loss > best_loss:
                early_stop_count += 1
                
        if early_stop_count > config['training']['estop_num_epochs'] and config['training']['use_estop']:
            break

    logging.info(f'Best val loss: {best_loss:4f}, Epoch: {best_loss_epoch}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    
    return model, train_loss, val_loss

def test_model(model, criterion, device, dataloader):
    """
    Trains a CNN model with the arguments passed in
    
    Parameters:
    model: Pytorch Model to Test
    criterion: Loss Function eg torch.nn.MSELoss()
    dataloader: Test dataloader
    device: Device to use for training
    
    Returns:
    model: Pytorch Model with weights from lowest validation loss
    actual_labels: Actual data values
    predicted_labels: Predicted data values
    """
    model.eval()
    
    running_loss = 0.0
    dataset_size = 0.0
    
    predicted_labels = []
    actual_labels = []
    
    for inputs, labels, in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)
            
            actual_labels.append(labels.cpu().numpy())
            predicted_labels.append(outputs.cpu().numpy())
            
    loss = running_loss / dataset_size
    
    actual_labels = np.concatenate(actual_labels)
    predicted_labels = np.concatenate(predicted_labels)
    
    return loss, actual_labels, predicted_labels

def save_results(actual, predicted, config):
    result_df = pd.DataFrame({'Actual ln BNPP': actual.squeeze(), 
                                  'Predicted ln BNPP': predicted.squeeze()})
    result_df.to_csv(config['filepaths']['results_csv_path'], index=False)

def plot_loss(train_loss, val_loss, config):
    
    epochs = np.arange(len(train_loss)) + 1
    
    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, 'b', label='Train Loss')
    ax.plot(epochs, val_loss, 'r', label='Validation Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title("Training Loss Plot")
    
    model_name, loss_name = config['model'], config['training']['criterion']
    
    fig.set_size_inches(8, 8)
    fig.tight_layout(pad=2.0)
    fig.suptitle(f'Model: {model_name}, Loss: {loss_name}')
    plt.savefig(config['filepaths']['loss_plot_path'])

def plot_results(actual_labels, predicted_labels, config):
    fig, ax = plt.subplots()
    
    ax.scatter(actual_labels.astype(float), predicted_labels.astype(float))
    
    # https://stackoverflow.com/questions/37234163/how-to-add-a-line-of-best-fit-to-scatter-plot
    z = np.polyfit(x=actual_labels, y=predicted_labels, deg=1)
    xx = np.linspace(*plt.gca().get_xlim()).T
    ax.plot(xx, z[0]*xx + z[1], '-r', label=f'{z[0]:.2f}*x + {z[1]:.2f}')

    ax.set_xlabel('Actual ln BNPP')
    ax.set_ylabel('Predicted ln BNPP')
    ax.set_title("Actual v Predicted log BNPP")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.legend()
    
    model_name, loss_name = config['model'], config['training']['criterion']
    
    pearson_corr = np.corrcoef(actual_labels, predicted_labels)
    fig.suptitle(f'Model: {model_name}, Loss: {loss_name} \nr: {pearson_corr[0][1]:.5f}')
    
    fig.set_size_inches(8, 8)
    fig.tight_layout(pad=2.0)
    plt.savefig(config['filepaths']['results_plot_path'])

def plot_combined(train_loss, val_loss, actual_labels, predicted_labels, config):
    fig, ax = plt.subplots(1,2)
    epochs = np.arange(len(train_loss)) + 1
    
    ax[0].plot(epochs, train_loss, 'b', label='Train Loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation Loss')
    
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title("Training Loss Plot")
    
    ax[1].scatter(actual_labels.astype(float), predicted_labels.astype(float))
    
    # https://stackoverflow.com/questions/37234163/how-to-add-a-line-of-best-fit-to-scatter-plot
    z = np.polyfit(x=actual_labels, y=predicted_labels, deg=1)
    xx = np.linspace(*plt.gca().get_xlim()).T
    ax[1].plot(xx, z[0]*xx + z[1], '-r', label=f'{z[0]:.2f}*x + {z[1]:.2f}')

    
    ax[1].set_xlabel('Actual log BNPP')
    ax[1].set_ylabel('Predicted log BNPP')
    ax[1].set_title("Actual v Predicted log BNPP")
    ax[1].set_xlim(0, 12)
    ax[1].set_ylim(0, 12)
    ax[1].legend()
    
    model_name, loss_name = config['model'], config['training']['criterion']
    
    pearson_corr = np.corrcoef(actual_labels, predicted_labels)
    fig.suptitle(f'Model: {model_name}, Loss: {loss_name}, \nr: {pearson_corr[0][1]:.5f}')
    
    fig.set_size_inches(16, 8)
    
    fig.tight_layout(pad=2.0)
    plt.savefig(config['filepaths']['combined_plot_path'])
