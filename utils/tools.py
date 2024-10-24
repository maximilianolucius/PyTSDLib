import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from pathlib import Path


# Switch matplotlib backend to 'agg' to allow saving plots without display.
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjusts the learning rate based on the specified adjustment strategy.
    """
    lr_adjust = None
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'cosine':
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}

    if lr_adjust and epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """

    def __init__(self, patience=7, verbose=False, delta=0, save_path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Saves the model when validation loss decreases.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # Ensure `path` is a Path object and concatenate using '/'
        checkpoint_path = Path(path) / 'checkpoint.pth'

        # Save the model's state dictionary
        torch.save(model.state_dict(), checkpoint_path)

        # Update the minimum validation loss
        self.val_loss_min = val_loss


class dotdict(dict):
    """
    Dot notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    """
    Standardization of input data based on provided mean and std.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Visualize the ground truth and prediction results.
    Saves the plot to the specified file.
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()  # Close the figure to avoid memory issues


def adjustment(gt, pred):
    """
    Adjusts predictions based on ground truth and maintains state between anomalies.
    Useful for anomaly detection tasks.
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Backpropagate the anomaly
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                elif pred[j] == 0:
                    pred[j] = 1
            # Forward propagate the anomaly
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                elif pred[j] == 0:
                    pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """
    Calculates the accuracy between predictions and true labels.
    """
    return np.mean(y_pred == y_true)
