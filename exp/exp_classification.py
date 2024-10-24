import logging
import os
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class Exp_Classification(Exp_Basic):
    """
    Experiment class for Classification tasks.
    """

    def __init__(self, args: Any):
        """
        Initialize the Classification experiment.

        Args:
            args (Any): Parsed command-line arguments containing experiment settings.
        """
        super(Exp_Classification, self).__init__(args)
        logger.info("Initialized Exp_Classification.")

    def _build_model(self) -> nn.Module:
        """
        Build and return the classification model.

        Returns:
            nn.Module: The constructed classification model.
        """
        # Retrieve training and testing data to set model input dimensions
        train_data, _ = self._get_data(flag='TRAIN')
        test_data, _ = self._get_data(flag='TEST')

        # Update arguments based on data
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0  # No prediction length for classification
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)

        # Initialize the model
        model = self.model_dict[self.args.model].Model(self.args).float()
        logger.info(f"Model {self.args.model} built with enc_in={self.args.enc_in} and num_class={self.args.num_class}.")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            logger.info(f"Using multiple GPUs: {self.args.device_ids}")

        return model

    def _get_data(self, flag: str) -> Tuple[Any, DataLoader]:
        """
        Get the dataset and dataloader for a given flag.

        Args:
            flag (str): One of 'TRAIN', 'TEST'.

        Returns:
            Tuple[Any, DataLoader]: The dataset and its DataLoader.
        """
        data_set, data_loader = data_provider(self.args, flag)
        logger.info(f"Data loaded for flag='{flag}' with {len(data_set)} samples.")
        return data_set, data_loader

    def _select_optimizer(self) -> optim.Optimizer:
        """
        Select and return the optimizer.

        Returns:
            optim.Optimizer: The selected optimizer.
        """
        optimizer = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        logger.info(f"Optimizer RAdam selected with learning rate {self.args.learning_rate}.")
        return optimizer

    def _select_criterion(self) -> nn.Module:
        """
        Select and return the loss criterion.

        Returns:
            nn.Module: The selected loss function.
        """
        criterion = nn.CrossEntropyLoss()
        logger.info("CrossEntropyLoss criterion selected.")
        return criterion

    def vali(self, vali_data: Any, vali_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate the model on the validation dataset.

        Args:
            vali_data (Any): Validation dataset.
            vali_loader (DataLoader): Validation DataLoader.
            criterion (nn.Module): Loss function.

        Returns:
            Tuple[float, float]: Average validation loss and accuracy.
        """
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        logger.info("Validation started.")
        with torch.no_grad():
            for batch_x, label, padding_mask in vali_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                # Adjust output dimensions based on features
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                loss = criterion(outputs, label.long().squeeze(-1))
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        average_loss = np.mean(total_loss)
        logger.info(f"Validation loss: {average_loss:.7f}")

        # Concatenate all predictions and true labels
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        # Compute accuracy
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        self.model.train()
        return average_loss, accuracy

    def train(self, setting: str) -> nn.Module:
        """
        Train the classification model.

        Args:
            setting (str): Experiment setting identifier.

        Returns:
            nn.Module: The trained model.
        """
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        checkpoint_dir = Path(self.args.checkpoints) / setting
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory set to: {checkpoint_dir}")

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(1, self.args.train_epochs + 1):
            iter_count = 0
            train_losses = []
            self.model.train()
            epoch_start_time = time.time()
            logger.info(f"Epoch {epoch} started.")

            for i, (batch_x, label, padding_mask) in enumerate(train_loader, 1):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_losses.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optimizer.step()

                if i % 100 == 0:
                    elapsed = time.time() - time_now
                    speed = elapsed / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info(f"\titers: {i}, epoch: {epoch} | loss: {loss.item():.7f}")
                    logger.info(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            epoch_duration = time.time() - epoch_start_time
            average_train_loss = np.mean(train_losses)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            logger.info(f"Epoch: {epoch}, Steps: {train_steps} | "
                        f"Train Loss: {average_train_loss:.3f} "
                        f"Vali Loss: {vali_loss:.3f} Vali Acc: {val_accuracy:.3f} "
                        f"Test Loss: {test_loss:.3f} Test Acc: {test_accuracy:.3f} | "
                        f"Epoch Time: {epoch_duration:.2f}s")

            # Early stopping based on validation accuracy
            early_stopping(-val_accuracy, self.model, checkpoint_dir)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break

            # Adjust learning rate
            adjust_learning_rate(optimizer, epoch, self.args)

        # Load the best model
        best_model_path = checkpoint_dir / 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Best model loaded from {best_model_path}")

        return self.model

    def test(self, setting: str, test_flag: int = 0) -> None:
        """
        Test the classification model.

        Args:
            setting (str): Experiment setting identifier.
            test_flag (int, optional): Flag to indicate loading the model. Defaults to 0.
        """
        test_data, test_loader = self._get_data(flag='TEST')

        if test_flag:
            model_path = Path(self.args.checkpoints) / setting / 'checkpoint.pth'
            if not model_path.exists():
                logger.error(f"Model checkpoint {model_path} does not exist.")
                raise FileNotFoundError(f"Model checkpoint {model_path} not found.")
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"Model loaded from {model_path}")

        preds = []
        trues = []
        results_dir = Path('./results') / setting
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Test results will be saved to {results_dir}")

        self.model.eval()
        logger.info("Testing started.")
        with torch.no_grad():
            for batch_x, label, padding_mask in test_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        # Concatenate all predictions and true labels
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        logger.info(f"Test data shape: preds={preds.shape}, trues={trues.shape}")

        # Compute probabilities and predictions
        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,)
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        logger.info(f"Test Accuracy: {accuracy:.4f}")

        # Save results
        result_file = results_dir / 'result_classification.txt'
        with result_file.open('a') as f:
            f.write(f"{setting}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
        logger.info(f"Test results saved to {result_file}")
