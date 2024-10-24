import logging
import os
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics import metric
from utils.dtw_metric import accelerated_dtw
from utils.summary import M4Summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class Exp_Short_Term_Forecast(Exp_Basic):
    """
    Experiment class for Short-Term Forecasting tasks.
    """

    def __init__(self, args: Any):
        """
        Initialize the Short-Term Forecasting experiment.

        Args:
            args (Any): Parsed command-line arguments containing experiment settings.
        """
        super(Exp_Short_Term_Forecast, self).__init__(args)
        logger.info("Initialized Exp_Short_Term_Forecast.")

    def _build_model(self) -> nn.Module:
        """
        Build and return the forecasting model.

        Returns:
            nn.Module: The constructed forecasting model.
        """
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]
            self.args.seq_len = 2 * self.args.pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
            logger.info(f"M4 dataset detected. Set pred_len={self.args.pred_len}, seq_len={self.args.seq_len}, "
                        f"label_len={self.args.label_len}, frequency_map={self.args.frequency_map}")

        model = self.model_dict[self.args.model].Model(self.args).float()
        logger.info(f"Model {self.args.model} built successfully.")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            logger.info(f"Using multiple GPUs: {self.args.device_ids}")

        return model

    def _get_data(self, flag: str) -> Tuple[Any, DataLoader]:
        """
        Get the dataset and dataloader for a given flag.

        Args:
            flag (str): One of 'train', 'val', or 'test'.

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
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        logger.info(f"Optimizer Adam selected with learning rate {self.args.learning_rate}.")
        return optimizer

    def _select_criterion(self, loss_name: str = 'MSE') -> nn.Module:
        """
        Select and return the loss criterion based on the loss name.

        Args:
            loss_name (str): Name of the loss function ('MSE', 'MAPE', 'MASE', 'SMAPE').

        Returns:
            nn.Module: The selected loss function.

        Raises:
            ValueError: If the loss_name is not supported.
        """
        if loss_name == 'MSE':
            criterion = nn.MSELoss()
        elif loss_name == 'MAPE':
            criterion = mape_loss()
        elif loss_name == 'MASE':
            criterion = mase_loss()
        elif loss_name == 'SMAPE':
            criterion = smape_loss()
        else:
            logger.error(f"Unsupported loss function: {loss_name}")
            raise ValueError(f"Unsupported loss function: {loss_name}")

        logger.info(f"{loss_name} loss criterion selected.")
        return criterion

    def vali(self, vali_data: Any, vali_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Validate the model on the validation dataset.

        Args:
            vali_data (Any): Validation dataset.
            vali_loader (DataLoader): Validation DataLoader.
            criterion (nn.Module): Loss function.

        Returns:
            float: Average validation loss.
        """
        total_loss = []
        self.model.eval()
        logger.info("Validation started.")
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

                # Forward pass with optional AMP
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, dec_inp, None)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                else:
                    outputs = self.model(batch_x, None, dec_inp, None)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)

                total_loss.append(loss.item())

        average_loss = np.mean(total_loss)
        self.model.train()
        logger.info(f"Validation completed with loss: {average_loss:.7f}")
        return average_loss

    def train(self, setting: str) -> nn.Module:
        """
        Train the forecasting model.

        Args:
            setting (str): Experiment setting identifier.

        Returns:
            nn.Module: The trained model.
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        checkpoint_dir = Path(self.args.checkpoints) / setting
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory set to: {checkpoint_dir}")

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(1, self.args.train_epochs + 1):
            iter_count = 0
            train_losses = []
            self.model.train()
            epoch_start_time = time.time()
            logger.info(f"Epoch {epoch} started.")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader, 1):
                iter_count += 1
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

                # Forward pass with optional AMP
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, dec_inp, None)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                        # Optional: Uncomment the next line to include loss sharpness
                        # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                        loss = loss_value  # + loss_sharpness * 1e-5
                        train_losses.append(loss.item())
                else:
                    outputs = self.model(batch_x, None, dec_inp, None)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                    # Optional: Uncomment the next line to include loss sharpness
                    # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                    loss = loss_value  # + loss_sharpness * 1e-5
                    train_losses.append(loss.item())

                # Backward pass and optimization
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Logging
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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = vali_loss  # Assuming test_loss is similar to vali_loss

            logger.info(f"Epoch: {epoch}, Steps: {train_steps} | "
                        f"Train Loss: {average_train_loss:.7f} "
                        f"Vali Loss: {vali_loss:.7f} "
                        f"Test Loss: {test_loss:.7f} | "
                        f"Epoch Time: {epoch_duration:.2f}s")

            # Early stopping based on validation loss
            early_stopping(vali_loss, self.model, checkpoint_dir)
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
        Test the forecasting model.

        Args:
            setting (str): Experiment setting identifier.
            test_flag (int, optional): Flag to indicate loading the model. Defaults to 0.
        """
        test_data, test_loader = self._get_data(flag='test')

        if test_flag:
            model_path = Path('./checkpoints') / setting / 'checkpoint.pth'
            if not model_path.exists():
                logger.error(f"Model checkpoint {model_path} does not exist.")
                raise FileNotFoundError(f"Model checkpoint {model_path} not found.")
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"Model loaded from {model_path}")

        preds = []
        trues = []
        folder_path = Path('./test_results') / setting
        folder_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Test results will be saved to {folder_path}")

        self.model.eval()
        logger.info("Testing started.")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

                # Forward pass with optional AMP
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, dec_inp, None)
                else:
                    outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Inverse transform if required
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                # Visualization every 20 iterations
                if i % 20 == 0:
                    input_data = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_data.shape
                        input_data = test_data.inverse_transform(input_data.reshape(shape[0] * shape[1], -1)).reshape(
                            shape)
                    gt = np.concatenate((input_data[0, :, 0], true[0, :, 0]), axis=0)
                    pd = np.concatenate((input_data[0, :, 0], pred[0, :, 0]), axis=0)
                    visual(gt, pd, folder_path / f"{i}.pdf")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        logger.info(f"Test data shape: preds={preds.shape}, trues={trues.shape}")

        # Reshape predictions and true values
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        logger.info(f"Reshaped test data shape: preds={preds.shape}, trues={trues.shape}")

        # Calculate DTW if required
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    logger.info(f"Calculating DTW for iteration: {i}")
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.mean(dtw_list)
        else:
            dtw = 'not calculated'

        # Calculate other metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logger.info(f"Forecasting Metrics - MSE: {mse}, MAE: {mae}, DTW: {dtw}")

        # Save results
        results_save_dir = Path('./results') / setting
        results_save_dir.mkdir(parents=True, exist_ok=True)

        result_file = results_save_dir / 'result_long_term_forecast.txt'
        with result_file.open('a') as f:
            f.write(f"{setting}\n")
            f.write(f"mse:{mse}, mae:{mae}, dtw:{dtw}\n\n")
        logger.info(f"Long-Term Forecasting results saved to {result_file}")

        # Save metrics and predictions
        np.save(results_save_dir / 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(results_save_dir / 'pred.npy', preds)
        np.save(results_save_dir / 'true.npy', trues)
        logger.info(f"Metrics and predictions saved to {results_save_dir}")

        # Additional M4 summary if applicable
        if self.args.data == 'm4':
            file_path = Path('./m4_results') / self.args.model
            if all(f"{freq}_forecast.csv" in os.listdir(file_path) for freq in
                   ['Weekly', 'Monthly', 'Yearly', 'Daily', 'Hourly', 'Quarterly']):
                m4_summary = M4Summary(str(file_path), self.args.root_path)
                smape_results, owa_results, mape, mase = m4_summary.evaluate()
                logger.info(f"SMAPE: {smape_results}")
                logger.info(f"MAPE: {mape}")
                logger.info(f"MASE: {mase}")
                logger.info(f"OWA: {owa_results}")
            else:
                logger.warning("After all 6 tasks are finished, you can calculate the averaged index.")

        return
