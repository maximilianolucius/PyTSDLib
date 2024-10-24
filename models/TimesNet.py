# Reference: https://openreview.net/pdf?id=ju_Uqw384Oq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    Perform Fast Fourier Transform (FFT) on the input tensor to identify dominant periods.

    Args:
        x (torch.Tensor): Input tensor of shape [B, T, C], where
                          B = batch size,
                          T = sequence length,
                          C = number of channels/features.
        k (int): Number of top periods to identify based on amplitude. Default is 2.

    Returns:
        tuple:
            - period (list of int): List containing the top k periods identified.
            - period_weight (torch.Tensor): Tensor containing the average amplitude
                                           corresponding to each of the top k periods.
    """
    # Compute the real FFT along the temporal dimension (dim=1)
    xf = torch.fft.rfft(x, dim=1)

    # Calculate the mean amplitude across batches and channels for each frequency
    frequency_list = abs(xf).mean(0).mean(-1)

    # Exclude the zero frequency component by setting its amplitude to zero
    frequency_list[0] = 0

    # Identify the top k frequencies with the highest amplitudes
    _, top_list = torch.topk(frequency_list, k)

    # Convert the top frequency indices to a NumPy array for processing
    top_list = top_list.detach().cpu().numpy()

    # Calculate the corresponding periods based on the top frequencies
    period = x.shape[1] // top_list

    # Return the identified periods and their corresponding average amplitudes
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    TimesBlock module implementing the core component of the TimesNet architecture.
    Utilizes FFT to identify dominant periods and applies inception-based convolutional blocks.

    Args:
        configs (object): Configuration object containing model hyperparameters.
    """

    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len        # Length of the input sequence
        self.pred_len = configs.pred_len      # Length of the prediction sequence
        self.k = configs.top_k                 # Number of top periods to consider

        # Define a sequence of inception-based convolutional blocks with GELU activation
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        """
        Forward pass of the TimesBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, C], where
                              B = batch size,
                              T = sequence length,
                              C = number of channels/features.

        Returns:
            torch.Tensor: Output tensor after processing, with the same shape as input.
        """
        B, T, N = x.size()  # Batch size, sequence length, number of channels

        # Identify the top k periods and their corresponding amplitude weights using FFT
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []  # List to store results from each period-specific convolution

        for i in range(self.k):
            period = period_list[i]  # Current period being processed

            # Determine if padding is needed to make the sequence length divisible by the period
            if (self.seq_len + self.pred_len) % period != 0:
                # Calculate the required length after padding
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                # Create a padding tensor with zeros
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                # Concatenate the padding to the input tensor along the temporal dimension
                out = torch.cat([x, padding], dim=1)
            else:
                # If no padding is needed, use the original input
                length = (self.seq_len + self.pred_len)
                out = x

            # Reshape the tensor to separate the periodic segments
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # Apply 2D convolutional blocks to capture variations within each period
            out = self.conv(out)

            # Reshape the tensor back to its original shape after convolution
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)

            # Trim the output to match the original sequence length
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # Stack the results from all periods along a new dimension
        res = torch.stack(res, dim=-1)

        # Apply softmax to the period weights for adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        # Reshape and repeat the weights to match the dimensions of the stacked results
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)

        # Perform weighted sum across the periods to aggregate the features
        res = torch.sum(res * period_weight, -1)

        # Add a residual connection to the input for better gradient flow and model stability
        res = res + x

        return res


class Model(nn.Module):
    """
    TimesNet Model implementing various time series tasks such as forecasting,
    and classification based on the task specified in configurations.

    Reference:
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Initialize a list of TimesBlock modules as per the number of encoder layers
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])

        # Initialize the data embedding layer for encoder inputs
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.layer = configs.e_layers  # Total number of encoder layers
        self.layer_norm = nn.LayerNorm(configs.d_model)  # Layer normalization for stability

        # Define task-specific output layers based on the task type
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # Linear layer to align temporal dimensions before projection
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            # Projection layer to map model outputs to the desired number of output channels
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name == 'classification':
            self.act = F.gelu  # Activation function for classification
            self.dropout = nn.Dropout(configs.dropout)  # Dropout layer for regularization
            # Projection layer mapping concatenated embeddings to the number of classes
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Perform forecasting by processing encoder inputs and generating predictions.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor): Decoder input tensor (unused in TimesNet).
            x_mark_dec (torch.Tensor): Decoder time features (unused in TimesNet).

        Returns:
            torch.Tensor: Forecasted values for the prediction horizon.
        """
        # Normalize the encoder inputs for non-stationary time series
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embed the normalized encoder inputs along with time features
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # Shape: [B, T, C]

        # Align temporal dimension using the prediction linear layer
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # Shape: [B, T, C]

        # Pass the embedded inputs through each TimesBlock layer
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))  # Shape maintained: [B, T, C]

        # Project the outputs back to the desired number of output channels
        dec_out = self.projection(enc_out)

        # De-normalize the outputs to revert to the original scale
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        Perform classification by aggregating encoder outputs and mapping to class scores.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.
            x_mark_enc (torch.Tensor): Encoder time features.

        Returns:
            torch.Tensor: Class scores for each input sample.
        """
        # Embed the encoder inputs (no time features used for classification)
        enc_out = self.enc_embedding(x_enc, None)  # Shape: [B, T, C]

        # Pass the embedded inputs through each TimesBlock layer
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))  # Shape maintained: [B, T, C]

        # Apply activation function
        output = self.act(enc_out)

        # Apply dropout for regularization
        output = self.dropout(output)

        # Zero-out padding embeddings if any (assuming x_mark_enc indicates padding)
        output = output * x_mark_enc.unsqueeze(-1)

        # Flatten the output by concatenating all temporal features
        output = output.reshape(output.shape[0], -1)  # Shape: [B, T * C]

        # Project the concatenated embeddings to obtain class scores
        output = self.projection(output)  # Shape: [B, num_classes]

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass of the TimesNet model, routing to task-specific methods based on the task name.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor): Decoder input tensor (unused in TimesNet).
            x_mark_dec (torch.Tensor): Decoder time features (unused in TimesNet).
            mask (torch.Tensor, optional): Mask tensor for imputation tasks. Defaults to None.

        Returns:
            torch.Tensor: Output tensor corresponding to the specified task.
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # Perform forecasting and return only the prediction horizon
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # Shape: [B, L, D]

        if self.task_name == 'classification':
            # Perform classification
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # Shape: [B, N]

        # If task name does not match any known task, return None
        return None
