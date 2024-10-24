import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class Splitting(nn.Module):
    """
    Module to split the input tensor into even and odd indexed sequences.
    """
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts even-indexed elements from the sequence.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, channels].

        Returns:
            torch.Tensor: Even-indexed tensor.
        """
        return x[:, ::2, :]

    def odd(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts odd-indexed elements from the sequence.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, channels].

        Returns:
            torch.Tensor: Odd-indexed tensor.
        """
        return x[:, 1::2, :]

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Splits the input tensor into even and odd parts.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, channels].

        Returns:
            tuple: (even_tensor, odd_tensor)
        """
        return self.even(x), self.odd(x)


class CausalConvBlock(nn.Module):
    """
    Causal Convolutional Block with replication padding, convolution, activation, and dropout.
    """
    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.0):
        """
        Initializes the CausalConvBlock.

        Args:
            d_model (int): Number of input and output channels.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 5.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(CausalConvBlock, self).__init__()
        padding = kernel_size - 1
        self.causal_conv = nn.Sequential(
            nn.ReplicationPad1d((padding, padding)),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the causal convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, seq_len].

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        return self.causal_conv(x)


class SCIBlock(nn.Module):
    """
    SCINet Block that performs splitting, causal convolutions, and interactions between even and odd sequences.
    """
    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.0):
        """
        Initializes the SCIBlock.

        Args:
            d_model (int): Number of input and output channels.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 5.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(SCIBlock, self).__init__()
        self.splitting = Splitting()
        self.modules_even = CausalConvBlock(d_model, kernel_size, dropout)
        self.modules_odd = CausalConvBlock(d_model, kernel_size, dropout)
        self.interactor_even = CausalConvBlock(d_model, kernel_size, dropout)
        self.interactor_odd = CausalConvBlock(d_model, kernel_size, dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the SCIBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, channels].

        Returns:
            tuple: (updated_even_tensor, updated_odd_tensor)
        """
        x_even, x_odd = self.splitting(x)
        x_even = x_even.permute(0, 2, 1)  # [batch_size, channels, seq_len_even]
        x_odd = x_odd.permute(0, 2, 1)    # [batch_size, channels, seq_len_odd]

        # Apply causal convolutions and interactions
        x_even_temp = x_even * torch.exp(self.modules_even(x_odd))
        x_odd_temp = x_odd * torch.exp(self.modules_odd(x_even))

        x_even_update = x_even_temp + self.interactor_even(x_odd_temp)
        x_odd_update = x_odd_temp - self.interactor_odd(x_even_temp)

        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)


class SCINet(nn.Module):
    """
    Recursive SCINet architecture with multiple levels.
    """
    def __init__(self, d_model: int, current_level: int = 3, kernel_size: int = 5, dropout: float = 0.0):
        """
        Initializes the SCINet.

        Args:
            d_model (int): Number of input and output channels.
            current_level (int, optional): Current recursion level. Defaults to 3.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 5.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(SCINet, self).__init__()
        self.current_level = current_level
        self.working_block = SCIBlock(d_model, kernel_size, dropout)

        if self.current_level > 0:
            self.SCINet_Tree_even = SCINet(d_model, current_level - 1, kernel_size, dropout)
            self.SCINet_Tree_odd = SCINet(d_model, current_level - 1, kernel_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SCINet.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, channels].

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        odd_flag = False
        if x.shape[1] % 2 == 1:
            odd_flag = True
            x = torch.cat((x, x[:, -1:, :]), dim=1)  # Pad to make even length

        x_even_update, x_odd_update = self.working_block(x)

        if odd_flag:
            x_odd_update = x_odd_update[:, :-1, :]

        if self.current_level == 0:
            return self.zip_up(x_even_update, x_odd_update)
        else:
            return self.zip_up(
                self.SCINet_Tree_even(x_even_update),
                self.SCINet_Tree_odd(x_odd_update)
            )

    def zip_up(self, even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
        """
        Merges even and odd sequences back into a single sequence.

        Args:
            even (torch.Tensor): Even-indexed tensor of shape [batch_size, seq_len/2, channels].
            odd (torch.Tensor): Odd-indexed tensor of shape [batch_size, seq_len/2, channels].

        Returns:
            torch.Tensor: Merged tensor of shape [batch_size, seq_len, channels].
        """
        even = even.permute(1, 0, 2)  # [seq_len/2, batch_size, channels]
        odd = odd.permute(1, 0, 2)    # [seq_len/2, batch_size, channels]
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        min_len = min(even_len, odd_len)

        zipped_data = []
        for i in range(min_len):
            zipped_data.append(even[i].unsqueeze(0))
            zipped_data.append(odd[i].unsqueeze(0))

        if even_len > odd_len:
            zipped_data.append(even[-1].unsqueeze(0))

        zipped = torch.cat(zipped_data, dim=0)  # [seq_len, batch_size, channels]
        return zipped.permute(1, 0, 2)          # [batch_size, seq_len, channels]


class TimeSeriesSCINetModel(nn.Module):
    """
    Time Series Forecasting Model based on SCINet Architecture.

    Paper Reference: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(self, configs: dict):
        """
        Initializes the TimeSeriesSCINetModel.

        Args:
            configs (dict): Configuration dictionary containing model parameters.
        """
        super(TimeSeriesSCINetModel, self).__init__()
        self.task_name = configs.get('task_name', 'forecast')
        self.seq_len = configs.get('seq_len', 96)
        self.label_len = configs.get('label_len', 48)
        self.pred_len = configs.get('pred_len', 24)
        self.enc_in = configs.get('enc_in', 10)
        self.d_layers = configs.get('d_layers', 1)
        self.dropout = configs.get('dropout', 0.0)
        self.kernel_size = configs.get('kernel_size', 5)

        # Initialize SCINet stacks
        if self.d_layers == 1:
            self.sci_net_1 = SCINet(self.enc_in, current_level=3, kernel_size=self.kernel_size, dropout=self.dropout)
            self.projection_1 = nn.Conv1d(self.seq_len, self.seq_len + self.pred_len, kernel_size=1, stride=1, bias=False)
        else:
            self.sci_net_1 = SCINet(self.enc_in, current_level=3, kernel_size=self.kernel_size, dropout=self.dropout)
            self.sci_net_2 = SCINet(self.enc_in, current_level=3, kernel_size=self.kernel_size, dropout=self.dropout)
            self.projection_1 = nn.Conv1d(self.seq_len, self.pred_len, kernel_size=1, stride=1, bias=False)
            self.projection_2 = nn.Conv1d(self.seq_len + self.pred_len, self.seq_len + self.pred_len, kernel_size=1, bias=False)

        # Positional Encoding
        self.pe_hidden_size = self.enc_in
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1

        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        self.register_buffer('inv_timescales', inv_timescales)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor = None,
                x_dec: torch.Tensor = None, x_mark_dec: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, enc_in].
            x_mark_enc (torch.Tensor, optional): Encoder time features. Defaults to None.
            x_dec (torch.Tensor, optional): Decoder input tensor. Defaults to None.
            x_mark_dec (torch.Tensor, optional): Decoder time features. Defaults to None.
            mask (torch.Tensor, optional): Mask tensor for imputation. Defaults to None.

        Returns:
            torch.Tensor: Output tensor corresponding to the task.
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [batch_size, pred_len, enc_in]
            dec_out = torch.cat([torch.zeros_like(x_enc), dec_out], dim=1)  # [batch_size, seq_len + pred_len, enc_in]
            return dec_out  # [batch_size, seq_len + pred_len, enc_in]
        return None

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor = None,
                x_dec: torch.Tensor = None, x_mark_dec: torch.Tensor = None) -> torch.Tensor:
        """
        Performs forecasting using SCINet.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, enc_in].
            x_mark_enc (torch.Tensor, optional): Encoder time features. Defaults to None.
            x_dec (torch.Tensor, optional): Decoder input tensor. Defaults to None.
            x_mark_dec (torch.Tensor, optional): Decoder time features. Defaults to None.

        Returns:
            torch.Tensor: Forecasted tensor of shape [batch_size, pred_len, enc_in].
        """
        # Normalize the input
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = (x_enc - means) / torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5)

        # Add positional encoding
        pe = self.get_position_encoding(x_enc)
        if pe.shape[2] > x_enc.shape[2]:
            x_enc = x_enc + pe[:, :, :-1]
        else:
            x_enc = x_enc + pe

        # First SCINet stack
        dec_out = self.sci_net_1(x_enc)  # [batch_size, seq_len, enc_in]
        dec_out = dec_out + x_enc  # Residual connection
        dec_out = self.projection_1(dec_out.permute(0, 2, 1))  # [batch_size, seq_len + pred_len, enc_in]

        if self.d_layers != 1:
            dec_out = torch.cat([x_enc.permute(0, 2, 1), dec_out], dim=1)  # [batch_size, seq_len + pred_len, enc_in]
            temp = dec_out
            dec_out = self.sci_net_2(dec_out.permute(0, 2, 1))  # [batch_size, seq_len + pred_len, enc_in]
            dec_out = dec_out + temp.permute(0, 2, 1)  # Residual connection
            dec_out = self.projection_2(dec_out.permute(0, 2, 1))  # [batch_size, seq_len + pred_len, enc_in]

        # Denormalize the output
        dec_out = dec_out * (torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach())
        dec_out = dec_out + means.detach()

        return dec_out[:, -self.pred_len:, :]  # [batch_size, pred_len, enc_in]

    def get_position_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates positional encoding for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, enc_in].

        Returns:
            torch.Tensor: Positional encoding tensor of shape [batch_size, seq_len, pe_hidden_size].
        """
        max_length = x.size(1)
        position = torch.arange(max_length, dtype=torch.float32, device=x.device).unsqueeze(1)  # [seq_len, 1]
        scaled_time = position * self.inv_timescales.unsqueeze(0)  # [seq_len, pe_hidden_size//2]
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  # [seq_len, pe_hidden_size]
        if self.pe_hidden_size % 2 == 1:
            signal = F.pad(signal, (0, 1))  # Pad if hidden size is odd
        signal = signal.unsqueeze(0).repeat(x.size(0), 1, 1)  # [batch_size, seq_len, pe_hidden_size]
        return signal
