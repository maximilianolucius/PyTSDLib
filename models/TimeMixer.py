import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import SeriesDecomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class DFTSeriesDecomposition(nn.Module):
    """
    Series Decomposition using Discrete Fourier Transform (DFT).

    Decomposes a time series into seasonal and trend components by retaining
    the top_k frequency components and zeroing out the rest.
    """

    def __init__(self, top_k: int = 5):
        """
        Initializes the DFTSeriesDecomposition module.

        Args:
            top_k (int): Number of top frequency components to retain.
        """
        super(DFTSeriesDecomposition, self).__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass for decomposition.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Seasonal and trend components.
        """
        # Perform real FFT along the time dimension
        xf = torch.fft.rfft(x, dim=1)
        freq_magnitude = torch.abs(xf)

        # Zero out the zero-frequency component (DC component)
        freq_magnitude[:, 0, :] = 0

        # Identify the top_k frequency components
        top_k_values, _ = torch.topk(freq_magnitude, self.top_k, dim=1)
        threshold = top_k_values.min(dim=1, keepdim=True).values.unsqueeze(-1)

        # Zero out frequencies below the threshold
        mask = freq_magnitude >= threshold
        xf = xf * mask.type_as(xf)

        # Inverse FFT to obtain the seasonal component
        x_season = torch.fft.irfft(xf, n=x.size(1), dim=1)

        # Calculate the trend component
        x_trend = x - x_season

        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Multi-Scale Season Mixing module for blending seasonal patterns.

    Utilizes a bottom-up approach by progressively downsampling and mixing
    seasonal components across multiple scales.
    """

    def __init__(self, configs):
        """
        Initializes the MultiScaleSeasonMixing module.

        Args:
            configs: Configuration object containing necessary parameters.
        """
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
                nn.GELU(),
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
            )
            for i in range(configs.down_sampling_layers)
        ])

    def forward(self, season_list: list) -> list:
        """
        Forward pass for multi-scale season mixing.

        Args:
            season_list (list): List of seasonal components at different scales.

        Returns:
            list: Mixed seasonal components.
        """
        if not season_list:
            raise ValueError("season_list should not be empty.")

        # Initialize with the highest scale seasonal component
        out_high = season_list[0]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            # Downsample the high-scale output
            out_low_res = self.down_sampling_layers[i](out_high)
            # Aggregate with the next lower scale seasonal component
            out_low = season_list[i + 1] + out_low_res
            out_high = out_low
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Multi-Scale Trend Mixing module for blending trend patterns.

    Utilizes a top-down approach by progressively upsampling and mixing
    trend components across multiple scales.
    """

    def __init__(self, configs):
        """
        Initializes the MultiScaleTrendMixing module.

        Args:
            configs: Configuration object containing necessary parameters.
        """
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** i),
                ),
                nn.GELU(),
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                ),
            )
            for i in reversed(range(configs.down_sampling_layers))
        ])

    def forward(self, trend_list: list) -> list:
        """
        Forward pass for multi-scale trend mixing.

        Args:
            trend_list (list): List of trend components at different scales.

        Returns:
            list: Mixed trend components.
        """
        if not trend_list:
            raise ValueError("trend_list should not be empty.")

        # Reverse the trend list for top-down mixing
        trend_list_reverse = trend_list[::-1]
        out_low = trend_list_reverse[0]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            # Upsample the low-scale output
            out_high_res = self.up_sampling_layers[i](out_low)
            # Aggregate with the next higher scale trend component
            out_high = trend_list_reverse[i + 1] + out_high_res
            out_low = out_high
            out_trend_list.append(out_low.permute(0, 2, 1))

        # Reverse back to original scale order
        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """
    Past Decomposable Mixing module that handles decomposition and mixing
    of past time series data into seasonal and trend components.
    """

    def __init__(self, configs):
        """
        Initializes the PastDecomposableMixing module.

        Args:
            configs: Configuration object containing necessary parameters.
        """
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence

        # Normalization and dropout layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # Series decomposition method
        if configs.decomp_method == 'moving_avg':
            self.decomposition = SeriesDecomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decomposition = DFTSeriesDecomposition(configs.top_k)
        else:
            raise ValueError('Unsupported decomposition method.')

        # Cross-layer for non-channel independent configurations
        if not self.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.d_model),
            )

        # Multi-scale mixing modules
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        # Output cross-layer
        self.out_cross_layer = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, x_list: list) -> list:
        """
        Forward pass for past decomposable mixing.

        Args:
            x_list (list): List of input tensors at different scales.

        Returns:
            list: List of output tensors after mixing.
        """
        # Decompose each input tensor into seasonal and trend components
        season_list, trend_list = self.decompose_series(x_list)

        # Perform multi-scale mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        # Aggregate seasonal and trend components
        out_list = []
        for ori, out_season, out_trend in zip(x_list, out_season_list, out_trend_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out)

        return out_list

    def decompose_series(self, x_list: list) -> (list, list):
        """
        Decomposes a list of tensors into seasonal and trend components.

        Args:
            x_list (list): List of input tensors.

        Returns:
            Tuple[list, list]: Lists of seasonal and trend components.
        """
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        return season_list, trend_list


class Model(nn.Module):
    """
    Main Model class handling different time series tasks such as forecasting,
    and classification.
    """

    def __init__(self, configs):
        """
        Initializes the Model.

        Args:
            configs: Configuration object containing all necessary parameters.
        """
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence

        # Past Decomposable Mixing blocks
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(configs) for _ in range(configs.e_layers)
        ])

        # Preprocessing decomposition
        self.preprocess = SeriesDecomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        # Embedding layer
        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(
                1, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )

        self.layer = configs.e_layers

        # Normalization layers for different scales
        self.normalize_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=(configs.use_norm == 0))
            for _ in range(configs.down_sampling_layers + 1)
        ])

        # Task-specific layers
        self._init_task_layers()

    def _init_task_layers(self):
        """
        Initializes task-specific layers based on the task name.
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # Prediction layers for forecasting tasks
            self.predict_layers = nn.ModuleList([
                nn.Linear(
                    self.configs.seq_len // (self.configs.down_sampling_window ** i),
                    self.pred_len,
                )
                for i in range(self.configs.down_sampling_layers + 1)
            ])

            # Projection layers based on channel independence
            if self.channel_independence:
                self.projection_layer = nn.Linear(self.configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(self.configs.d_model, self.configs.c_out, bias=True)
                self.out_res_layers = nn.ModuleList([
                    nn.Linear(
                        self.configs.seq_len // (self.configs.down_sampling_window ** i),
                        self.configs.seq_len // (self.configs.down_sampling_window ** i),
                    )
                    for i in range(self.configs.down_sampling_layers + 1)
                ])
                self.regression_layers = nn.ModuleList([
                    nn.Linear(
                        self.configs.seq_len // (self.configs.down_sampling_window ** i),
                        self.pred_len,
                    )
                    for i in range(self.configs.down_sampling_layers + 1)
                ])

        if self.task_name == 'classification':
            # Classification layers
            self.act = F.gelu
            self.dropout = nn.Dropout(self.configs.dropout)
            self.projection = nn.Linear(
                self.configs.d_model * self.seq_len, self.configs.num_class
            )

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor = None,
                x_mark_dec: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape (B, T, N).
            x_mark_enc (torch.Tensor): Encoder time features tensor.
            x_dec (torch.Tensor, optional): Decoder input tensor.
            x_mark_dec (torch.Tensor, optional): Decoder time features tensor.
            mask (torch.Tensor, optional): Mask tensor for imputation.

        Returns:
            torch.Tensor: Output tensor based on the task.
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            raise ValueError('Unsupported task.')

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor,
                 x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> torch.Tensor:
        """
        Forecasting task forward pass.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.
            x_mark_enc (torch.Tensor): Encoder time features tensor.
            x_dec (torch.Tensor): Decoder input tensor.
            x_mark_dec (torch.Tensor): Decoder time features tensor.

        Returns:
            torch.Tensor: Forecasting output.
        """
        # Multi-scale processing
        x_enc_processed, _ = self._multi_scale_process_inputs(x_enc, x_mark_enc)

        # Normalize and prepare input list
        x_list, x_mark_list = self._prepare_input_list(x_enc_processed, x_mark_enc)

        # Embedding
        enc_out_list = self._embed_inputs(x_list, x_mark_list)

        # Past Decomposable Mixing
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Mixing (Assuming this is defined elsewhere)
        dec_out_list = self.future_multi_mixing(x_enc.size(0), enc_out_list, x_list)

        # Aggregate decoder outputs
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        return dec_out

    def future_multi_mixing(self, B: int, enc_out_list: list, x_list: list) -> list:
        """
        Handles multi-scale future mixing for forecasting.

        Args:
            B (int): Batch size.
            enc_out_list (list): List of encoder outputs.
            x_list (list): List of encoder inputs.

        Returns:
            list: List of decoder outputs.
        """
        dec_out_list = []
        if self.channel_independence:
            # Channel independent processing
            for i, enc_out in enumerate(enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            # Channel dependent processing
            for i, (enc_out, out_res) in enumerate(zip(enc_out_list, x_list[1])):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        return dec_out_list

    def out_projection(self, dec_out: torch.Tensor, i: int, out_res: torch.Tensor) -> torch.Tensor:
        """
        Applies projection and residual connections.

        Args:
            dec_out (torch.Tensor): Decoder output tensor.
            i (int): Index for selecting layers.
            out_res (torch.Tensor): Residual output tensor.

        Returns:
            torch.Tensor: Projected decoder output.
        """
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def _multi_scale_process_inputs(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> (list, list):
        """
        Processes inputs through multi-scale downsampling.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.
            x_mark_enc (torch.Tensor): Encoder time features tensor.

        Returns:
            Tuple[list, list]: Processed encoder inputs and time features.
        """
        down_sampling_method = self.configs.down_sampling_method
        down_sampling_window = self.down_sampling_window

        # Select downsampling method
        if down_sampling_method == 'max':
            down_pool = nn.MaxPool1d(down_sampling_window, return_indices=False)
        elif down_sampling_method == 'avg':
            down_pool = nn.AvgPool1d(down_sampling_window)
        elif down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(
                in_channels=self.configs.enc_in,
                out_channels=self.configs.enc_in,
                kernel_size=3,
                padding=padding,
                stride=down_sampling_window,
                padding_mode='circular',
                bias=False
            )
        else:
            # No downsampling
            return [x_enc], [x_mark_enc] if x_mark_enc is not None else [x_enc], None

        # Prepare for downsampling
        x_enc = x_enc.permute(0, 2, 1)  # B, C, T
        x_enc_ori = x_enc
        x_mark_enc_ori = x_mark_enc

        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]  # Initial scale
        x_mark_sampling_list = [x_mark_enc] if x_mark_enc is not None else None

        for _ in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling = x_mark_enc_ori[:, ::down_sampling_window, :]
                x_mark_sampling_list.append(x_mark_sampling)
                x_mark_enc_ori = x_mark_sampling

        return x_enc_sampling_list, x_mark_sampling_list if x_mark_sampling_list is not None else None

    def _prepare_input_list(self, x_enc_processed: list, x_mark_enc: torch.Tensor) -> (list, list):
        """
        Normalizes and prepares the input list for embedding.

        Args:
            x_enc_processed (list): List of processed encoder inputs.
            x_mark_enc (torch.Tensor): Encoder time features tensor.

        Returns:
            Tuple[list, list]: Lists of normalized inputs and corresponding time features.
        """
        x_list = []
        x_mark_list = []

        if x_mark_enc is not None:
            for i, (x, x_mark) in enumerate(zip(x_enc_processed, x_mark_enc)):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in enumerate(x_enc_processed):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        return x_list, x_mark_list

    def _embed_inputs(self, x_list: list, x_mark_list: list) -> list:
        """
        Applies embedding to the input list.

        Args:
            x_list (list): List of input tensors.
            x_mark_list (list): List of time feature tensors.

        Returns:
            list: List of embedded tensors.
        """
        enc_out_list = []
        if self.channel_independence:
            for x in x_list:
                enc_out = self.enc_embedding(x, None)  # [B*T, C]
                enc_out_list.append(enc_out)
        else:
            for x, x_mark in zip(x_list, x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B, T, C]
                enc_out_list.append(enc_out)
        return enc_out_list

    def classification(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """
        Classification task forward pass.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.
            x_mark_enc (torch.Tensor): Encoder time features tensor.

        Returns:
            torch.Tensor: Classification output.
        """
        # Multi-scale processing
        x_enc_processed, _ = self._multi_scale_process_inputs(x_enc, None)
        x_list = x_enc_processed

        # Embedding
        enc_out_list = self._embed_inputs(x_list, None)

        # Past Decomposable Mixing
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Aggregate encoder outputs
        enc_out = enc_out_list[0]

        # Apply activation and dropout
        output = self.act(enc_out)
        output = self.dropout(output)

        # Zero-out padding embeddings if time features are provided
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)

        # Flatten for classification
        output = output.reshape(output.shape[0], -1)

        # Final projection
        output = self.projection(output)  # (batch_size, num_classes)
        return output