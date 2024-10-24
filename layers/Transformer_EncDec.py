import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvLayer(nn.Module):
    """
    Convolutional Layer with Batch Normalization, ELU Activation, and Max Pooling.

    This layer applies a 1D convolution with circular padding, followed by batch normalization,
    ELU activation, and max pooling to downsample the input sequence.

    Args:
        c_in (int): Number of input channels/features.
    """

    def __init__(self, c_in: int):
        super(ConvLayer, self).__init__()
        self.down_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvLayer.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Length, Channels].

        Returns:
            torch.Tensor: Output tensor after convolution, normalization, activation, and pooling.
        """
        # Permute to [Batch, Channels, Length] for Conv1d
        x = self.down_conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        # Permute back to [Batch, Length, Channels]
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Attention and Convolutional Feed-Forward Network.

    This layer performs multi-head attention followed by a convolutional feed-forward network,
    both equipped with residual connections, layer normalization, dropout, and activation.

    Args:
        attention (nn.Module): Attention mechanism module.
        d_model (int): Dimension of the model.
        d_ff (int, optional): Dimension of the feed-forward network. Defaults to 4 * d_model.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        activation (str, optional): Activation function ('relu' or 'gelu'). Defaults to "relu".
    """

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int = None,
                 dropout: float = 0.1, activation: str = "relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None,
                tau: torch.Tensor = None, delta: torch.Tensor = None) -> tuple:
        """
        Forward pass of the EncoderLayer.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Length, Channels].
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            tau (torch.Tensor, optional): Additional parameter for attention. Defaults to None.
            delta (torch.Tensor, optional): Additional parameter for attention. Defaults to None.

        Returns:
            tuple: Tuple containing the output tensor and attention weights.
        """
        # Multi-head Attention
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        # Convolutional Feed-Forward Network
        y = self.norm1(x)
        y = self.activation(self.conv1(y.transpose(-1, 1)))  # [Batch, d_ff, Length]
        y = self.dropout(self.conv2(y).transpose(-1, 1))     # [Batch, Length, d_model]

        # Residual Connection and Layer Normalization
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    Transformer Encoder composed of multiple EncoderLayers and optional Convolutional Layers.

    Args:
        attn_layers (list of nn.Module): List of attention layers.
        conv_layers (list of nn.Module, optional): List of convolutional layers. Defaults to None.
        norm_layer (nn.Module, optional): Normalization layer to apply after encoding. Defaults to None.
    """

    def __init__(self, attn_layers: list, conv_layers: list = None, norm_layer: nn.Module = None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None,
                tau: torch.Tensor = None, delta: torch.Tensor = None) -> tuple:
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Length, Channels].
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            tau (torch.Tensor, optional): Additional parameter for attention. Defaults to None.
            delta (torch.Tensor, optional): Additional parameter for attention. Defaults to None.

        Returns:
            tuple: Tuple containing the final encoded tensor and a list of attention weights.
        """
        attns = []
        if self.conv_layers:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                current_delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=current_delta)
                x = conv_layer(x)
                attns.append(attn)
            # Apply the last attention layer without convolution
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with Self-Attention, Cross-Attention, and Convolutional Feed-Forward Network.

    This layer performs self-attention, cross-attention with encoder outputs, and a convolutional
    feed-forward network, each equipped with residual connections, layer normalization, dropout, and activation.

    Args:
        self_attention (nn.Module): Self-attention mechanism module.
        cross_attention (nn.Module): Cross-attention mechanism module.
        d_model (int): Dimension of the model.
        d_ff (int, optional): Dimension of the feed-forward network. Defaults to 4 * d_model.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        activation (str, optional): Activation function ('relu' or 'gelu'). Defaults to "relu".
    """

    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module, d_model: int,
                 d_ff: int = None, dropout: float = 0.1, activation: str = "relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: torch.Tensor = None,
                cross_mask: torch.Tensor = None, tau: torch.Tensor = None,
                delta: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the DecoderLayer.

        Args:
            x (torch.Tensor): Decoder input tensor of shape [Batch, Length, Channels].
            cross (torch.Tensor): Encoder output tensor for cross-attention.
            x_mask (torch.Tensor, optional): Attention mask for self-attention. Defaults to None.
            cross_mask (torch.Tensor, optional): Attention mask for cross-attention. Defaults to None.
            tau (torch.Tensor, optional): Additional parameter for attention. Defaults to None.
            delta (torch.Tensor, optional): Additional parameter for cross-attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after self-attention, cross-attention, and convolutional feed-forward network.
        """
        # Self-Attention with Residual Connection
        self_attn_out, _ = self.self_attention(
            x, x, x, attn_mask=x_mask, tau=tau, delta=None
        )
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        # Cross-Attention with Residual Connection
        cross_attn_out, _ = self.cross_attention(
            x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
        )
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)

        # Convolutional Feed-Forward Network with Residual Connection
        y = self.norm2(x)
        y = self.activation(self.conv1(y.transpose(-1, 1)))  # [Batch, d_ff, Length]
        y = self.dropout(self.conv2(y).transpose(-1, 1))     # [Batch, Length, d_model]
        x = x + self.dropout(y)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    """
    Transformer Decoder composed of multiple DecoderLayers, optional Layer Normalization, and Projection.

    Args:
        layers (list of nn.Module): List of decoder layers.
        norm_layer (nn.Module, optional): Normalization layer to apply after decoding. Defaults to None.
        projection (nn.Module, optional): Projection layer to map decoder outputs to desired dimensions. Defaults to None.
    """

    def __init__(self, layers: list, norm_layer: nn.Module = None, projection: nn.Module = None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: torch.Tensor = None,
                cross_mask: torch.Tensor = None, tau: torch.Tensor = None,
                delta: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Decoder.

        Args:
            x (torch.Tensor): Decoder input tensor of shape [Batch, Length, Channels].
            cross (torch.Tensor): Encoder output tensor for cross-attention.
            x_mask (torch.Tensor, optional): Attention mask for self-attention. Defaults to None.
            cross_mask (torch.Tensor, optional): Attention mask for cross-attention. Defaults to None.
            tau (torch.Tensor, optional): Additional parameter for attention. Defaults to None.
            delta (torch.Tensor, optional): Additional parameter for cross-attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after decoding, normalization, and projection.
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm:
            x = self.norm(x)

        if self.projection:
            x = self.projection(x)

        return x
