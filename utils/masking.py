import torch


class TriangularCausalMask:
    """
    Creates a triangular causal mask where positions in the sequence can only attend to previous positions (causal masking).
    """

    def __init__(self, B: int, L: int, device: str = "cpu"):
        """
        Initializes the triangular causal mask.

        :param B: Batch size.
        :param L: Sequence length.
        :param device: Device to place the mask (e.g., "cpu" or "cuda").
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # Create an upper triangular mask that blocks future positions
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """
        Returns the generated triangular causal mask.
        """
        return self._mask


class ProbMask:
    """
    Creates a probability-based mask for attention scores, masking out certain positions.
    """

    def __init__(self, B: int, H: int, L: int, index: torch.Tensor, scores: torch.Tensor, device: str = "cpu"):
        """
        Initializes the probability mask.

        :param B: Batch size.
        :param H: Number of heads.
        :param L: Sequence length.
        :param index: Tensor containing the index for masking.
        :param scores: Tensor of attention scores to be masked.
        :param device: Device to place the mask (e.g., "cpu" or "cuda").
        """
        # Create an upper triangular mask for attention scores
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)

        # Expand the mask to fit the batch and head dimensions
        _mask_expanded = _mask[None, None, :].expand(B, H, L, scores.shape[-1])

        # Gather mask based on the index
        indicator = _mask_expanded[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)

        # Reshape the mask to match the shape of scores
        self._mask = indicator.view_as(scores).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """
        Returns the generated probability mask.
        """
        return self._mask
