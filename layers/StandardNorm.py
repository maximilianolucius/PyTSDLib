import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            affine: bool = False,
            subtract_last: bool = False,
            non_norm: bool = False
    ):
        """
        Normalize module.

        :param num_features: Number of features or channels.
        :param eps: A value added for numerical stability.
        :param affine: If True, includes learnable affine parameters.
        :param subtract_last: If True, subtracts the last element instead of the mean.
        :param non_norm: If True, skips normalization.
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm

        if self.affine:
            # Initialize affine parameters
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Forward pass for normalization or denormalization.

        :param x: Input tensor.
        :param mode: 'norm' to normalize, 'denorm' to denormalize.
        :return: Normalized or denormalized tensor.
        """
        if self.non_norm:
            return x

        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented. Use 'norm' or 'denorm'.")

    def _get_statistics(self, x: torch.Tensor):
        """
        Compute and store the mean and standard deviation.

        :param x: Input tensor.
        """
        dim2reduce = tuple(range(1, x.ndim - 1))  # Typically, reduce over all dimensions except batch and channel
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1).detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.

        :param x: Input tensor.
        :return: Normalized tensor.
        """
        self._get_statistics(x)
        if self.subtract_last:
            x = x.sub_(self.last)
        else:
            x = x.sub_(self.mean)
        x = x.div_(self.stdev)
        if self.affine:
            x = x.mul_(self.affine_weight).add_(self.affine_bias)
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the input tensor.

        :param x: Normalized tensor.
        :return: Denormalized tensor.
        """
        if self.affine:
            x = x.sub_(self.affine_bias)
            x = x.div_(self.affine_weight + self.eps)  # Fixed potential bug by removing the extra eps multiplication
        x = x.mul_(self.stdev)
        if self.subtract_last:
            x = x.add_(self.last)
        else:
            x = x.add_(self.mean)
        return x
