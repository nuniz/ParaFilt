from .base import BaseFilter
import torch
from typing import Optional


class TemplateFilter(BaseFilter):
    def __init__(self, hop: int, framelen: int, filterlen: int = 1024, weights_delay: Optional[int] = None,
                 weights_range: (float, float) = (-65535, 65535)):
        '''
        Template filter class that extends the BaseFilter class.
        :param hop: Hop size for frame processing.
        :param framelen: Length of each frame.
        :param filterlen: Length of the filter.
        :param weights_delay: Delay for the weights, If None, it is set to framelen - 1 (default: None).
        :param weights_range: Range for the filter weights (default: (-65535, 65535)).
        '''
        super().__init__(hop=hop, framelen=framelen, filterlen=filterlen, weights_delay=weights_delay,
                         weights_range=weights_range)

    @torch.no_grad()
    def forward_settings(self, d: torch.Tensor, x: torch.Tensor):
        '''
        Placeholder for the settings during forward.
        :param d: Desired signal tensor.
            Shape: (batch_size, frame_length)
        :param x: Input tensor.
            Shape: (batch_size, frame_length, filter_length)
        '''
        return

    @torch.no_grad()
    def iterate(self, d: torch.Tensor, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Placeholder for the filter iteration.
        :param d: Desired signal tensor.
            Shape: (batch_size, frame_length)
        :param x: Input tensor.
            Shape: (batch_size, frame_length, filter_length)
        :return:
            torch.Tensor: Estimated output tensor.
                Shape: (batch_size, frame_length)
            torch.Tensor: Error tensor.
                Shape: (batch_size, frame_length)
        '''
        raise NotImplementedError


class LMS(BaseFilter):
    def __init__(self, hop: int, framelen: int, filterlen: int = 1024, weights_delay: Optional[int] = None,
                 weights_range: (float, float) = (-65535, 65535), learning_rate: float = 0.01, normalized: bool = True):
        '''
        LMS filter class that extends the BaseFilter class.
        :param hop: Hop size for frame processing.
        :param framelen: Length of each frame.
        :param filterlen: Length of the filter (default: 1024).
        :param weights_delay: Delay for the weights, If None, it is set to framelen - 1 (default: None).
        :param learning_rate: Learning rate for the LMS algorithm (default: 0.01).
        :param normalized: Flag indicating whether to normalize the input energy (default: True).
        '''
        super().__init__(hop=hop, framelen=framelen, filterlen=filterlen, weights_delay=weights_delay,
                         weights_range=weights_range)
        self.learning_rate = learning_rate
        self.normalized = normalized

    @torch.no_grad()
    def iterate(self, d: torch.Tensor, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Performs one iteration of the LMS algorithm.
        :param d: Desired signal tensor.
        :param x: Input tensor.
        :return:
            (torch.Tensor, torch.Tensor): Tuple containing the estimated output and the error signal.
        '''
        d_est = self.run_filter(x)  # Estimated output using the current filter weights
        e = d - d_est  # Compute the error signal

        # Update the filter weights using the LMS update rule
        x_energy = 1 + self.normalized * (torch.einsum('ijk, ijk->j', x, x) - 1)
        self.w += self.learning_rate * e.unsqueeze(-1) * x / (
                torch.finfo(torch.float64).eps + x_energy.unsqueeze(0).unsqueeze(-1))
        return d_est, e


class RLS(BaseFilter):
    def __init__(self, hop: int, framelen: int, filterlen: int = 20, weights_delay: Optional[int] = None,
                 weights_range: (float, float) = (-65535, 65535), delta: float = 0.1, lmbd: float = 0.999):
        '''
        RLS filter class that extends the BaseFilter class.
        :param hop: Hop size for frame processing.
        :param framelen: Length of each frame.
        :param filterlen: Length of the filter (default: 20).
        :param weights_delay: Delay for the weights, If None, it is set to framelen - 1 (default: None).
        :param delta: Forgetting factor delta for the RLS algorithm (default: 0.1).
        :param lmbd: Lambda parameter for the RLS algorithm (default: 0.999).
        '''
        super().__init__(hop=hop, framelen=framelen, filterlen=filterlen, weights_delay=weights_delay,
                         weights_range=weights_range)
        self.delta = delta
        self.lmbd = lmbd
        self.register_buffer('inverse_correlation', self.delta * torch.eye(self.filterlen).unsqueeze(0).unsqueeze(0))

    @torch.no_grad()
    def reset(self):
        '''
        Reset the RLS filter weights and inverse correlation matrix.
        '''
        self.w *= 0
        self.inverse_correlation = self.delta * torch.eye(self.filterlen).unsqueeze(0).unsqueeze(0).to(
            self.inverse_correlation.device)

    @torch.no_grad()
    def iterate(self, d: torch.Tensor, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Perform a single iteration of the RLS algorithm.
        :param d: Desired signal tensor.
        :param x: Input signal tensor.
        :return:
            (torch.Tensor, torch.Tensor): Tuple containing the estimated output and the error signal.
        '''
        d_est = self.run_filter(x)  # Estimated output using the current filter weights
        e = d - d_est  # Compute the error signal

        # Update the filter weights using the RLS update rule
        g = torch.einsum('kli, klij -> kli', x, self.inverse_correlation)
        g /= (self.lmbd + torch.einsum('ijk, ijk -> ij', g, x).unsqueeze(-1))
        self.w += g * e.unsqueeze(-1)

        # Update inverse correlation matrix
        self.inverse_correlation -= torch.einsum('ijk, ijp -> ijkp',
                                                 torch.einsum('ijk, ijkl -> ijk', x, self.inverse_correlation), g)
        self.inverse_correlation /= self.lmbd
        return d_est, e

    @torch.no_grad()
    def forward_settings(self, d: torch.Tensor, x: torch.Tensor):
        '''
        Set the forward settings for the RLS filter.
        :param d: Desired signal tensor.
            Shape: (batch_size, frame_length)
        :param x: Input tensor.
            Shape: (batch_size, frame_length, filter_length)
        '''
        # Repeat inverse_correlation buffer along batch dimension
        self.inverse_correlation = self.inverse_correlation.repeat(d.shape[0], d.shape[1], 1, 1)
