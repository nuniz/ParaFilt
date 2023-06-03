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
        :param weights_delay: Delay for the weights, If None, it is set to framelen/2 (default: None).
        :param weights_range: Range for the filter weights (default: (-65535, 65535)).
        '''
        super().__init__(hop=hop, framelen=framelen, filterlen=filterlen, weights_delay=weights_delay,
                         weights_range=weights_range)

    def iterate(self, d: torch.Tensor, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Placeholder method for the filter iteration.
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
        :param weights_delay: Delay for the weights, If None, it is set to framelen/2 (default: None).
        :param learning_rate: Learning rate for the LMS algorithm (default: 0.01).
        :param normalized: Flag indicating whether to normalize the input energy (default: True).
        '''
        super().__init__(hop=hop, framelen=framelen, filterlen=filterlen, weights_delay=weights_delay,
                         weights_range=weights_range)
        self.learning_rate = learning_rate
        self.normalized = normalized

    def iterate(self, d: torch.Tensor, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Performs one iteration of the LMS algorithm.
        :param d: Desired signal tensor.
        :param x: Input tensor.
        :return:
            (torch.Tensor, torch.Tensor): Tuple containing the estimated output and the error signal.
        '''
        d_est = self.filt(x)  # Estimated output using the current filter weights
        e = d - d_est  # Compute the error signal

        # Update the filter weights using the LMS update rule
        x_energy = 1 + self.normalized * (torch.einsum('ijk, ijk->j', x, x) - 1)
        self.w += self.learning_rate * e.unsqueeze(-1) * x / (
                torch.finfo(torch.float64).eps + x_energy.unsqueeze(0).unsqueeze(-1))
        return d_est, e


class RLS(BaseFilter):
    def __init__(self, hop: int, framelen: int, filterlen: int = 1024, weights_delay: Optional[int] = None,
                 weights_range: (float, float) = (-65535, 65535), forgetting_factor: float = 1,
                 inverse_cc_init: float = 1.001):
        super().__init__(hop=hop, framelen=framelen, filterlen=filterlen, weights_delay=weights_delay,
                         weights_range=weights_range)
        self.forgetting_factor = forgetting_factor
        self.inverse_cc_init = inverse_cc_init
        self.register_buffer('inverse_cc', inverse_cc_init * torch.eye(filterlen).unsqueeze(0))

    def reset(self):
        self.w *= 0
        self.inverse_cc = self.inverse_cc_init * torch.eye(self.filterlen).unsqueeze(0).to(self.inverse_cc.device)

    def iterate(self, d: torch.Tensor, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        d_est = self.filt(x)
        e = d - d_est
        g = torch.einsum('ijk, lmk')
        # torch.matmul(x, self.inverse_cc) / (
        #         self.forgetting_factor + torch.matmul(torch.matmul(x, self.inverse_cc), x.permute(0, 2, 1)))
        self.w += g * e

        # Update inverse correlation matrix
        self.inverse_cc = 1 / self.forgetting_factor * self.inverse_cc - e * g
        return d_est, e
