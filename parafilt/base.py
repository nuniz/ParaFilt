import torch
from typing import Optional


class BaseFilter(torch.nn.Module):
    def __init__(self, hop: int, framelen: int, filterlen: int, weights_delay: Optional[int] = None,
                 weights_range: (float, float) = (-65535, 65535)):
        '''
        Base class for a filter module.
        :param hop: Hop size for frame processing.
        :param framelen: Length of each frame.
        :param filterlen: Length of the filter.
        :param weights_delay: Delay for the weights, If None, it is set to framelen/2 (default: None).
        :param weights_range: Range for the filter weights (default: (-65535, 65535)).
        '''
        super(BaseFilter, self).__init__()

        # Validate and set filter length
        assert filterlen > 0, f'filter_length must be bigger than zero, but obtained {filterlen}'
        self.filterlen = filterlen
        self.register_buffer('w', torch.zeros(1, 1, filterlen))

        # Validate and set hop size
        assert hop > 0, 'hop must be larger than zero'
        self.hop = hop

        # Validate and set frame length
        assert framelen >= hop, 'framelen must be larger than hop'
        assert framelen > filterlen, 'framelen must be larger than filterlen'
        self.framelen = framelen

        # Validate and set weights delay
        assert weights_delay is None or 0 <= weights_delay < filterlen, \
            f'delay must be between 0 to {filterlen} (filterlen), but obtained {weights_delay}'
        self.weights_delay = filterlen // 2 if weights_delay is None else weights_delay
        self.weights_range = weights_range

    def reset(self):
        '''
        Reset the filter weights.
        '''
        self.w *= 0

    def run_filter(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply the filter to the input tensor.
        :param x: Input tensor.
        :return:
            torch.Tensor: Filtered output tensor.
        '''
        return torch.einsum('ijk, pjk->ij', self.w, x)

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

    def forward_settings(self, d: torch.Tensor, x: torch.Tensor):
        '''
        Placeholder for the settings during forward.
        :param d: Desired signal tensor.
        :param x: Input tensor.
        '''
        return

    def forward(self, d: torch.Tensor, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Apply the filter to the input signals.
        :param d: Desired signal tensor.
        :param x: Input tensor.
        :return:
            torch.Tensor: Estimated output tensor.
            torch.Tensor: Error tensor.
        '''
        self.reset()

        # Input validation
        assert d.ndim == 2, f'd ndim must be 2, but obtained {d.ndim}'
        assert x.ndim == 1, f'x ndim must be 1, but obtained {x.ndim}'

        # Unfold input tensors
        d = d.unfold(dimension=-1, size=self.framelen, step=self.hop)
        x = x.unfold(dimension=-1, size=self.framelen, step=self.hop).unsqueeze(0)

        # Forward additional settings
        self.forward_settings(d, x)

        # Initialize intermediate tensors
        d_est = torch.zeros_like(d)[..., self.weights_delay: -(self.filterlen - self.weights_delay)]
        e = torch.zeros_like(d)[..., self.weights_delay: -(self.filterlen - self.weights_delay)]

        # Repeat weight buffer along batch dimension
        self.w = self.w.repeat(d.shape[0], d.shape[1], 1)

        # Iterate over samples
        for i in range(self.weights_delay, e.shape[-1]):
            d_est[..., i], e[..., i] = \
                self.iterate(d[..., i], x[..., i - self.weights_delay: i - self.weights_delay + self.filterlen])
            self.w = torch.clip(self.w, min=self.weights_range[0], max=self.weights_range[1])

        # Concatenate and reshape intermediate tensors
        d_est = torch.cat((d_est[:, 0, :], d_est[:, 1:, self.framelen - self.hop:].reshape(d_est.shape[0], -1)), -1)
        e = torch.cat((e[:, 0, :], e[:, 1:, self.framelen - self.hop:].reshape(e.shape[0], -1)), -1)

        return d_est, e
