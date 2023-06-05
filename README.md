# ParaFilt
ParaFilt is a Python package that provides a collection of parallel adaptive filter implementations for efficient signal processing applications. It leverages the power of parallel processing using PyTorch, enabling faster and scalable computations on multi-core CPUs and GPUs.

## Features
- Parallel algorithm framework that allows computing iterative algorithms in a parallel way.
- Parallel implementation of popular adaptive filter algorithms, including LMS, NLMS, RLS, and more.
- Possibility for researchers to integrate their own adaptive filter algorithms for parallel computing.
- Comprehensive documentation and examples for quick start and usage guidance.

## Installation
To install ParaFilt, you can use `pip`:
```
pip install parafilt
```

## Usage
Inputs:

    desired_signal: (batch_size, samples) - Desired signal tensor.
    input_signal: (samples) - Input tensor.

Returns:

    Tuple containing the estimated output and the error signal (d_est, e).
	d_est: (batch_size, samples) - Estimated output tensor.
    e: (batch_size, samples) - Error signal tensor.


Here's an example of how to use the package to create and apply the LMS filter:
```python
import parafilt

# Create an instance of the LMS filter
lms_filter = parafilt.LMS(hop=1024, framelen=4096, filterlen=1024).cuda()

# Perform parallel filter iteration
d_est, e = lms_filter(desired_signal, input_signal)
```

Here's an example of how to use the package to create and apply the RLS filter:
```python
import parafilt

# Create an instance of the LMS filter
rls_filter = parafilt.RLS(hop=1024, framelen=4096, filterlen=1024).cuda()

# Perform parallel filter iteration
d_est, e = rls_filter(desired_signal, input_signal)
```

For detailed usage example, please refer to this [notebook](https://github.com/nuniz/ParaFilt/blob/main/notebooks/1.0-parafilt-demo.ipynb).

## Parallel Algorithm Framework
Parafilt provides a parallel algorithm framework that enables researchers to implement and execute iterative algorithms in a parallelized manner. This framework allows for efficient utilization of multi-core CPUs and GPUs, resulting in significant speedup for computationally intensive algorithms.

To leverage the parallel algorithm framework, researchers can extend the base classes provided by Parafilt and utilize the parallel computation capabilities provided by PyTorch.

Here's an example of how to use the package to create your own filter:
```python
from parafilt import BaseFilter

class TemplateFilter(BaseFilter):
    def __init__(self, hop: int, framelen: int, filterlen: int = 1024, weights_delay: Optional[int] = None, 
	weights_range: (float, float) = (-65535, 65535)):
        '''
        Template filter class that extends the BaseFilter class.
        :param hop: Hop size for frame processing.
        :param framelen: Length of each frame.
        :param filterlen: Length of the filter.
        :param weights_delay: Delay for the weights, If None, it is set to framelen-1 (default: None).
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
            Shape: (1, frame_length, filter_length)
        '''
        return

    @torch.no_grad()
    def iterate(self, d: torch.Tensor, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Placeholder for the filter iteration.
        :param d: Desired signal tensor.
            Shape: (batch_size, frame_length)
        :param x: Input tensor.
            Shape: (1, frame_length, filter_length)
        :return:
            torch.Tensor: Estimated output tensor.
                Shape: (batch_size, frame_length)
            torch.Tensor: Error tensor.
                Shape: (batch_size, frame_length)
        '''
        raise NotImplementedError
```

## Future Work
- Implementation of CUDA code for the parallel frameworks and filter algorithms to achieve even faster computations.
- Addition of an option for zero-padding, enabling the output size to match the input size without discarding any samples during the frame decomposition and reconstruction process after performing the filter.

## Citation
[![DOI](https://zenodo.org/badge/648500046.svg)](https://zenodo.org/badge/latestdoi/648500046)

```
@software{asaf_zorea_2023_8004059,
  author       = {Asaf Zorea},
  title        = {nuniz/ParaFilt: zenodo upload},
  month        = jun,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.1.2-beta},
  doi          = {10.5281/zenodo.8004059},
  url          = {https://doi.org/10.5281/zenodo.8004059}
}
```

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the GitHub repository.

## License
This project is licensed under the MIT License. See the [LICENSE file](https://github.com/nuniz/ParaFilt/blob/main/LICENSE) for more information.

## Contact
For any inquiries or questions, please contact zoreasaf@gmail.com.

