# ParaFilt
parafilt is a Python package that provides a collection of parallel adaptive filter implementations for efficient signal processing applications. It leverages the power of parallel processing using PyTorch, enabling faster and scalable computations on multi-core CPUs and GPUs.

## Features
- Parallel algorithm framework that allows computing iterative algorithms in a parallel way.
- Parallel implementation of popular adaptive filter algorithms, including LMS, NLMS, RLS, and more.
- Possibility for researchers to integrate their own adaptive filter algorithms for parallel computing.
- Comprehensive documentation and examples for quick start and usage guidance.

## Installation
To install Parafilt, you can use `pip`:
```
pip install parafilt
```

## Usage
Here's an example of how to use the package to create and apply the LMS filter:

```python
import parafilt

# Create an instance of the LMS filter
lms_filter = parafilt.LMS(hop=1024, framelen=4096, filterlen=1024).cuda()

# Perform parallel filter iteration
d_est, e = lms_filter(desired_signal, input_signal)
```


## Parallel Algorithm Framework
Parafilt provides a parallel algorithm framework that enables researchers to implement and execute iterative algorithms in a parallelized manner. This framework allows for efficient utilization of multi-core CPUs and GPUs, resulting in significant speedup for computationally intensive algorithms.

To leverage the parallel algorithm framework, researchers can extend the base classes provided by Parafilt and utilize the parallel computation capabilities provided by PyTorch.

## Documentation
For detailed usage instructions, examples, and API documentation, please refer to the documentation.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the GitHub repository.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.

## Contact
For any inquiries or questions, please contact zoreasaf@gmail.com.
