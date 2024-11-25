# Needle++: A Deep Learning Framework

A deep learning framework implemented from scratch, featuring automatic differentiation, neural network components, and support for both CPU and CUDA backends.

## Project Structure

```
.
├── CMakeLists.txt           # CMake configuration for C++ backend
├── Makefile                 # Build automation
├── apps/                    # Application examples
│   ├── models.py           # Model implementations
│   └── simple_ml.py        # Simple machine learning examples
├── python/
│   └── needle/             # Main framework package
│       ├── autograd.py     # Automatic differentiation engine
│       ├── backend_ndarray/# NDArray implementation
│       │   ├── ndarray.py  # NDArray interface
│       │   └── ndarray_backend_numpy.py
│       ├── backend_numpy.py# NumPy backend implementation
│       ├── backend_selection.py
│       ├── data/           # Data loading and processing
│       │   ├── data_basic.py
│       │   ├── data_transforms.py
│       │   └── datasets/   # Dataset implementations
│       │       ├── cifar10_dataset.py
│       │       ├── mnist_dataset.py
│       │       ├── ndarray_dataset.py
│       │       └── ptb_dataset.py
│       ├── init/          # Parameter initialization
│       │   ├── init_basic.py
│       │   └── init_initializers.py
│       ├── nn/            # Neural network modules
│       │   ├── nn_basic.py
│       │   ├── nn_conv.py
│       │   ├── nn_sequence.py
│       │   └── nn_transformer.py
│       ├── ops/           # Mathematical operations
│       │   ├── ops_logarithmic.py
│       │   ├── ops_mathematic.py
│       │   └── ops_tuple.py
│       └── optim.py       # Optimization algorithms
└── src/                   # C++/CUDA backend implementation
    ├── ndarray_backend_cpu.cc
    └── ndarray_backend_cuda.cu
```

## Features

- **Automatic Differentiation**: Reverse-mode automatic differentiation implementation
- **Multiple Backends**:
  - NumPy backend for CPU operations
  - CUDA backend for GPU acceleration
- **Neural Network Components**: Basic neural network modules, optimizers, and loss functions

## Requirements

- Python 3.8+
- NumPy

## Installation

1. Create a new conda environment:

```bash
conda create --name <env> --file <this file>
```

2. Clone the repository:

```bash
git clone https://github.com/DavidBao03/Needle-plusplus.git
```

3. Navigate to the project directory:

```bash
cd Needle-plusplus
```

4. Update `CMakeLists.txt` to specify the Python path in your conda environment:

```cmake
set(PYTHON_EXECUTABLE <path_to_your_conda_env_python>)
```

## Quick Start

To get started quickly, you can also explore the `demo.ipynb` notebook, which provides an interactive demonstration of the framework in action.

## Major Components

### NDArray Backend

- Efficient array operations implementation
- Support for both CPU and GPU computation
- NumPy-compatible interface

### Automatic Differentiation

- Dynamic computational graph construction
- Support for complex neural network architectures
- Efficient gradient computation

### Neural Network Modules

- Basic layers (Linear, Conv2d)
- Advanced architectures (Transformer)
- Activation functions
- Loss functions

### Data Processing

- Efficient data loading
- Data transformations
- Support for common datasets

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## TODO

- [ ] Fixed numerical problems on auto gradient
- [ ] Support more than one dims ops (e.g., sum, max)
- [ ] Implement more neural network layers (e.g., Vit, Bert)
- [ ] Add more dataset loaders (e.g., ImageNet)
- [ ] Improve CUDA backend performance
  - [ ] Speed up DGEMM operations
  - [ ] Implement kernel fusion
  - [ ] Integrate flash attention
- [ ] Expand test coverage for all modules

## Acknowledgments

This project is inspired by the concepts covered in the [Deep Learning Systems Course](https://dlsyscourse.org/lectures/) and the open-source GitHub repository [Needle](https://github.com/YconquestY/Needle).

