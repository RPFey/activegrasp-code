# Code for ActiveNGF

## Setup

```bash
# Graspnet API
cd /tmp
git clone https://github.com/graspnet/graspnetAPI.git && cd graspnetAPI
sed -i 's/sklearn/scikit-learn/g' setup.py
sed -i 's/numpy==1.23.4/numpy<2.0.0/g' setup.py
python -m pip install Cython
python -m pip install .

# MinKowski Engine
cd /tmp
apt install build-essential python3-dev libopenblas-dev
python -m pip install ninja

git clone https://github.com/NVIDIA/MinkowskiEngine.git && cd MinkowskiEngine
sed -i '30i #include <thrust/execution_policy.h> \n' src/convolution_kernel.cuh
sed -i '33i #include <thrust/unique.h> \n#include <thrust/remove.h> \n' src/coordinate_map_gpu.cu
sed -i '29i #include <thrust/execution_policy.h> \n#include <thrust/reduce.h> \n#include <thrust/sort.h>' src/spmm.cu
sed -i '30i #include <thrust/execution_policy.h> \n' src/3rdparty/concurrent_unordered_map.cuh
CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" python -m pip install -v .
```