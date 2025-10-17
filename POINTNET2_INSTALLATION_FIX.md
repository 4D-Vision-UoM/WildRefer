# PointNet2 Extension Compilation Fix Documentation

## Problem Overview

When running the WildRefer test script, the following error was encountered:

```
ModuleNotFoundError: No module named 'pointnet2._ext'
ImportError: Could not import _ext module.
```

This error occurred because the PointNet2 C++/CUDA extension module was not properly compiled and installed.

## Root Causes

The compilation failure was due to multiple compatibility issues:

1. **Missing source files**: The `_ext_src/src/` directory was missing `.cpp` and `.cu` files
2. **PyTorch version mismatch**: PyTorch 2.4.1+cu121 was installed instead of required 1.11.0+cu113
3. **CUDA version mismatch**: CUDA 12.1 vs required CUDA 11.3
4. **GCC version incompatibility**: System GCC 13.3.0 was too new for CUDA 11.3 (max supported: GCC 10)
5. **Missing system headers**: `crypt.h` header was missing
6. **Linker issues**: Conda's `compiler_compat` directory caused linking problems with system libraries

## Solution Steps

### Step 1: Obtain Missing Source Files

The PointNet2 source files were missing from `_ext_src/src/`. Downloaded from the original repository:

```bash
cd /home/avishka/sasika/WildRefer/pointnet2/_ext_src
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git temp_pointnet2

# Copy source files to correct location
mkdir -p src
cp -r temp_pointnet2/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/* src/

# Clean up
rm -rf temp_pointnet2
```

**Files copied:**
- `ball_query.cpp` and `ball_query_gpu.cu`
- `bindings.cpp`
- `group_points.cpp` and `group_points_gpu.cu`
- `interpolate.cpp` and `interpolate_gpu.cu`
- `sampling.cpp` and `sampling_gpu.cu`

### Step 2: Install Correct PyTorch Version

Uninstalled incompatible PyTorch 2.4.1 and installed required version:

```bash
conda activate wildrefer_env

# Uninstall current PyTorch
pip uninstall torch torchvision -y

# Install PyTorch 1.11.0 with CUDA 11.3 support
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

**Verification:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
# Output: PyTorch: 1.11.0+cu113, CUDA: 11.3
```

### Step 3: Install Compatible GCC Compiler

CUDA 11.3 requires GCC ≤ 10, but system had GCC 13.3.0. Installed GCC 10 via conda:

```bash
conda activate wildrefer_env
conda install -c conda-forge gxx=10.4.0 -y
```

**Verification:**
```bash
gcc --version
# Output: gcc (conda-forge gcc 10.4.0-19) 10.4.0
```

### Step 4: Install Missing Dependencies

Install `libxcrypt` to provide the missing `crypt.h` header:

```bash
conda activate wildrefer_env
conda install -c conda-forge libxcrypt -y
```

### Step 5: Resolve Linker Issues

The conda `compiler_compat` directory interfered with system library linking. Solution:

```bash
cd /home/avishka/anaconda3/envs/wildrefer_env
mv compiler_compat compiler_compat.bak

# Compile with explicit library paths
cd /home/avishka/sasika/WildRefer/pointnet2
rm -rf build dist pointnet2.egg-info
LDFLAGS="-L/lib/x86_64-linux-gnu" python setup.py install

# Restore compiler_compat
cd /home/avishka/anaconda3/envs/wildrefer_env
mv compiler_compat.bak compiler_compat
```

### Step 6: Configure Environment Variables

The compiled extension requires PyTorch's library path in `LD_LIBRARY_PATH`:

```bash
# Create activation script directory
mkdir -p /home/avishka/anaconda3/envs/wildrefer_env/etc/conda/activate.d

# Create activation script
cat > /home/avishka/anaconda3/envs/wildrefer_env/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH}"
EOF

chmod +x /home/avishka/anaconda3/envs/wildrefer_env/etc/conda/activate.d/env_vars.sh
```

## Verification

Test that the module imports successfully:

```bash
conda activate wildrefer_env
python -c "import pointnet2._ext as _ext; print('PointNet2 extension imported successfully!')"
```

Expected output: `PointNet2 extension imported successfully!`

## Final Test

Run the original test command:

```bash
conda activate wildrefer_env
cd /home/avishka/sasika/WildRefer
python test.py --dataset liferefer --pretrain weights/liferefer_weights.pth --frame_num 2 --batch_size 32
```

The PointNet2 import error should now be resolved.

## Key Takeaways

1. **Version Compatibility**: Ensure PyTorch, CUDA, and GCC versions are compatible
2. **Complete Source Files**: Verify all required source files exist before compilation
3. **Compiler Compatibility**: CUDA has strict GCC version requirements
4. **Library Paths**: CUDA extensions need proper runtime library paths
5. **Conda Environments**: The `compiler_compat` directory can cause linking issues

## Environment Summary

**Working Configuration:**
- Python: 3.8
- PyTorch: 1.11.0+cu113
- CUDA: 11.3
- GCC: 10.4.0 (conda-forge)
- OS: Ubuntu 24.04 (Linux 6.14.0-33-generic)

## References

- [PointNet2 PyTorch Repository](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [PyTorch Installation Guide](https://pytorch.org/get-started/previous-versions/)
- [CUDA Compatibility Matrix](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

---

**Document Created:** October 17, 2025  
**Last Updated:** October 17, 2025

