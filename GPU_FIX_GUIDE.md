# GPU Segfault Fix Guide for CATBOSS

## Understanding the Problem

Your GPU passes initial tests but segfaults during actual processing. This guide will help you diagnose and fix the issue.

## Step 1: Run the Diagnostic Script

```bash
cd /home/user/catboss
python test_gpu.py
```

## Step 2: Identify Which Test Fails

### If Test 5 fails (Second memory query):
**Problem**: Your CUDA driver has a bug where repeated context access crashes.

**Fix Options**:
1. **Update NVIDIA driver**:
   ```bash
   # Check current driver version
   nvidia-smi

   # Update to latest driver (Ubuntu/Debian)
   sudo apt update
   sudo apt install nvidia-driver-535  # or latest version
   sudo reboot
   ```

2. **Reinstall CUDA Toolkit**:
   ```bash
   # Remove old CUDA
   sudo apt remove --purge cuda*
   sudo apt autoremove

   # Install fresh CUDA 11.8 (compatible with most systems)
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt update
   sudo apt install cuda
   sudo reboot
   ```

3. **Reinstall numba with CUDA support**:
   ```bash
   pip uninstall numba
   pip install --upgrade numba
   ```

### If Test 6 fails (CUDA stream creation):
**Problem**: CUDA stream creation is broken.

**Fix**:
```bash
# Reinstall numba and cudatoolkit
conda install -c conda-forge numba cudatoolkit=11.8
```

### If Test 11 fails (ThreadPoolExecutor):
**Problem**: Multi-threading with CUDA is broken.

**Workaround**: This is a known CUDA limitation. Use single-threaded mode:
```bash
export CATBOSS_NO_GPU=1  # Force CPU mode
# OR
catboss --cat pooh data.ms --max-threads 1  # Single thread
```

## Step 3: Common Fixes

### Fix 1: Check CUDA Environment Variables
```bash
# Add to ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Reload
source ~/.bashrc
```

### Fix 2: Check GPU Power/Thermal
Your GPU might be overheating or power-throttling:
```bash
# Monitor GPU while running test
nvidia-smi -l 1

# Check for thermal throttling or power issues
nvidia-smi -q -d POWER,TEMPERATURE
```

### Fix 3: Disable GPU Persistence Mode (sometimes helps)
```bash
sudo nvidia-smi -pm 0
```

### Fix 4: Force CPU Mode (if all else fails)
```bash
export CATBOSS_NO_GPU=1
catboss --cat pooh data.ms ...
```

## Step 4: Verify Fix

After applying a fix, run:
```bash
python test_gpu.py
```

All tests should pass. Then try catboss again.

## Debug Information to Share

If tests still fail, share this info:

```bash
# System info
uname -a
lspci | grep -i nvidia

# Driver info
nvidia-smi

# CUDA info
nvcc --version

# Python/Numba info
python -c "import numba; print(numba.__version__)"
python -c "from numba import cuda; print(cuda.detect())"

# Run test and save output
python test_gpu.py 2>&1 | tee gpu_test_output.txt
```

## Understanding the Segfault

The segfault happens because:

1. **First CUDA call works**: Initial `cuda.current_context()` succeeds
2. **Second CUDA call crashes**: Repeated access segfaults
3. **This suggests**: Driver bug, memory corruption, or incompatible CUDA/driver versions

The most common cause is **mismatched CUDA toolkit and NVIDIA driver versions**.

## Quick Check: Version Compatibility

```bash
# Check driver version
nvidia-smi | grep "Driver Version"

# Check CUDA version
nvcc --version

# They should be compatible:
# Driver 520+ → CUDA 11.8
# Driver 535+ → CUDA 12.0
```

## Still Broken?

If nothing works, your GPU hardware might be faulty. Test with:
```bash
# Run CUDA stress test
git clone https://github.com/wili-yu/gpu-burn.git
cd gpu-burn
make
./gpu_burn 60  # Run for 60 seconds

# If this crashes, hardware is faulty
```
