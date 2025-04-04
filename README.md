# GPU Cluster Acceptance Testing

This repository provides a portable, lightweight internal test harness for GPU cluster acceptance. It builds a Docker container that performs distributed training using PyTorch DDP on the CIFAR-10 dataset to verify multi-GPU compute environments.

## Features
- **GPU Validation**: Checks CUDA availability, tensor ops, and NCCL communication via distributed training  
- **Image**: Builds a portable container based on nvcr.io/nvidia/pytorch:24.07-py3 that runs the GPU validation test  
- **CI/CD**: Includes GitHub Actions workflow to build, test, and push the image to GHCR on success

## What It Tests

1. **CUDA Device Check**: Ensures GPUs are available via `torch.cuda`
2. **Tensor Performance**: Measures execution time of `torch.matmul` on GPU
3. **NCCL Communication**: Tests `all_gather` across GPUs (when available)
4. **Training Test**: Runs 2 epochs of ResNet18 on CIFAR-10 with DDP
5. **Diagnostics**: Logs temperature, memory, power, utilization, and topology via `nvidia-smi`

## Quick Start

### 1. **Run from Pre-Built Image** (recommended)
```bash
docker run --rm --gpus all --ipc=host \
  -v $(pwd)/reports:/workspace/reports \
  ghcr.io/HoomanRamezani/gpu-cluster-test:latest
```
> Pulls the validated image from GHCR and runs the distributed GPU test.

---
### 2. **Build and Test Locally** (for development/debugging)
```bash
docker build -t gpu-cluster-test:latest .

docker run --rm --gpus all --ipc=host \
  -v $(pwd)/reports:/workspace/reports \
  gpu-cluster-test:latest
```
> Builds the container locally using `nvcr.io/nvidia/pytorch:24.07-py3` and runs the test. 



## CI/CD: Automated Test and Push
1. Builds the Docker image using docker build
2. Runs the container on available GPU hardware and mounts a local reports/ volume
3. Validates test success by checking `cluster_test_report.json`
4. If `"status": "success"`, pushes image to `ghcr.io`


## Sample Output Report (`cluster_test_report.json`)

```json
{
  "test_timestamp": "2025-04-04 03:52:05",
  "world_size": 1,
  "validation": {
    "rank": 0,
    "status": "passed",
    "checks": {
      "gpu_available": true,
      "gpu_name": "Tesla T4",
      "gpu_memory_total": 15828320256,
      "cuda_version": "12.4",
      "tensor_op_time": 0.347,
      "tensor_op_success": true
    },
    "errors": [],
    "nvidia_smi": {
      "gpus": [
        {
          "index": 0,
          "name": "Tesla T4",
          "driver_version": "550.54.15",
          "temperature_gpu": "55",
          "utilization_gpu": "3 %",
          "memory_used": "134 MiB",
          "memory_total": "15360 MiB",
          "power_draw": "27.56 W"
        }
      ]
    }
  },
  "performance": {
    "rank": 0,
    "epochs": [
      {
        "epoch": 1,
        "avg_loss": 1.58,
        "time_seconds": 22.47
      },
      {
        "epoch": 2,
        "avg_loss": 1.23,
        "time_seconds": 20.38
      }
    ],
    "final_status": "success",
    "model_init_time": 0.89,
    "data_load_time": 6.95
  },
  "status": "success",
  "acceptance": {
    "status": "PASS",
    "summary": [
      "GPU Available: YES",
      "NCCL Communication: N/A",
      "Training Status: PASS"
    ]
  }
}
```
