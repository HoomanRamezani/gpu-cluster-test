# Distributed GPU Training Example for Cluster Acceptance Testing

This repository provides a lightweight and portable example of distributed GPU training using PyTorch. It is designed for Cloud/Infrastructure engineers to validate a GPU cluster's multi-GPU training capabilities. 

## Features

- **Standard Model & Dataset**: Uses a ResNet-18 model (from TorchVision) on the CIFAR-10 dataset – a well-known small image classification benchmark
- **Distributed Data Parallel Training**: Implements PyTorch's native DDP, which launches one training process per GPU and synchronizes gradients across them
- **NVIDIA PyTorch Base Image**: Uses `nvcr.io/nvidia/pytorch:24.07-py3` as the base Docker image, with PyTorch, CUDA, cuDNN, and NCCL pre-installed
- **CI Workflow with GitHub Actions**: Includes an automated GitHub Actions workflow to build the Docker image, run a short multi-GPU training test, and push the image to GitHub Container Registry (GHCR)
- **Portable & Simple Design**: Self-contained code with minimal dependencies that can be easily extended for different cluster configurations
- **NVIDIA-SMI Logging**: Automatically captures GPU information via `nvidia-smi` before and after training in structured JSON format
- **Clear Acceptance Status**: Provides a prominent PASS/FAIL determination at the top of the output

## Acceptance Test Validation Criteria

The script includes automated checks to validate GPU cluster functionality across all GPUs:

### Key Validation Tests

1. **GPU Availability & Properties**
   - Verifies each GPU is accessible via CUDA
   - Reports GPU model, memory capacity, and CUDA version

2. **Computation Performance**
   - Tests basic tensor operations on each GPU
   - Measures time taken for matrix multiplication

3. **Inter-GPU Communication (NCCL)**
   - Tests communication between GPUs via all_gather operations
   - Verifies distributed synchronization works correctly
   - Measures communication latency

4. **Distributed Training**
   - Tests model initialization and data loading
   - Runs actual training for configurable number of epochs
   - Measures training throughput and loss convergence
   
5. **NVIDIA-SMI Diagnostics**
   - Captures GPU name, driver version, temperature, utilization
   - Logs memory usage and power draw 
   - Records GPU topology for multi-GPU setups
   - Compares before/after metrics to detect issues

### Acceptance Criteria

The cluster passes validation if:

- All GPUs are detected and accessible
- Tensor operations complete successfully on all GPUs
- NCCL operations complete without errors
- Training loss decreases over epochs
- No GPU errors or out-of-memory conditions occur
- Inter-GPU communication latency is within acceptable bounds
- NVIDIA-SMI reports normal temperature and utilization profiles

A comprehensive report is generated in JSON format (`reports/cluster_test_report.json`) with detailed metrics and validation results.

## Quick Start

### Pull and Run the Container

```bash
# Pull the pre-built container
docker pull ghcr.io/OWNER/REPO:latest

# Run with GPU access
docker run --rm --gpus all --ipc=host ghcr.io/OWNER/REPO:latest
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/OWNER/REPO.git
cd REPO

# Build the Docker image
docker build -t gpu-cluster-test:latest .

# Run with GPU access
docker run --rm --gpus all --ipc=host gpu-cluster-test:latest
```

### Run Validation Only

```bash
# Run just the validation tests without full training
docker run --rm --gpus all --ipc=host gpu-cluster-test:latest python train.py --validation-only
```

## Sample Output

The script displays a clear acceptance status at the top of the output:

```
==================================================
FINAL ACCEPTANCE: PASS
- GPU Available: YES
- NCCL Communication: PASS
- Training Status: PASS
==================================================
```

Or in case of failure:

```
==================================================
FINAL ACCEPTANCE: FAIL
- GPU Available: YES
- NCCL Communication: PASS
- Training Status: FAIL
- Error: CUDA out of memory
==================================================
```

The JSON report also includes this acceptance information:

```json
{
  "test_timestamp": "2023-04-01 12:34:56",
  "world_size": 4,
  "validation": { /*validation details*/ },
  "performance": { /*performance details*/ },
  "status": "success",
  "acceptance": {
    "status": "PASS",
    "summary": [
      "GPU Available: YES",
      "NCCL Communication: PASS",
      "Training Status: PASS"
    ]
  }
}
```

## Implementation Details

### Distributed Training

The training script (`train.py`) uses PyTorch's Distributed Data Parallel (DDP) to utilize multiple GPUs:

- Each GPU runs a separate process with its own copy of the model
- Gradient synchronization happens automatically during backpropagation
- The CIFAR-10 dataset is partitioned across processes using `DistributedSampler`
- NCCL is used as the communication backend for optimal GPU-to-GPU performance

### Container Image

The Docker image is based on NVIDIA's PyTorch container:

- Includes PyTorch, CUDA, cuDNN, and NCCL pre-installed and optimized
- Minimal additional dependencies required (only checks for torchvision)
- Optimized for multi-GPU training out of the box

### Kubernetes Deployment

The included Kubernetes manifest (`k8s-deployment.yaml`) shows how to deploy the training job in a Kubernetes environment:

```bash
# Deploy the job to a Kubernetes cluster with GPU support
kubectl apply -f k8s-deployment.yaml

# Monitor the training job
kubectl get jobs
kubectl logs -f job/gpu-cluster-training
```

## Interpreting NVIDIA-SMI Output

The test captures two NVIDIA-SMI snapshots:
1. **Before training**: Baseline GPU state
2. **After training**: Final GPU state after workload

Key metrics to examine:
- **Temperature**: Should be within normal range (usually below 85°C)
- **Memory usage**: Training should use significant but not all GPU memory
- **Utilization**: Should show high utilization during training
- **GPU Topology**: Shows interconnect type between GPUs (important for multi-GPU performance)

## Customization

You can customize the training by passing arguments:

```bash
# Run for more epochs with larger batch size
docker run --rm --gpus all gpu-cluster-test:latest python train.py --epochs 5 --batch-size 128
```

For more complex configurations, modify the `train.py` file and rebuild the container.

## License

MIT 