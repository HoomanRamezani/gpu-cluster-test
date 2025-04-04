# Use NVIDIA NGC PyTorch container 
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set working directory in container
WORKDIR /workspace

# Ensure TorchVision is available for datasets
RUN python -c "import torchvision" || pip install --no-cache-dir torchvision

# Copy the training script into the container
COPY train.py .

# Create directory for reports
RUN mkdir -p /workspace/reports

# Set default command to run a short training (2 epochs) with report generation
CMD ["python", "train.py", "--epochs", "2", "--batch-size", "64"]

# Label the image
LABEL maintainer="Cloud/Infrastructure" \
      description="GPU Cluster Acceptance Testing with PyTorch" \
      version="1.0" 