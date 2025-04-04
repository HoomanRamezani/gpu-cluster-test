#!/usr/bin/env python3
import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import json
import sys
import subprocess

def log_nvidia_smi(rank):
    """Log basic GPU information using nvidia-smi"""
    smi_info = {}
    
    try:
        # Only rank 0 logs to avoid duplicate info
        if rank != 0:
            return {}
            
        # Try to run nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw", 
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        
        if result.returncode == 0:
            # Store raw output for reference
            smi_info["raw_output"] = result.stdout
            
            # Parse CSV output into structured data
            gpus = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                values = [v.strip() for v in line.split(',')]
                if len(values) >= 8:
                    gpu_info = {
                        "index": int(values[0]),
                        "name": values[1],
                        "driver_version": values[2],
                        "temperature_gpu": values[3],
                        "utilization_gpu": values[4],
                        "memory_used": values[5],
                        "memory_total": values[6],
                        "power_draw": values[7]
                    }
                    gpus.append(gpu_info)
            
            smi_info["gpus"] = gpus
            
            # Print formatted output to console
            print("\nNVIDIA-SMI GPU Information:")
            for gpu in gpus:
                print(f"GPU {gpu['index']} ({gpu['name']}):")
                print(f"  Driver: {gpu['driver_version']}")
                print(f"  Temperature: {gpu['temperature_gpu']}")
                print(f"  Utilization: {gpu['utilization_gpu']}")
                print(f"  Memory: {gpu['memory_used']} / {gpu['memory_total']}")
                print(f"  Power: {gpu['power_draw']}")
                print("")
            
            # Also get GPU topology information if multiple GPUs
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                topo_result = subprocess.run(
                    ["nvidia-smi", "topo", "-m"],
                    capture_output=True, text=True
                )
                if topo_result.returncode == 0:
                    smi_info["gpu_topology"] = topo_result.stdout
                    
                    # Parse topology into matrix
                    topo_lines = topo_result.stdout.strip().split('\n')
                    topo_matrix = []
                    
                    # Extract connectivity data
                    for line in topo_lines:
                        if line.startswith('GPU'):
                            # Skip header line that just has GPU indices
                            if not line.startswith('GPU0'):
                                parts = line.split()
                                if len(parts) > 1:
                                    # First part is the GPU ID (e.g., "GPU0")
                                    # Rest are connectivity types
                                    topo_matrix.append(parts[1:])
                    
                    smi_info["topology_matrix"] = topo_matrix
                    
                    print("\nGPU Topology:")
                    print(topo_result.stdout)
    
    except Exception as e:
        print(f"Warning: nvidia-smi logging failed: {str(e)}")
        smi_info["error"] = str(e)
    
    return smi_info

def setup_process(rank, world_size):
    """Initialize the distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use NCCL backend for NVIDIA GPUs for efficient communication
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_process():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def validate_gpu_setup(rank, world_size):
    """Perform validation checks to verify the GPU cluster setup."""
    validation_results = {
        "rank": rank,
        "status": "passed",
        "checks": {},
        "errors": []
    }
    
    # Check 1: GPU availability and properties
    try:
        gpu_available = torch.cuda.is_available()
        validation_results["checks"]["gpu_available"] = gpu_available
        
        if gpu_available:
            validation_results["checks"]["gpu_name"] = torch.cuda.get_device_name(rank)
            validation_results["checks"]["gpu_memory_total"] = torch.cuda.get_device_properties(rank).total_memory
            validation_results["checks"]["cuda_version"] = torch.version.cuda
        else:
            validation_results["status"] = "failed"
            validation_results["errors"].append("GPU not available")
    except Exception as e:
        validation_results["status"] = "failed"
        validation_results["errors"].append(f"GPU check error: {str(e)}")
    
    # Check 2: Test basic tensor operations on GPU
    try:
        if gpu_available:
            start_time = time.time()
            test_tensor = torch.randn(1000, 1000, device=rank)
            test_result = torch.matmul(test_tensor, test_tensor)
            # Force synchronization to get accurate timing
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            validation_results["checks"]["tensor_op_time"] = elapsed
            validation_results["checks"]["tensor_op_success"] = True
    except Exception as e:
        validation_results["status"] = "failed" 
        validation_results["errors"].append(f"Tensor operation error: {str(e)}")
    
    # Check 3: Test NCCL communication between GPUs
    if world_size > 1:
        try:
            test_tensor = torch.ones(1, device=rank) * rank
            tensor_list = [torch.zeros(1, device=rank) for _ in range(world_size)]
            
            # All-gather operation to test GPU communication
            start_time = time.time()
            dist.all_gather(tensor_list, test_tensor)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            # Verify all tensors were gathered correctly
            all_gathered = all(tensor_list[i].item() == i for i in range(world_size))
            validation_results["checks"]["nccl_allgather_success"] = all_gathered
            validation_results["checks"]["nccl_allgather_time"] = elapsed
            
            if not all_gathered:
                validation_results["status"] = "failed"
                validation_results["errors"].append("NCCL all_gather check failed")
        except Exception as e:
            validation_results["status"] = "failed"
            validation_results["errors"].append(f"NCCL communication error: {str(e)}")
    
    # Add nvidia-smi information for this GPU
    if rank == 0:  # Only in rank 0 to avoid duplication
        validation_results["nvidia_smi"] = log_nvidia_smi(rank)
    
    return validation_results

def train_one_epoch(rank, model, dataloader, criterion, optimizer, epoch):
    """Train the model for one epoch (one pass through the dataset)."""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move data to this process's GPU
        inputs = inputs.to(rank, non_blocking=True)
        targets = targets.to(rank, non_blocking=True)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Print progress for first few batches
        if batch_idx < 3 and rank == 0:
            print(f"Rank {rank}: Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Log average loss and timing from this epoch on this rank
    avg_loss = total_loss / len(dataloader)
    epoch_time = time.time() - start_time
    print(f"Rank {rank}: Epoch {epoch+1} completed, Avg Loss = {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    return {
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "time_seconds": epoch_time
    }

def train_ddp(rank, world_size, args):
    """Function run by each process for training."""
    # Step 1: Validate GPU setup
    setup_process(rank, world_size)
    validation_results = validate_gpu_setup(rank, world_size)
    
    # Log validation results
    if rank == 0:
        print(f"Validation results: {json.dumps(validation_results, indent=2)}")
    
    # Abort if validation failed
    if validation_results["status"] == "failed":
        print(f"Rank {rank}: Validation failed. Errors: {validation_results['errors']}")
        if rank == 0:
            print("\n" + "="*50)
            print("FINAL ACCEPTANCE: FAIL")
            print(f"- GPU Available: NO")
            print(f"- Validation Failed: {validation_results['errors']}")
            print("="*50 + "\n")
        cleanup_process()
        return False
    
    # Step 2: Run the training
    torch.cuda.set_device(rank)
    performance_metrics = {
        "rank": rank,
        "epochs": [],
        "final_status": "success"
    }
    
    try:
        # Initialize model and optimizer
        start_time = time.time()
        model = models.resnet18(num_classes=10).to(rank)
        model = DDP(model, device_ids=[rank])
        criterion = nn.CrossEntropyLoss().to(rank)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model_init_time = time.time() - start_time
        performance_metrics["model_init_time"] = model_init_time
        
        # Prepare CIFAR-10 dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        start_time = time.time()
        dataset = datasets.CIFAR10(root="./data", train=True, download=(rank==0), transform=transform)
        dist.barrier()  # Wait for rank 0 to download data
        data_load_time = time.time() - start_time
        performance_metrics["data_load_time"] = data_load_time
        
        # Create dataloader with DistributedSampler
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
        
        # Training loop
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            epoch_metrics = train_one_epoch(rank, model, dataloader, criterion, optimizer, epoch)
            performance_metrics["epochs"].append(epoch_metrics)
            
            # Synchronize all processes to get clean epoch boundaries
            dist.barrier()
        
        # Capture nvidia-smi after training completes
        if rank == 0:
            print("\nTraining completed. GPU status after training:")
            performance_metrics["nvidia_smi_after_training"] = log_nvidia_smi(rank)
        
        # Save performance report
        if rank == 0:
            # Print final acceptance status at the top
            print("\n" + "="*50)
            print("FINAL ACCEPTANCE: PASS")
            print(f"- GPU Available: YES")
            if world_size > 1:
                print(f"- NCCL Communication: PASS")
            print(f"- Training Status: PASS")
            print("="*50 + "\n")
            
            print(f"Training completed successfully on {world_size} GPUs")
            
            # Write performance report to file
            with open("reports/cluster_test_report.json", "w") as f:
                json.dump({
                    "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "world_size": world_size,
                    "validation": validation_results,
                    "performance": performance_metrics,
                    "status": "success",
                    "acceptance": {
                        "status": "PASS",
                        "summary": [
                            "GPU Available: YES",
                            f"NCCL Communication: {'PASS' if world_size > 1 else 'N/A'}",
                            "Training Status: PASS"
                        ]
                    }
                }, f, indent=2)
                
    except Exception as e:
        performance_metrics["final_status"] = "failed"
        performance_metrics["error"] = str(e)
        print(f"Rank {rank}: Training failed with error: {str(e)}")
        
        if rank == 0:
            # Print final acceptance failure at the top
            print("\n" + "="*50)
            print("FINAL ACCEPTANCE: FAIL")
            print(f"- GPU Available: YES")
            if world_size > 1:
                print(f"- NCCL Communication: PASS")
            print(f"- Training Status: FAIL")
            print(f"- Error: {str(e)}")
            print("="*50 + "\n")
            
            # Write failure report to file
            with open("reports/cluster_test_report.json", "w") as f:
                json.dump({
                    "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "world_size": world_size,
                    "validation": validation_results,
                    "performance": performance_metrics,
                    "status": "failed",
                    "error": str(e),
                    "acceptance": {
                        "status": "FAIL",
                        "summary": [
                            "GPU Available: YES",
                            f"NCCL Communication: {'PASS' if world_size > 1 else 'N/A'}",
                            "Training Status: FAIL",
                            f"Error: {str(e)}"
                        ]
                    }
                }, f, indent=2)
        
    cleanup_process()

    if rank == 0:
      print(f"\nFinal Acceptance Status: {performance_metrics['final_status'].upper()}")
    return performance_metrics["final_status"] == "success"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Cluster Acceptance Test with PyTorch DDP")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--validation-only", action="store_true", help="Run only validation, skip training")
    args, unknown = parser.parse_known_args()  # Ignore extra Jupyter arguments
    
    # Determine total GPUs available and spawn one process per GPU
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPU devices found for training. This test requires at least one GPU.")
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    print(f"Starting distributed GPU cluster test across {world_size} GPUs...")
    if world_size > 1:
        # Spawn processes for each GPU
        mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # If only 1 GPU, just run in main process
        success = train_ddp(rank=0, world_size=1, args=args)
        print(f"\nFinal Acceptance Status: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0 if success else 1) 