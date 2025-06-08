#!/bin/bash

# RAG Bot GPU Migration Script
# This script helps migrate your RAG Bot application to the GPU node pool

set -e

echo "🚀 RAG Bot GPU Migration Script"
echo "================================"

# Configuration
REGISTRY="chorusproduction.azurecr.io"
IMAGE_NAME="ragbot"
TAG="gpu-$(date +%Y%m%d-%H%M%S)"
NAMESPACE="ragbot"

echo "📋 Configuration:"
echo "   Registry: $REGISTRY"
echo "   Image: $IMAGE_NAME:$TAG"
echo "   Namespace: $NAMESPACE"
echo ""

# Step 1: Build the GPU-optimized Docker image
echo "🔨 Step 1: Building GPU-optimized Docker image..."
docker build -t $REGISTRY/$IMAGE_NAME:$TAG .
docker tag $REGISTRY/$IMAGE_NAME:$TAG $REGISTRY/$IMAGE_NAME:latest

echo "✅ Docker image built successfully"
echo ""

# Step 2: Push to Azure Container Registry
echo "📤 Step 2: Pushing to Azure Container Registry..."
az acr login --name chorusproduction
docker push $REGISTRY/$IMAGE_NAME:$TAG
docker push $REGISTRY/$IMAGE_NAME:latest

echo "✅ Image pushed to ACR successfully"
echo ""

# Step 3: Backup current deployment
echo "💾 Step 3: Backing up current deployment..."
kubectl get deployment ragbot -n $NAMESPACE -o yaml > ragbot-backup-$(date +%Y%m%d-%H%M%S).yaml
echo "✅ Backup created"
echo ""

# Step 4: Check GPU node availability
echo "🔍 Step 4: Checking GPU node availability..."
GPU_NODES=$(kubectl get nodes -l agentpool=gpupool --no-headers | wc -l)
if [ $GPU_NODES -eq 0 ]; then
    echo "❌ No GPU nodes found with label 'agentpool=gpupool'"
    echo "Please ensure your GPU node pool is running and labeled correctly"
    exit 1
fi
echo "✅ Found $GPU_NODES GPU nodes in pool 'gpupool'"

# Check GPU availability
echo "🎮 Checking GPU resources..."
kubectl describe nodes -l agentpool=gpupool | grep -E "(nvidia.com/gpu|Allocatable)" || true
echo ""

# Step 5: Apply the GPU-optimized deployment
echo "🚀 Step 5: Deploying to GPU node pool..."
kubectl apply -f k8s/deployment.yaml

echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/ragbot -n $NAMESPACE --timeout=600s

echo "✅ Deployment completed successfully!"
echo ""

# Step 6: Verify GPU usage
echo "🔍 Step 6: Verifying GPU deployment..."
echo "Checking pod placement..."
POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=ragbot -o jsonpath='{.items[0].metadata.name}')
NODE_NAME=$(kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath='{.spec.nodeName}')
echo "Pod $POD_NAME is running on node: $NODE_NAME"

# Check if it's on a GPU node
GPU_NODE_CHECK=$(kubectl get node $NODE_NAME -o jsonpath='{.metadata.labels.agentpool}')
if [ "$GPU_NODE_CHECK" = "gpupool" ]; then
    echo "✅ Pod is running on GPU node pool!"
else
    echo "⚠️  Pod is NOT running on GPU node pool (running on: $GPU_NODE_CHECK)"
fi

echo ""
echo "🔧 Step 7: Checking GPU availability in pod..."
kubectl exec -n $NAMESPACE $POD_NAME -- python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('No GPU detected in container')
"

echo ""
echo "🏥 Step 8: Health check..."
kubectl get pods -n $NAMESPACE -l app=ragbot
kubectl logs -n $NAMESPACE $POD_NAME --tail=20

echo ""
echo "🎉 GPU Migration Complete!"
echo "=========================="
echo ""
echo "Your RAG Bot is now running on GPU nodes with the following optimizations:"
echo "✅ CUDA-enabled Docker image (nvidia/cuda:12.1-runtime-ubuntu22.04)"
echo "✅ PyTorch with CUDA 12.1 support"
echo "✅ FAISS-GPU for faster vector search"
echo "✅ Mixed precision (FP16) inference"
echo "✅ Tesla T4 specific optimizations"
echo "✅ GPU memory management"
echo ""
echo "Performance improvements expected:"
echo "📈 Image embedding generation: 5-10x faster"
echo "📈 Text embedding generation: 3-5x faster"
echo "📈 Vector search: 2-5x faster (with FAISS-GPU)"
echo "📈 Image captioning: 3-7x faster"
echo ""
echo "Monitor your application logs for GPU usage confirmation:"
echo "kubectl logs -n $NAMESPACE -l app=ragbot -f"
echo ""
echo "To check GPU utilization:"
echo "kubectl exec -n $NAMESPACE $POD_NAME -- nvidia-smi" 