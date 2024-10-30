# Use CUDA 12.4 as the base image and install Ubuntu
FROM nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu20.04

# Install Python and necessary system dependencies, then clean up
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4 support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# Install remaining Python dependencies and clean up pip cache
RUN pip3 install --no-cache-dir transformers einops timm peft sentencepiece flash_attn Pillow boto3

# Copy the inference script to the image
COPY inference.py /opt/ml/model/code/

# Define the entry point
ENTRYPOINT ["python3", "/opt/ml/model/code/inference.py"]
