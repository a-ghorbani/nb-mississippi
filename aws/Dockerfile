FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4
RUN pip3 install --no-cache-dir torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir -U wheel setuptools packaging && \
    pip3 install --no-cache-dir \
    transformers==4.46.1 einops==0.8.0 timm==1.0.9 peft==0.13.2 \
    sentencepiece==0.2.0 flash_attn==2.6.3 Pillow==9.5.0 boto3==1.35.51 && \
    pip3 cache purge

# Copy the inference script to the image
COPY inference.py /opt/ml/model/code/

# Define the entry point
ENTRYPOINT ["python3", "/opt/ml/model/code/inference.py"]
