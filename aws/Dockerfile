FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 11.8
RUN pip3 install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118


RUN pip3 install --no-cache-dir -U wheel setuptools packaging && \
    pip3 install --no-cache-dir \
      transformers==4.46.1 einops==0.8.0 timm==1.0.9 peft==0.13.2 \
      sentencepiece==0.2.0 Pillow==11.0.* boto3==1.35.* ninja && \
    pip3 cache purge

RUN pip3 install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Copy the inference script to the image
COPY inference.py /opt/ml/model/code/

# Define the entry point
ENTRYPOINT ["python3", "/opt/ml/model/code/inference.py"]
