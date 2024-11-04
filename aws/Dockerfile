FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir -U wheel setuptools packaging && \
    pip3 install --no-cache-dir \
      transformers==4.46.1 einops==0.8.0 timm==1.0.9 peft==0.13.2 \
      sentencepiece==0.2.0 Pillow==10.2.* boto3==1.35.* ninja && \
    pip3 cache purge

RUN pip3 install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 


COPY inference.py /opt/ml/model/code/

ENTRYPOINT ["python3", "/opt/ml/model/code/inference.py"]
