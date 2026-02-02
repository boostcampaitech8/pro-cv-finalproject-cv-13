FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY nnUNet ./nnUNet
WORKDIR /app/nnUNet
RUN pip install -e .

WORKDIR /app
RUN pip install --no-cache-dir TotalSegmentator

COPY . .

RUN sed -i 's/\r$//' inference.sh && chmod +x inference.sh

ENTRYPOINT ["bash"]