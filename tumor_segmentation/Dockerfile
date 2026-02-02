FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY STU-Net ./STU-Net

WORKDIR /app/STU-Net
RUN pip install -e .

WORKDIR /app

COPY . .

ENTRYPOINT ["bash"]