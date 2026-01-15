FROM pytorch/2.6.0-cuda11.8-cudnn9-runtime

WORKDIR .
RUN apt-get update && apt-get install -y \
    git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt