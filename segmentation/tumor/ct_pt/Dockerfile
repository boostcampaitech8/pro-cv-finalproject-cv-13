FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

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

ENTRYPOINT ["python"]
CMD ["inference.py", "/test_dir/CHUM-001/CHUM-001__CT.nii.gz", "outputs/CHUM-001", "--model-folder", "plans", "--checkpoint", "checkpoint_best.pth", "--pt-path", "/test_dir/CHUM-001/CHUM-001__PT.nii.gz"]