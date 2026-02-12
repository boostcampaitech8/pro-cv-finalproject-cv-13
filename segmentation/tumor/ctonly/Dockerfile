FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# nnUNet 2.6.3 설치
COPY nnUNet ./nnUNet
WORKDIR /app/nnUNet
RUN pip install -e .
WORKDIR /app

COPY inference.py .
COPY model ./model

ENTRYPOINT ["python"]
CMD ["inference.py"]
