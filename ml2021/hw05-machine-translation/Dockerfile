FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG FAIRSEQ_COMMIT=9a1c497

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/hw5-env

COPY requirements-docker.txt ./requirements-docker.txt

# 舊版 fairseq 的 omegaconf metadata 需要 pip 24.0 以下。
RUN python -m pip install --no-cache-dir --upgrade "pip<24.1" setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements-docker.txt

RUN git clone https://github.com/pytorch/fairseq.git /opt/fairseq \
    && cd /opt/fairseq \
    && git checkout "${FAIRSEQ_COMMIT}" \
    && python -m pip install --no-cache-dir --no-build-isolation --editable /opt/fairseq

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/fairseq

WORKDIR /workspace

CMD ["bash"]
