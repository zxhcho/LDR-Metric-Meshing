# Toolchain and runtime for the obstacle verification.
# docker build -f docker/Dockerfile . produces the image.
# The release tree and the unpacked archive are bind-mounted at run time.
# This image does not copy that source and does not sleep as its entrypoint.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        g++ \
        cmake \
        git \
        ca-certificates \
        pkg-config \
        perl \
        libscotch-dev \
        libsuitesparse-dev \
        libmetis-dev \
        libopenblas-dev \
        liblapack-dev \
        zlib1g-dev \
    && ln -sf python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        torch==1.13.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

RUN python3 -m pip install --no-cache-dir \
    numpy==1.24.4 \
    scipy==1.10.1 \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    scikit-learn==1.3.2 \
    potpourri3d==1.0.0 \
    robust-laplacian==0.2.7 \
    libigl==2.5.1 \
    tensorboard==2.14.0 \
    six==1.16.0

WORKDIR /tmp
