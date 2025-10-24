FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# ------------------------
# Environment setup
# ------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=US/Pacific \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    PYOPENGL_PLATFORM=egl \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    VIRTUAL_ENV="/app/venv" \
    PATH="/app/venv/bin:$PATH"

# ------------------------
# Base system dependencies
# ------------------------
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        # GPU / graphics libs
        libxau6 libxdmcp6 libxcb1 libxext6 libx11-6 libglvnd0 \
        libgl1 libglx0 libegl1 libgles2 libglvnd-dev libvulkan1 \
        # Build tools
        build-essential curl wget unzip git cmake ninja-build \
        # Dev libraries
        libxml2 libxml2-dev libxslt1-dev libfreetype6-dev \
        libopenmpi-dev libcups2-dev libssl-dev \
        libboost-all-dev libosmesa6-dev libglib2.0-0 libxrender1 \
        libglew-dev libglfw3-dev pybind11-dev \
        libjpeg-dev ca-certificates freeglut3-dev \
        libsm6 libxext-dev libgmp-dev llvm-dev libsdl2-dev \
        # Utilities
        gnupg2 lsb-release swig patchelf ffmpeg rsync \
        # Python
        python3 python3-pip python3-dev python3-setuptools python3-venv && \
    # Make Python 3 the default
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    # Cleanup
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ------------------------
# Working directory
# ------------------------
WORKDIR /app

# ------------------------
# Eigen 3.4
# ------------------------
RUN curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
    | tar xz && \
    cd eigen-3.4.0 && mkdir build && cd build && \
    cmake .. && make -j$(nproc) install && \
    cd ../.. && rm -rf eigen-3.4.0

# ------------------------
# Intel MKL
# ------------------------
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-mkl-devel=2025.2.0-628 && \
    rm -rf /var/lib/apt/lists/*

# ------------------------
# SymEngine
# ------------------------
RUN git clone --depth=1 https://github.com/symengine/symengine && \
    cd symengine && mkdir build && cd build && \
    cmake -DWITH_LLVM=on -DBUILD_BENCHMARKS=off -DBUILD_TESTS=off .. && \
    make -j$(nproc) install && \
    cd ../.. && rm -rf symengine

# ------------------------
# Collision libs (libccd + FCL)
# ------------------------
RUN git clone --depth=1 https://github.com/danfis/libccd && \
    cd libccd && cmake -DBUILD_SHARED_LIBS=ON . && \
    make -j$(nproc) install && \
    cd .. && rm -rf libccd

RUN git clone --depth=1 https://github.com/flexible-collision-library/fcl && \
    cd fcl && mkdir build && cd build && \
    cmake .. && make -j$(nproc) install && \
    cd ../.. && rm -rf fcl

# ------------------------
# Magnum viewer + deps
# ------------------------
RUN git clone --depth=1 https://github.com/mosra/corrade.git && \
    cd corrade && mkdir build && cd build && \
    cmake .. && make -j$(nproc) install && \
    cd ../.. && rm -rf corrade

RUN git clone --depth=1 https://github.com/mosra/magnum.git && \
    cd magnum && mkdir build && cd build && \
    cmake -DMAGNUM_WITH_SDL2APPLICATION=ON .. && \
    make -j$(nproc) install && \
    cd ../.. && rm -rf magnum

RUN git clone --depth=1 https://github.com/mosra/magnum-plugins.git && \
    cd magnum-plugins && mkdir build && cd build && \
    cmake -DMAGNUM_WITH_PNGIMAGECONVERTER=ON .. && \
    make -j$(nproc) install && \
    cd ../.. && rm -rf magnum-plugins

# ------------------------
# Build and install py_dismech
# ------------------------
ENV MKL_DIR=/opt/intel/oneapi/mkl/2025.2
RUN git clone --depth=1 https://github.com/StructuresComp/dismech-rods.git /app/dismech-rods && \
    cd dismech-rods && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && cd ../..

# ------------------------
# Python environment & requirements
# ------------------------
COPY requirements.txt /app/

RUN python3 -m venv $VIRTUAL_ENV && \
    pip install --upgrade pip wheel setuptools packaging && \
    pip install torch==2.6.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt && \
    pip cache purge && \
    find $VIRTUAL_ENV -name '*.pyc' -delete && \
    find $VIRTUAL_ENV -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true


RUN git clone --depth=1 https://github.com/HorizonRobotics/alf.git /app/alf
ENV PYTHONPATH=$PYTHONPATH:/app/dismech-rods:/app/alf
