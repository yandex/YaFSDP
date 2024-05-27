FROM nvcr.io/nvidia/pytorch:24.02-py3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /workspace

COPY ./ ya-fsdp/

RUN git clone -b v4.39-release --depth 1 https://github.com/huggingface/transformers.git \
    && git apply --directory transformers ya-fsdp/patches/transformers.diff

RUN git clone -b v0.27.0-release --depth 1 https://github.com/huggingface/accelerate.git \
    && git apply --directory accelerate ya-fsdp/patches/accelerate.diff

RUN git clone -b v0.8.3 --depth 1 https://github.com/huggingface/trl.git \
    && git apply --directory trl ya-fsdp/patches/trl.diff

RUN pip install --no-cache-dir \
    ./ya-fsdp \
    ./transformers \
    ./accelerate \
    ./trl

RUN pip install --no-cache-dir \
    -r transformers/examples/pytorch/language-modeling/requirements.txt
