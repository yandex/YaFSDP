#!/usr/bin/env sh

docker buildx build \
    --load \
    --network host \
    -f docker/Dockerfile \
    -t ya-fsdp:latest .
