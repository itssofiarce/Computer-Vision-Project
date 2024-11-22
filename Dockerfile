#FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as base
FROM ubuntu:20.04

ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

RUN apt-get update && apt-get install -y \
    pip install -no-cache-dir --upgrade -r /app/requirements.txt
