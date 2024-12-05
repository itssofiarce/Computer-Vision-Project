#FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as base
FROM ubuntu:22.04

ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

COPY requirements.txt  /requirements.txt
COPY data/* models/* utils/* /app/

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip install -r /requirements.txt
