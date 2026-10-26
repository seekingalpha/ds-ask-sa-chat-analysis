ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Hugging Face model cache (~18 GB for gemma-2-9b-it) on the processing volume
    HF_HOME=/opt/ml/processing/hf_cache

WORKDIR /opt/ml/code

# CUDA build of torch, same as the tested SageMaker kernel (2.8.0, CUDA 12.9).
# Own layer: the biggest download, changes rarely.
RUN pip install --upgrade pip && \
    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# Project packages (src/) and the rest of the dependencies from pyproject.toml
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# main.py, variables.yaml, queries/, logger/, ...
COPY . .

# The pipeline passes e.g. ["main.py", "--task", "ner"]
ENTRYPOINT ["python3"]
