FROM python:3-slim

ARG NB_USER=jupyter
ARG NB_UID=1000
ARG NB_GID=100
RUN useradd -m -u ${NB_UID} -g ${NB_GID} -s /bin/bash ${NB_USER}

# --- System deps (tiny) ---
# - tini: proper signal handling (Ctrl+C, stop)
# - fonts-dejavu: readable matplotlib text out of the box
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    fonts-dejavu \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# pip
RUN python -m pip install --upgrade pip

# notebook jupyterlab ipykernel
RUN pip install --no-cache-dir \
      notebook jupyterlab ipykernel

# numpy pandas matplotlib seaborn
RUN pip install --no-cache-dir \
      numpy pandas matplotlib seaborn

# scikit-learn
RUN pip install --no-cache-dir \
      scikit-learn

# nltk
RUN pip install --no-cache-dir \
      nltk

# tensorflow (5-)
RUN pip install --no-cache-dir \
      tensorflow

# transformers (8-)
RUN pip install --no-cache-dir \
      transformers

# ipywidgets
RUN pip install --no-cache-dir \
      ipywidgets

# typorch
RUN pip install --no-cache-dir \
      torch

# tf-keras
RUN pip install --no-cache-dir \
      tf-keras

USER ${NB_USER}
WORKDIR /workspace
EXPOSE 8888

# Optional: set a token via `-e JUPYTER_TOKEN=...` on `docker run`
ENV JUPYTER_TOKEN=""

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash","-lc","jupyter notebook --NotebookApp.token=\"${JUPYTER_TOKEN}\" --ip=0.0.0.0 --no-browser --port=8888"]
