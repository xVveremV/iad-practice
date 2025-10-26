FROM python:3.11-slim

ARG NB_USER=jupyter
ARG NB_UID=1000
ARG NB_GID=100
RUN useradd -m -u ${NB_UID} -g ${NB_GID} -s /bin/bash ${NB_USER}

# --- sys ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    fonts-dejavu \
    python3-tk \
    build-essential \
    gcc \
    libfreetype6-dev libpng-dev libjpeg-dev \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data

# --- pip ---
RUN /usr/local/bin/python -m pip install --no-cache-dir --upgrade pip

# --- libs ---
RUN /usr/local/bin/python -m pip install --no-cache-dir \
    notebook jupyterlab ipykernel \
    numpy pandas matplotlib seaborn ipywidgets \
    scikit-learn scipy statsmodels \
    nltk \
    wordcloud

# till this point image size is 1.8Gb

# this layer adds 5Gb, but contains all the test data
# --- Pre-download ALL NLTK data into a shared, system-wide location ---
RUN /usr/local/bin/python -m nltk.downloader all -d /usr/local/share/nltk_data

# this layer adds 2.9b, but useful for future
# --- tensorflow ---
#RUN pip install --no-cache-dir \
#    tensorflow \
#    torch \
#    transformers

USER ${NB_USER}
WORKDIR /workspace
EXPOSE 8888

# Optional: set a token via `-e JUPYTER_TOKEN=...` on `docker run`
ENV JUPYTER_TOKEN=""

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash","-lc","jupyter notebook --NotebookApp.token=\"${JUPYTER_TOKEN}\" --ip=0.0.0.0 --no-browser --port=8888"]
