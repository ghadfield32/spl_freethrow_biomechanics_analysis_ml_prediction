# Use NVIDIA CUDA base image for GPU support (if needed)
FROM nvidia/cuda:12.0.0-base-ubuntu22.04

# --- Build Arguments ---
ARG ENV_NAME=data_science_ft_bio_predictions
ARG PYTHON_VER=3.10

# --- Environment Variables ---
# Prevent tzdata from asking for user input
ENV DEBIAN_FRONTEND=noninteractive
# Set up CUDA library paths
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# Define Conda directory and add it to PATH
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV ENV_NAME=${ENV_NAME}
# Java (for any Java-based tools) and MLflow default settings can be added here if needed
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# --- Install System Dependencies ---
RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    vim \
    python3-pip \
    python3-dev \
    python3-opencv \
    openjdk-11-jdk \
    libglib2.0-0 \
    libstdc++6 \
    libgcc1 \
    libc6 \
    libc6-dev \
    libc-dev-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Install Miniconda ---
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda clean -a -y

# --- Create a New Conda Environment ---
# Copy the environment file (adjust path as needed) into the image
COPY .devcontainer/environment.yml /tmp/environment.yml
# Replace placeholders for ENV_NAME and PYTHON_VER in the YAML file
RUN sed -i "s/name: \${ENV_NAME}/name: ${ENV_NAME}/" /tmp/environment.yml && \
    sed -i "s/- python=\${PYTHON_VER}/- python=${PYTHON_VER}/" /tmp/environment.yml && \
    conda env create -f /tmp/environment.yml && \
    conda clean -a -y

# --- Activate the Conda Environment by Default ---
# Add environment activation to the default shell configuration
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate ${ENV_NAME}" >> /etc/bash.bashrc

# --- Install Additional Python Packages (if needed) ---
# Here you can install extra pip packages (for example, PyTorch for GPU support).
RUN /bin/bash -c ". /opt/conda/etc/profile.d/conda.sh && conda activate ${ENV_NAME} && \
    pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"

# # --- Install Mito Packages (if not already installed via environment.yml) ---
# # This step is optional if your environment.yml already installs mitosheet and mito-ai.
# RUN /bin/bash -c ". /opt/conda/etc/profile.d/conda.sh && conda activate ${ENV_NAME} && \
#     pip install --no-cache-dir mitosheet mito-ai"

# # --- Enable Mito for Jupyter (if using Jupyter Notebook or JupyterLab 3.x) ---
# # For JupyterLab 4.x, this step may not be necessary.
# RUN /bin/bash -c ". /opt/conda/etc/profile.d/conda.sh && conda activate ${ENV_NAME} && \
#     jupyter nbextension install mitosheet && \
#     jupyter nbextension enable mitosheet"
    
# --- Copy the Project Code into the Container ---
# This ensures your complete ML pipeline (modules, configs, tasks, etc.) is present.
WORKDIR /workspace
COPY . /workspace

# --- Optional: Create an Entrypoint Script ---
# You can create a small shell script (entrypoint.sh) to choose different modes (like running
# invoke tasks, serving MLflow UI, or launching Streamlit) based on command-line arguments.
# For simplicity, we set the default CMD below.
# (If you create an entrypoint, remember to add: RUN chmod +x /workspace/entrypoint.sh )
#uncomment below if you want to test entrypoint.sh, and comment out the CMD["/bin/bash"]
# Copy entrypoint script
COPY .devcontainer/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Set the default command (you can override this in docker-compose)
CMD ["/workspace/entrypoint.sh"]

# --- Set the Default Command ---
# The command can be overridden by docker-compose (or command-line parameters).
# CMD ["/bin/bash"]
