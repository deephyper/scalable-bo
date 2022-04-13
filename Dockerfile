FROM continuumio/miniconda3

WORKDIR /app

# Copy the repo
COPY . .

# Move to build
WORKDIR /app/build

# Install base requirements
# RUN apt-get update -y
# RUN apt-get build-essential -y

# Create new conda environment
RUN conda create -n dhenv python=3.8 -y --quiet

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "dhenv", "/bin/bash", "-c"]

# Install OpenMPI
RUN conda install openmpi mpi4py -y
RUN echo "export OMPI_ALLOW_RUN_AS_ROOT=1" >> ~/.bashrc
RUN echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" >> ~/.bashrc

# Clone & Install DeepHyper/Scikit-Optimize (develop)
RUN git clone https://github.com/deephyper/scikit-optimize.git deephyper-scikit-optimize
RUN cd deephyper-scikit-optimize/ && git checkout 4cdc150f74bb066d07a7e57986ceeaa336204e26 && cd ..
RUN pip install -e deephyper-scikit-optimize/

# Clone & Install DeepHyper (develop)
RUN git clone https://github.com/deephyper/deephyper.git
RUN cd deephyper/ && git checkout 7c677e99aea0c6d906dc5f2f24a3fac913119b7a && cd ..
RUN pip install -e deephyper/

# Install Scalable-BO
RUN pip install -e ../src/scalbo/

# activate 'dh' environment by default
RUN echo "conda activate dhenv" >> ~/.bashrc

WORKDIR /app