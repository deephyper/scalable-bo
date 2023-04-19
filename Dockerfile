FROM continuumio/miniconda3

WORKDIR /root/build/
RUN touch /root/.bashrc

RUN apt-get update
RUN apt-get install build-essential -y
RUN apt-get install lsb-release -y
RUN apt-get install vim -y

# Install LLVM
RUN apt-get install lsb-release wget software-properties-common gnupg -y
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# Create new conda environment
RUN conda update -n base -c defaults conda -y 
RUN conda create -n dhenv python=3.9 -y --quiet

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "dhenv", "/bin/bash", "-c"]

# Update pip
RUN pip install --upgrade pip

# Install OpenMPI
RUN conda install -c conda-forge openmpi mpi4py=3.1.4 -y
RUN echo "export OMPI_ALLOW_RUN_AS_ROOT=1" >> /root/.bashrc
RUN echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" >> /root/.bashrc

# Install Redis and RedisJSON
RUN conda install redis rust
RUN git clone https://github.com/RedisJSON/RedisJSON.git
RUN cd RedisJSON/ && cargo build --release

# Clone & Install DeepHyper
RUN git clone https://github.com/deephyper/deephyper.git
RUN cd deephyper/ && git checkout b3726dc832ab38e06577446c9d0261ff068fe328 && cd ..
RUN pip install -e "deephyper/[default,mpi,redis]"

# Install Benchmarks
RUN git clone https://github.com/deephyper/benchmark.git deephyper-benchmark
RUN pip install -e "deephyper-benchmark/"

# Activate 'dhenv' environment by default
RUN touch /root/.redis.conf
RUN echo "loadmodule /root/build/RedisJSON/target/release/librejson.so" >> /root/.redis.conf
RUN echo "conda activate dhenv" >> /root/.bashrc

# Install Scalable-BO
WORKDIR /root
RUN git clone https://github.com/deephyper/scalable-bo.git
RUN pip install -e scalable-bo/src/scalbo/
RUN python -c "import deephyper_benchmark as dhb; dhb.install('HPOBench/tabular/navalpropulsion');"
WORKDIR /root/scalbo/experiments/docker
