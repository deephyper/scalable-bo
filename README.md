# Scaling Bayesian Optimization

[![DOI](https://zenodo.org/badge/464852027.svg)](https://zenodo.org/badge/latestdoi/464852027)


The code is available at [Scalable-BO GitHub repo](https://github.com/deephyper/scalable-bo).

This project is used to experiment the *Asynchronous Distributed Bayesian optimization* (ADBO) algorithm at HPC scale. ADBO advantages are:

* derivative-free optimization
* parallel evaluations of black-box functions
* asynchronous communication between agents
* no congestion in the optimization queue

The implementation of ADBO is directly available in the DeepHyper project (https://github.com/deephyper/deephyper/blob/develop/deephyper/search/hps/_dmbs_mpi.py).

## Environment information

The experiments were executed on the [Theta/ThetaGPU](https://www.alcf.anl.gov/alcf-resources/theta) supercomputers at the Argonne Leadership Computing Facility (ALCF). The environment used is based on available MPI implementations at the facility and a Conda environment for Python packages. The main Python dependencies of this project are `deephyper/deephyper` and `deephyper/scikit-optimize` with the following commits:

* `deephyper/deephyper`: `(7a2d553227bc62aa5ba7a307375cf729fc6178ca)`
* `deephyper-scikit-optimize`: `(4cdc150f74bb066d07a7e57986ceeaa336204e26)`

## Installations

On all the systems of the Argonne Leadership Computing Facility (ALCF) we used the `/lus/grand/projects` filesystem. Start by cloning this repository:

```console
git clone https://github.com/deephyper/scalable-bo.git
cd scalable-bo/
mkdir build
cd build/
```

Then move to the sub-section corresponding to your environment.

### For MacOSX 

Install the Xcode command line tools:

```console
xcode-select --install
```

Then check your current platform (`x86_64/arm64`) and move to the corresponding sub-section:

```console
python -c "import platform; print(platform.platform());"
```

#### For MacOSX (arm64)

If your architecture is `arm64` download MiniForge and install it:

```console
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
chmod +x Miniforge3-MacOSX-arm64.sh
sh Miniforge3-MacOSX-arm64.sh
```

After installing Miniforge clone the DeepHyper and DeepHyper/Scikit-Optimize repos and install them:

```console
git clone https://github.com/deephyper/deephyper.git
cd deephyper/
git checkout b027148046d811e466c65cfc969bfdf85eeb7c49
conda env create -f install/environment.macOS.arm64.yml
cd ..
conda activate dh-arm
git clone https://github.com/deephyper/scikit-optimize.git
cd scikit-optimize/
git checkout c272896c4e3f75ebd3b09b092180f5ef5b12692e
pip install -e .
```

Install OpenMPI and `mpi4py`:

```console
conda install openmpi
pip install mpi4py
```

### For Theta (ALCF)

From the `scalable-bo/build` folder, execute the following commands:

```console
../install/theta.sh
```

### For ThetaGPU (ALCF)

From the `scalable-bo/build` folder, execute the following commands:

```console
../install/thetagpu.sh
```
### For Summit (OLCF)

From the `scalable-bo/build` folder, execute the following commands:

```console
../install/summit.sh
```

## Organization of the repository

The repository is organized as follows:

```console
experiments/    # bash scripts for experiments and plotting tools
install/        # installation scripts 
notebooks/      # notebooks for complementary analysis
src/scalbo/     # Python package to manage experiments
test/           # test scripts to verify installation
```

## Experiments

In general experiments are launched with MPI and the `src/scalbo/exp.py` script with a command such as:

```console
$ mpirun -np 8 python -m scalbo.exp --problem ackley \
    --search DMBS \
    --timeout 20 \
    --acq-func qUCB \
    --strategy qUCB \
    --random-state 42 \
    --log-dir output \
    --verbose 1 
```

where we execute the Ackley benchmark (`problem`) with the distributed search (`DMBS`) for 20 seconds (`timeout`) with the qUCB acquisition function strategy (`acq-func` and `strategy`) with random state 42 (`random-state`), verbose mode active (`verbose`) and results are saved in the `output` (`log-dir`) directory.

Complementary information about the `python -m scalbo.exp` command can be found by using the `--help` argument:

```console
$ python -m scalbo.exp --help
usage: exp.py [-h] --problem
              {ackley_5,ackley_10,ackley_30,ackley_50,ackley_100,hartmann6D,levy,griewank,schwefel,frnn,minimalistic-frnn,molecular,candle_attn,candle_attn_sim}
              --search {AMBS,DMBS} [--sync SYNC] [--acq-func ACQ_FUNC] [--strategy {cl_max,topk,boltzmann,qUCB}] [--timeout TIMEOUT]
              [--max-evals MAX_EVALS] [--random-state RANDOM_STATE] [--log-dir LOG_DIR] [--cache-dir CACHE_DIR] [-v VERBOSE]

Command line to run experiments.

optional arguments:
  -h, --help            show this help message and exit
  --problem {ackley_5,ackley_10,ackley_30,ackley_50,ackley_100,hartmann6D,levy,griewank,schwefel,frnn,minimalistic-frnn,molecular,candle_attn,candle_attn_sim}
                        Problem on which to experiment.
  --search {AMBS,DMBS}  Search the experiment must be done with.
  --sync SYNC           If the search workers must be syncronized or not.
  --acq-func ACQ_FUNC   Acquisition funciton to use.
  --strategy {cl_max,topk,boltzmann,qUCB}
                        The strategy for multi-point acquisition.
  --timeout TIMEOUT     Search maximum duration (in min.) for each optimization.
  --max-evals MAX_EVALS
                        Number of iterations to run for each optimization.
  --random-state RANDOM_STATE
                        Control the random-state of the algorithm.
  --log-dir LOG_DIR     Logging directory to store produced outputs.
  --cache-dir CACHE_DIR
                        Path to use to cache logged outputs (e.g., /dev/shm/).
  -v VERBOSE, --verbose VERBOSE
                        Wether to activate or not the verbose mode.
```

### Docker (Single Node)

Experiments are challenging to reproduce at large scale, therefore we provide a Docker image to reproduce similar results on a single machine with multiple cores. We assume that Docker is already installed. If it is not the case please check [how to install Docker](https://docs.docker.com/get-docker/).

**Your Docker configuration needs to use at least 8 CPUs.**

Pull the docker image at:
```console
docker pull romainegele/scalable-bo
```

Start a Docker container with this image:
```console
docker run --platform linux/amd64 -ti romainegele/scalable-bo /bin/bash
```

Then go to the experimental folder for Docker:
```console
cd experiments/docker/
```

Execute the synchronous distributed BO with UCB and Boltzmann policy (SDBO+bUCB):
```console
./fast_ackley_2-DMBS-sync-UCB-boltzmann-1-8-30-42.sh
```

Execute the asynchronous distributed BO with qUCB (ADBO+qUCB):
```console
./fast_ackley_2-DMBS-async-qUCB-qUCB-1-8-30-42.sh
```

The results should no be in `experiments/docker/output/`. Each experiment's output will contain an:
* a `results.csv` file containing the evaluated configurations with the corresponding objectives and some more information about when the function was evaluated.
* a `deephyper*.log` file containing logging information from the algorithm on the rank 0 generally.

Then you can plot figures with the following command:
```console
python ../plot.py --config plot.yaml
```

### For Theta (ALCF)

```console
cd experiments/theta/jobs/
```

### For ThetaGPU (ALCF)

```console
cd experiments/thetagpu/jobs/
```


### For Summit (OLCF)

```console
cd experiments/summit/jobs/
```
