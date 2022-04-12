# Scaling Bayesian Optimization

The code is available at [Scalable-BO GitHub repo](https://github.com/deephyper/scalable-bo).

This project is used to experiment the *Asynchronous Distributed Bayesian optimization* (ADBO) algorithm at HPC scale. ADBO advantages are:

* derivative-free optimization
* parallel evaluations of black-box functions
* asynchronous communication between agents
* no congestion in the optimization queue

## Environment information

The experiments were executed on the [Theta/ThetaGPU](https://www.alcf.anl.gov/alcf-resources/theta) supercomputers at the Argonne Leadership Computing Facility (ALCF). The environment used is based on available MPI implementations at the facility and a Conda environment for Python packages. The main Python dependencies of this project are `deephyper/deephyper` and `deephyper/scikit-optimize` with the following commits:

* `deephyper/deephyper`: `(b027148046d811e466c65cfc969bfdf85eeb7c49)`
* `deephyper-scikit-optimize`: `(c272896c4e3f75ebd3b09b092180f5ef5b12692e)`

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

## Organization of the repository

...

## Experiments

- [ ] explain how the experiments are executed (python entry point, scripts)
- [ ] explain how results are presented

In general experiments are launched with an MPI executable and the `src/scalbo/exp.py` script with a command such as:

```console
mpirun -np 8 python -m scalbo.exp --problem ackley \
    --search DMBS \
    --timeout 20 \
    --acq-func qUCB \
    --strategy qUCB \
    --random-state 42 \
    --log-dir output \
    --verbose 1 
```

where we execute the Ackley benchmark (`problem`) with the distributed search (`DMBS`) for 20 seconds (`timeout`) with the qUCB acquisition function strategy (`acq-func` and `strategy`) with random state 42 (`random-state`), verbose mode active (`verbose`) and results are saved in the `output` (`log-dir`) directory.

### Single Node

```console
cd experiments/local/
```

### For Theta (ALCF)

```console
cd experiments/theta/jobs/
```

### For ThetaGPU (ALCF)

```console
cd experiments/thetagpu/jobs/
```

