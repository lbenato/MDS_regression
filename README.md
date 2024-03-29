# MDS_regression

.....

-----

## Getting started: setting up the environment on NAF GPUs

Instructions to use naf-gpu infrastructure at DESY.

**Note:** The full installation will take a while. Please be patient and carefully follow these instructions.

### Login

We will use the naf-gpu login node. Here `<username>` is your DESY username.

```bash
ssh -XY <username>@naf-cms-gpu01.desy.de
```

### Creating a link to your /nfs/dust/ area

This is needed to be able to see /nfs/dust/ files on JupyterHub.

```bash
cd ~
ln -s /nfs/dust/cms/user/<username> nfs_dust
```

### Installing anaconda

We will install all the needed packages with anaconda.

```bash
# load cuda module
module load cuda

# change directory to your home directory on dust
cd /nfs/dust/cms/user/<username>/

# get the anaconda installer
# Note: To have the latest version, you can look up the list at
# https://repo.continuum.io/archive/
wget https://repo.continuum.io/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

# install anaconda
bash Anaconda3-2023.09-0-Linux-x86_64.sh

# press ENTER to review the license agreement and type "yes" to accept

# ATTENTION! When asked where to install anaconda,
# do NOT press enter to confirm the default location,
# but provide your dust home directory instead
# (type: /nfs/dust/cms/user/<username>/anaconda3).

# Answer all other prompts with the recommended option (in brackets).
# Optional: To have an easier way to activate your conda environment,
# you can allow anaconda to edit your .bashrc file.

# load anaconda
# IMPORTANT: You have to run this command every time you log in to NAF!
export PATH=/nfs/dust/cms/user/<username>/anaconda3/bin:$PATH

```


### Creating a conda environment

We will be working inside a conda environment. To create and activate the environment, follow the instructions below:

```bash
# create a conda environment called "mds_regression"
conda create -n mds_regression python=3.7

# activate the environment
# IMPORTANT: You also need to run this command every time you log in to NAF!
source activate /nfs/dust/cms/user/<username>/anaconda3/envs/mds_regression
```

### Installing required packages

**Note:** When installing packages with conda, the "solving environment" step can take a long time. This is normal behavior so do not abort the installation (unless it runs longer than several hours).

```bash
# cd to your environment directory
# Note: This is important! If you try to install packages when not in
# your environment directory, you might get file I/O errors!
cd /nfs/dust/cms/user/<username>/anaconda3/envs/mds_regression/

#change solver and install root
conda config --set solver libmamba

# install pytables
conda install pytables

# install ROOT
conda install -c conda-forge root

## install root_numpy
conda install -c conda-forge root_numpy

#create a requirements.txt file including the following lines:
awkward <= 2.0    # apparently in 2.X there is no conversion to ak0 anymore
awkward0
#coffea
#hist
#hls4ml
keras
matplotlib
#mplhep
numpy
#qkeras
#pytables
pandas
#root
root-numpy
scikit-learn 
scipy
seaborn
#shap
uproot
vector
tensorflow

pip install -r path/to/requirements.txt

# pytorch

# GPU install (CUDA 11.7)
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# CPU Only
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch

# pytorch-geometric
conda install pyg -c pyg

# pytorch-cluster
conda install pytorch-cluster -c pyg

# install jupyterhub kernel
cd /nfs/dust/cms/user/<username>/anaconda3/envs/mds_regression #you should be here already, better to be sure
conda activate mds_regression
pip install ipykernel --user
python -m ipykernel install --user --name="mds_regression"


```


### Cloning this repository

If you followed these instructions, you should have tensorflow2.0 installed. Let's now clone this repository.

```bash
cd /nfs/dust/cms/user/<username>
mkdir ML_LLP #this is just my personal choice
cd ML_LLP
git clone https://github.com/lbenato/MDS_regression.git
```

### Location of samples...

Work in progress

### Loading a conda environment to jupyter notebooks

Launch JupyterHub at https://naf-jhub.desy.de/

Check `Select GPU node` and click on `Start`

On the browser, click on the directories to go to:

`nfs_dust/ML_LLP/MDS_regression/`

Here, you can find two jupyter notebooks:
- work in progress....
- [convert_dataset.ipynb](tf-keras/convert_dataset.ipynb)
- [keras_train.ipynb](tf-keras/keras_train.ipynb)

**Note**: Every time you open a notebook, make sure to load your conda environment!\
You just need to click on `Kernel`, `Change kernel`, and `mds_regression`.

### Run Keras/Tensorflow scripts
...

**Note:** Don't forget to change the `username` when needed!

------
