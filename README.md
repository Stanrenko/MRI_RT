# Dynamic MRI illustration with BART 

## Requirements

After cloning the repository, one should create the environment and install the dependencies :
```
conda create -n <env> python=3.10
conda activate <env>
pip install -e .
```

BART is required: https://bart-doc.readthedocs.io/en/latest/install.html

To use the environment in a jupyter notebook:
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=<env>
```
