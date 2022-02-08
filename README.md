# JARVIS-HybridNet

## Install Instructions

- Clone the repository with
```
git clone https://github.com/JARVIS-MoCap/JARVIS-HybridNet.git
cd JARVIS-HybridNet
```

- Make sure [Anaconda](https://www.anaconda.com/products/individual) is installed on your machine.

- Setup the jarvis Anaconda environment and activate it
```
conda create -n jarvis python=3.9  pytorch=1.10.1 torchvision cudatoolkit=11.3 notebook  -c pytorch
conda activate jarvis
```

- Make sure your setuptools package is up to date \
  `pip install -U setuptools==59.5.0`

- Install JARVIS
  `pip install -e .`