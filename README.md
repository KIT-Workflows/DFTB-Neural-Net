# DFTB-Neural-Net
Install conda environment, the following packages would be needed:

```
conda create -n NAME python=3.6 numpy scikit-learn tensorflow-gpu keras cython pandas matplotlib seaborn ase pyyaml
```

Install via pip ordered-enum

# Installing DFTB+ in int-nano
1. git clone -b machine-learning https://github.com/tomaskubar/dftbplus.git 
2. cd dftbplus
3. module load gnu8/8.3.0
4. module load openblas/0.3.7
5. module load cmake
6. mkdir _build 
7. FC=gfortran CC=gcc cmake -DCMAKE_INSTALL_PREFIX=$HOME/opt/dftb+ -B _build .
8. cmake --build _build -- -j 
9. cmake --install _build

```diff 
+ Check if the `dftb+` executable exist in the dftbplus/_build/prog/dftb+/ folder. If so, then everything is okay. 
```

```diff 
+ be cautious with conda env., it must have libgfortran5 if the installation wants to be done inside the conda environment
```
