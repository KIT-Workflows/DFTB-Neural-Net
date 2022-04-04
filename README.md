# DFTB-Neural-Net

Install conda environment, the following packages would be needed:

```
conda create -n NAME python=3.6 numpy scikit-learn tensorflow-gpu keras cython pandas matplotlib seaborn ase pyyaml
```

Install via pip ordered-enum

# Installing DFTB+ in int-nano
1. git clone github.com/tomaskubar/dftbplus/tree/machine-learning
2. cd dftbplus
3. module load intel/19.0.5.281
4. export  INTEL_LICENSE_FILE=(ask for this info)
5. mkdir \_build inside the dftbplus folder
6. FC=ifort CC=icc cmake -DCMAKE_INSTALL_PREFIX=$HOME/dftbplus -B \_build .
7. cmake --build \_build -- -j
8. ./utils/get_opt_externals slakos
9. pushd \_build; ctest -j; popd
