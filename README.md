# DFTB-Neural-Net

Install conda environment, the following packages would be needed:

```
conda create -n NAME python=3.6 numpy scikit-learn tensorflow-gpu keras cython pandas matplotlib seaborn ase pyyaml
```

Install via pip ordered-enum

# Installing DFTB+ in int-nano
1. git clone -b machine-learning https://github.com/tomaskubar/dftbplus.git 
2. cd dftbplus
3. git checkout e149ec0e8a1ca7ea0401d5f30f2245fc4a0a1d50
4. module load intel/19.0.5.281
5. module load cmake
6. export  INTEL_LICENSE_FILE=(ask for this info)
7. mkdir _build 
8. cd _build
9. ccmake -DCMAKE_Fortran_COMPILER=ifort -DCMAKE_C_COMPILER=icc -DBUILD_EXPORTED_TARGETS_ONLY=True -DCMAKE_TOOLCHAIN_FILE=../sys/intel.cmake ..
10. Note: While inside the ccmake interactive change the installation directory to the same level of _build
11. make 
12. make install 
