# DFTB-Neural-Net

## Description
This workflow developed in the SimStack framework enables the correction of DFTB potential energy surfaces into ab-initio methods using the $\Delta$-learning neural network. This correction is implemented into the **DFTB+** code for further use on QM/MM simulations. The workflow enables users with no programming or machine learning background to take advantage of this correction and expand it into new systems.

In the folder WaNos there are several different WaNos: **DFT-Turbomole**, **DFTBplus**, **Mult-It**, **NN-Delta-ML**, **ORCA**, **Super-XYZ**, **Table-Generator** and **UnpackMol**, used to build the workflow. Below we describe each one and the main parameter exposed.

## In this workflow, we will be able to:

1. Load a set of molecular trial structures in a `.tar` file.
2. Unpack all molecular structures inside the AdvancedForEach loop control in the Simstack framework.
3. Compute the reference energy of the system.
4. Run the DFT calculations using ORCA or Turbomole codes, **ORCA** or **DFT-Turbomole** WaNos.
5. Run the DFTB calculations using BFTB+ code using **DFTBplus** WaNo. 
6. Arrange all the total energy values of the system in a table format (Table-Generator).
7. Append all files from the `.tar` input file in a specific order and shift the total DFT and DFTB energies from the previously computed reference energies.
8. Compute the $\Delta$ energy to generate the machine learning (ML) model and the learning report.
9. Apply the ML model to predict the $\Delta E$ for a similar system when stimulated via the DFTB method.

## Workflow for $\Delta$-learning neural network

![](ML-Fig1.png)

**Fig 1** _This workflow aims to create an ML model to correct DFTB method accuracy concerning the DFT level. It is composed of **DFT-Turbomole**, **DFTBplus**, **Mult-It**, **NN-Delta-ML**, **ORCA**, **Super-XYZ**, **Table-Generator** and **UnpackMol**  **WaNos** connected by the AdvancedFor loop control. (a) When the system is far from the equilibrium region, we compute the reference energy for DFT and DFTB levels. (b) In this step, the set of molecular structures in a `.tar` file is loaded, and a high throughput calculation (single shot) is performed for DFT and DFTB theory levels. The workflow automatically creates a machine learning report at the end of ML model generation._ 


## 1. Installation and dependencies
Here you will find the steps to install **DFTB+** code and python dependencies necessary to run the workflow. Of course, we assume that you already installed the **DFT** code. You can choose any option available in the [list of quantum chemistry codes.](https://en.wikipedia.org/wiki/List_of_quantum_chemistry_and_solid-state_physics_software.) In this case, we use **ORCA** or **Turbomole**.  

### 1.1 Conda Environment and Python dependencies
Install conda environment, the following packages would be needed:

```
conda create --name environment_name python=3.6 --file environment.yml
```
Install via pip ordered-enum

```
pip3 install ordered_enum
```
## Installing DFTB+ in int-nano machine
> :warning: **If you are installing DFTB+ on a different machine**: Be very careful and make the necessary adjustments.

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
## 2. Inputs and Outputs
### 2.1 Inputs
  - Molecular geometry far from the equilibrium region (as shown in **Fig 1 (a)**).
  - Set of randmized structures around the Potential energy surface (`.tar` file as shown in **Fig 1 (b)**).
### 2.2 Outputs
  - Machine learning report 
  - Pickle files of the machine learning model.

In the video _$\Delta$-learning workflow_, we teach how to set up the workflow to run it in SimStack. 
