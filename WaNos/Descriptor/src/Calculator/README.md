# File Structure for Calculator Folder #

## Calculator ##
* **NeuralCalc.py**:_(Abstract)_ Have the abstract class NeuralCalc is the parent class of
all of the following calculators
* **ModelCalc.py**: _(Numerical)_ Have the calculator with numerical force calculation. (Only works for
glycine). This calculator is used for a benchmark calculation of glycine molecule
to verify the calculation results.
* **CompileCalc.py**: _(Compiled)_ Have an optimized calculator with compiled calculations.
* **CompileVecCalc.py**: _(Vectorized)_ is the vectorized version of the CompileCalc.

## Symmetry Function Calculation (Individual) ##
_Notice_: Individual means that those code only calculates the symmetry function
vector of a single conformation of the molecular system, rather than for
many different conformations of the molecular system at once.

* **SymmFuncIndPython.py**: _(For Understanding)_ Python implementation of the calculation of symmetry
function vectors. NOT CURRENTLY IN USE, but is easier to understand the algorithm
than the Cython version.
* **SymmFuncIndCython.py**: Cython implementation of the calculation of symmetry
function vectors. Used for `CompileCalc.py`.  Used
to calculate energy.
* **SymmFuncIndVec.py**: _(For Understanding)_ Cython implementation of the
vectorized calculation of symmetry function vectors. NOT CURRENTLY IN USE, but
is easier to understand the algorithm than the Cython version.
* **SymmFuncIndVecCython.py**: Cython implmentation of the vectorized calculation
of symmetry function vectors. Used for `CompileVecCalc.py`. Used to calculate
energy.

## Symmetry Function Derivative Calculation (Individual) ##
_Notice_: Same Notice as above.
* **SymmDerivIndCython.py**: Cython implementation of the calculation for symmetry
function derivative calculation. Used for the `CompileCalc.py`. Used to calculate
the force.
* **SymmDerivIndVecCython.py**: Cython implementation of the vectorized calculation
of the symmetry function derivatives. Used for the `CompileVecCalc.py`. Used to
calculate the force.

## Force Calculation (Individual) ##
* **ForceInd.py**: Calculation of the Force for individual conformations. Uses
`SymmDerivIndCython.py`. Used for the `CompileCalc.py`
* **ForceIndVec.py**: Vectorized calculation of the Force for individual conformations.
Uses `SymmDerivIndVecCython.py`. Used for the `CompileVecCalc.py`.
* **ForceNum.py**: Numerical Calculation of the force for individual conformations.
Uses 3-point formula for derivative. Used for the `ModelCalc.py`.

## Compiled Related Calculations##
The Compilation process is dividided into two sub process.
1. **Compile**. For MD simulation of the same molecular system, many
information are repeated **throughout the entire simulation**.
Including the atomic indices, atomic element, and symmetry function hyperparameters.
The "Compile" process encode those information into arrays so that it is easier
to do the for-loop.

2. **Pre-calculate**: For **1 Step of the MD simulation**, many calculations are
repeated. For instance, the symmetry function and its derivatives looks very
similar. To reduce the amount of calculation, a few calculations are first
calculated and stored in the cache, and then called when the calculation needs it.
The use of cache dramatically improves the calcualtion efficiency.



* **CompileArr.py**: Use for the "Compile" process by `CompileCalc.py`
* **CompileVec.py**: The vectorized version of `CompileArr.py`, used by
'CompileVecCalc.py'.
* **PreCalcVec.py**: Used for the "Pre-calculate" process by  `CompileVeCalc.py`.



## Other Files ##
* **ASEDFTBMod.py**: A modified version of ASE DFTB Calculator. It supports the
updated version of DFTB. The original ASE DFTB calculator has some problem
with the IO.
* **ModelIO.py**: Used to load the Neural Network (NN) weight.
* **CompileCython.sh**: Used to compile the Cython code to C code. Set it up in
`setup.py`
* **Setup.py**: Just as the `Makefile` for Python to set up the Cython compilation.
* **UninstallCython.sh**: Delete all the C code of the Cython code.
* **speedtest.sh**: Used to use `snakeviz` to do the performance test.

## Calculation for An Ensemble of conformations (In Progress)##
_Notice_: All previous files deals with the calculation of a single conformation
of the molecular system. However, for the training, it is necessray to provide the
symmetry function and derivatives for an ensemble of different conformations of
molecules. It is planned to prepare those calculations while currently it is
still in progress. Those files usually have `Arr` in their file names.

* **PreCalcArrVec.py**
* **SymmFuncArrVecCython.py**



# Explaination #
## Compiled Version ##
`CompileCalc.py` and `CompileVecCalc.py` both uses the "compiled version" to speed up
the calculation process. "Compiled Version" refers to reduce the amount of repeated
calculation by storing those calculation results in the cache. For the force calculation,
many of the calculations are repeated and therefore the compiled version would
dramatically increase the speed of the calculation.


## Vectorized Version ##
Different from the `CompileCalc.py`, `CompileVecCalc.py` uses vectorization to
further speed up some of the calculation. Therefore, a few calculations are done
by placing the data into numpy arrays (vectors). Numpy improves the memory locality
and solved the problem of memory fragmentation by other Python code. Whether the vectorization
would improve the code is not entirely sure since `CompileCalc.py` uses the
Cython to do the for loop calculation, while `CompileVecCalc.py` uses the numpy for
the for loop calculations.



## Cython ##
Most calculation-intensive code are now implemented by the Cython.
