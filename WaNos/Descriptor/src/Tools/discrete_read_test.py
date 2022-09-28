"""
This file contains the old testing module for the reading function used before by reading a discrete molecular
configuration. 

Currently I have no plan to maintain it. 

"""
from discrete_read import get_forces
from discrete_read import get_energy
get_energy("/root/samples/samples", "1.out", 6)


##### Load serious of atoms structure form the given directory ### 


# Create empty list to contain all the samples
# Maintain: Improve the list to be the Numpy array or pandas DataFrame
md_samplesArr = pd.Series([]);
md_energyArr  = pd.Series([]);
md_Mulliken = pd.Series([]);
md_ForceArrX = pd.Series([])
md_ForceArrY = pd.Series([])
md_ForceArrZ = pd.Series([])

file_dir = "/root/samples/samples"




for file in os.listdir(file_dir):
    if file.endswith(".xyz"):
        fileName = os.path.join(file_dir, file)
        fileIndexStr = os.path.splitext(file)[0]
        fileIndex = int( fileIndexStr )
        
        fileOut  = fileIndexStr + ".out"
        
        fileAtom = read(fileName)
        
        
        
        
        md_energyArr[fileIndex] = get_energy(file_dir, fileOut, len(fileAtom))
        md_samplesArr[fileIndex] = fileAtom

        
        md_Mulliken[fileIndex] = get_mulliken_charge(file_dir, fileOut, len(fileAtom))
        md_ForceArrX[fileIndex],md_ForceArrY[fileIndex],md_ForceArrZ[fileIndex]  = get_forces(file_dir, fileOut, len(fileAtom))

        
        
#md_sampleArr = np.array(md_samples, dtype=object)
#md_energyArr = np.array(md_energy)
#md_MullikenArr  = np.array(md_Mulliken)


print(md_samplesArr.shape)
print(md_energyArr.shape)

print(md_ForceArrX.shape)

forceArrX, forceArrY, forceArrZ = get_forces(file_dir, "0.out", 3)
print(forceArrX)

md_MullikArr =  get_mulliken_charge(file_dir, "5.out", 3)
print(md_MullikArr)
