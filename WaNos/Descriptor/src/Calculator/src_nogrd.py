import numpy as np
import pandas as pd
import yaml
import re

##### IMPORT Atomic Simulation Environment Functions #####
import ase
import ase.build
from ase import Atom
from ase.atoms import Atoms
from ase.calculators.dftb import Dftb
from ase.io import read, write
import sys

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from ordered_enum import ValueOrderedEnum
import os.path as path

p = path.abspath(path.join(__file__ ,"../../.."))

with open(p+ "/rendered_wano.yml") as file:
    wano_file = yaml.full_load(file)
geom_filename = wano_file['Geometries']
"""
Global Variable Management:
"""

Mulliken_Feat = 0 # Represent the feat_idx for the Mulliken Charge in the feature array.

# https://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
# Only in case where numpy.__version__ < 1.7,
# Handle the error function
def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

dictonary_elements={"1"	:	"H"	,
    "2"	:	"He"	,
    "3"	:	"Li"	,
    "4"	:	"Be"	,
    "5"	:	"B"	,
    "6"	:	"C"	,
    "7"	:	"N"	,
    "8"	:	"O"	,
    "9"	:	"F"	,
    "10"	:	"Ne"	,
    "11"	:	"Na"	,
    "12"	:	"Mg"	,
    "13"	:	"Al"	,
    "14"	:	"Si"	,
    "15"	:	"P"	,
    "16"	:	"S"	,
    "17"	:	"Cl"	,
    "18"	:	"Ar"	,
    "19"	:	"K"	,
    "20"	:	"Ca"	,
    "21"	:	"Sc"	,
    "22"	:	"Ti"	,
    "23"	:	"V"	,
    "24"	:	"Cr"	,
    "25"	:	"Mn"	,
    "26"	:	"Fe"	,
    "27"	:	"Co"	,
    "28"	:	"Ni"	,
    "29"	:	"Cu"	,
    "30"	:	"Zn"	,
    "31"	:	"Ga"	,
    "32"	:	"Ge"	,
    "33"	:	"As"	,
    "34"	:	"Se"	,
    "35"	:	"Br"	,
    "36"	:	"Kr"	,
    "37"	:	"Rb"	,
    "38"	:	"Sr"	,
    "39"	:	"Y"	,
    "40"	:	"Zr"	,
    "41"	:	"Nb"	,
    "42"	:	"Mo"	,
    "43"	:	"Tc"	,
    "44"	:	"Ru"	,
    "45"	:	"Rh"	,
    "46"	:	"Pd"	,
    "47"	:	"Ag"	,
    "48"	:	"Cd"	,
    "49"	:	"In"	,
    "50"	:	"Sn"	,
    "51"	:	"Sb"	,
    "52"	:	"Te"	,
    "53"	:	"I"	,
    "54"	:	"Xe"	,
    "55"	:	"Cs"	,
    "56"	:	"Ba"	,
    "57"	:	"La"	,
    "58"	:	"Ce"	,
    "59"	:	"Pr"	,
    "60"	:	"Nd"	,
    "61"	:	"Pm"	,
    "62"	:	"Sm"	,
    "63"	:	"Eu"	,
    "64"	:	"Gd"	,
    "65"	:	"Tb"	,
    "66"	:	"Dy"	,
    "67"	:	"Ho"	,
    "68"	:	"Er"	,
    "69"	:	"Tm"	,
    "70"	:	"Yb"	,
    "71"	:	"Lu"	,
    "72"	:	"Hf"	,
    "73"	:	"Ta"	,
    "74"	:	"W"	,
    "75"	:	"Re"	,
    "76"	:	"Os"	,
    "77"	:	"Ir"	,
    "78"	:	"Pt"	,
    "79"	:	"Au"	,
    "80"	:	"Hg"	,
    "81"	:	"Tl"	,
    "82"	:	"Pb"	,
    "83"	:	"Bi"	,
    "84"	:	"Po"	,
    "85"	:	"At"	,
    "86"	:	"Rn"	,
    "87"	:	"Fr"	,
    "88"	:	"Ra"	,
    "89"	:	"Ac"	,
    "90"	:	"Th"	,
    "91"	:	"Pa"	,
    "92"	:	"U"	,
    "93"	:	"Np"	,
    "94"	:	"Pu"	,
    "95"	:	"Am"	,
    "96"	:	"Cm"	,
    "97"	:	"Bk"	,
    "98"	:	"Cf"	,
    "99"	:	"Es"	,
    "100"	:	"Fm"	,
    "101"	:	"Md"	,
    "102"	:	"No"	,
    "103"	:	"Lr"	,
    "104"	:	"Rf"	,
    "105"	:	"Db"	,
    "106"	:	"Sg"	,
    "107"	:	"Bh"	,
    "108"	:	"Hs"	,
    "109"	:	"Mt"	,
    "110"	:	"Ds"	,
    "111"	:	"Rg"	,
    "112"	:	"Cn"	,
    "113"	:	"Uut"	,
    "114"	:	"Fl"	,
    "115"	:	"Uup"	,
    "116"	:	"Lv"	,
    "117"	:	"Uus"	,
    "118"	:	"Uuo"	}

reversed_dictionary = {v: k for k, v in dictonary_elements.items()}

def supported(geom_filename):
    struct = ase.io.read(geom_filename, format = "xyz")
    elem = list(dict.fromkeys(struct.get_chemical_symbols()))
    dict2 = {x:reversed_dictionary[x] for x in elem}
    print(dict2.items())
    return dict2

supported_elements = ValueOrderedEnum('supported_elements', supported(p+'/'+ geom_filename))

def at_idx_map_generator(atoms):
    """
    For the given ase.Atoms object, generate an element list
    of all the atomic indexes that belong to the same element.
        Args:
            atoms: ase.atoms object
        Outputs:
            at_idx_map: maps element symbols to array of atom indices
    """
    assert isinstance(atoms, Atoms), 'at_idx_map_generator: Input file should be atoms'
    nAtoms = len(atoms)
    elements = set([atom.symbol for atom in atoms])
    ne = len(elements)
    tmp_dict = {element : [] for element in elements}
    for idx, atom in enumerate(atoms):
        tmp_dict[atom.symbol].append(idx)
    sorted_dict = {}
    for el in sorted(supported_elements):
        sorted_dict[el.name] = np.array(tmp_dict[el.name])

    return sorted_dict

def at_ele_arr_generator(atoms):
    """Loop through all the elements in the given molecule (atoms) and then
    generate the atom element map (at_ele_map)
        at_ele_arr[index] = element str ('H')

            Args:
                atoms: ase.Atoms object that represent the molecule.

            Outputs:
                at_ele_arr: np.array of the atom
                            where the element is represented as a element_idx

    Explaination:
    The purpose is to use the array to increase the performance of the Python Code.
    The objective is to use at_ele_map, xyz_arr to entirely replace the ase.Atoms,

    J. Zhu's experiments has shown that the previous methods of access the xyz
    position and the element by the ase.Atoms takes a lot of time in __getitem__()
    method. Replacing that with array will increase the time.


    Comments:
    1. May Combine with at_idx_map generator.
    This is for temporary testing, at_ele_map may entirely replace at_idx_map
    in the future.

    """

    """
    Element Idx Array Declaration:

    ! Important

    This array is used to represent the element with the element idx .
    ele_dict[element str] = element index (ele_idx)

    (For convinience it is declared inside the generator )

    """
    #TODO: Design a better way to represent the element
    ele_dict =   {
                'H': 0,
                'O': 1,
                'C': 2,
                'N': 3,
                'S': 4,
                'P': 5,
                'X': 6,
                'Y': 7,
                'Z': 8
    }

    if isinstance(atoms, Atoms) == False:
        raise TypeError('at_ele_arr_generator: Input file should be atoms')

    n_atoms = len(atoms)

    at_ele_map = np.empty(shape=n_atoms, dtype= np.str )
    at_ele_arr = np.empty(shape = n_atoms, dtype = np.int32)

    for at_idx in np.arange(n_atoms):
        element = atoms[at_idx].symbol
        at_ele_map[at_idx] = element
        at_ele_arr[at_idx] = ele_dict[element]

    return at_ele_map, at_ele_arr, ele_dict


def ang_list_generator(element_list):
    """Generate all the combo of the element str

            Args:
                element_list: list of all the element str (should not have repeated)
                    e.g. ['H', 'O', 'N', 'C']

            Outputs:
                ang_list: contains all the combo of the element str
                    e.g. ['HH', 'HO', "OO"]

            Raises:
                ValueError: If the element list contains repeated elements

    """
    n_ele = len(element_list)
    ang_list  = []

    for idx_1, ele_str_1 in enumerate(element_list):
        ang_list.append(ele_str_1 + ele_str_1)
        for idx_2 in range(idx_1 +1, n_ele, 1):
            ele_str_2 = element_list[idx_2]
            if ele_str_1 == ele_str_2:
                raise ValueError("element_list contains repeated elements")
            pair_str = ele_str_1 + ele_str_2
            ang_list.append(pair_str)

    return ang_list










def xyzArr_generator(md_samplesArr):
    """
    For the given array of the all the configurations (md_samplesArr), generate an 1D array of all the
    atomic cartesian coordinates [x,y,z] for all the atoms sampled.

        Args:
            md_samplesArr: an array of type ase.Atoms.

        Output:
        nAtoms: Number of atoms (integer)
        xyzArr: list that includes all atoms in all given configurations (input)
                has shape (nb_samples * n_atoms, 3)   3 refers to (x,y,z)

    Comment:
    for current version, xyzArr_generator assumes that the number of atoms does not change
    for all the sample input.
    """
    if isinstance(md_samplesArr[0], ase.Atoms) == False:
        raise TypeError('xyzArr_generator: Input file should be ase.Atoms')

    n_samples = len(md_samplesArr)


    # Maintaincence:

    n_atoms = md_samplesArr[0].get_positions().shape[0]
    #xyzArr = np.zeros(shape=(nSamples * nAtoms, 3))
    xyz_list = [[0.0,0.0,0.0]] * (n_samples * n_atoms)

    count = 0
    for fileIndex, Atoms in md_samplesArr.items():
        #import pdb; pdb.set_trace()
        temp_positions = Atoms.get_positions()
        for j in np.arange(temp_positions.shape[0]):
            xyz_list[count * n_atoms + j] = temp_positions[j]
        count += 1

    xyz_arr = np.array(xyz_list, dtype=np.float64)
    return n_atoms, xyz_arr



# The calling of this function before the data processing made the code less general
# At least not suitable for the complex system, and only suitable for systems of same amount of atoms.


# Maintain: A better version of distances_from_xyz is required
#           It only works if the number of atoms does not change over time
def distances_from_xyz(xyz, Natom):
    """calculate distances from coordinates
    # Arguments
        Natom: number (nb) of atoms; integer
        xyz: coordinates; 2D numpy array of shape (nb_samples * nb_atoms, 3)

    # Returns
        distances_df: distance values; pandas dataframe of shape (nb_samples, nb_distances),
                    column names as [(0,1),(0,2),...]
    """
    Nsamples = xyz.shape[0]//Natom
    distances = np.zeros(shape=(Nsamples, int(Natom*(Natom-1)/2)))
    count = 0
    # Going through All the atoms
    for i in range(Natom):
        atom1_array = xyz[Natom*np.arange(Nsamples) + i]
        # Going through the
        for j in range(i+1,Natom):
            atom2_array = xyz[Natom*np.arange(Nsamples) + j]
            distances[:,count] = np.sqrt(np.sum((atom1_array - atom2_array)**2, axis = 1))
            count += 1
    distances_df = pd.DataFrame(data= distances, columns = [(a,b) for a in range(Natom) for b in range(a+1,Natom)])
    return distances_df


"""
#####################

    Symmetry Function
    Related Functions

#####################
"""

####################################################
# Radial Filter function is the Radial Symmetry Function
# For given set of (Rs,eta), calculate the Radial component of G value for all the neighbour atoms


def cutoff(Rc, Rij):
    """ Cutoff function for both angular and radial symmetry function
        Args:
            Rc: the cutoff radius
            Rij: (arr) distance between two atoms of index i and j.

        Outputs:
            cutoff_arr: the value of the cut off function
            f = 0.5 * (Cos[ pi * Rij / Rc] + 1)

    """
    const = 1 / Rc

    cutoff_arr = 0.5 * ( np.cos( np.pi * Rij * const ) + 1) *  (Rij < Rc)

    return cutoff_arr


def radial_filter(Rs, eta, Rij):
    """radial filter for symmetry functions
    # Arguments
        Rs, eta: radial symmetry function parameters; float
        Rij: distance values between two given atoms i and j;
                1D numpy array of length Nsamples

    # Returns
        G_rad_ij: radial filter values; 1D numpy array of length nb_samples
    """
    G_rad_ij = np.exp(-eta * (Rij-Rs)**2)
    return G_rad_ij


# Angular Filter Function is the Angular Symmetry Function
# For a given set of (eta, zeta, lambd), caclaulte
# The augular component of G value for all the neighbour atoms

# To change the filter function, modify it here.
def angular_filter(Rij, Rik, Rjk, eta, zeta, lambd):
    """angular filter for angular symmetry functions
    # Arguments
        eta, zeta, lambd: angular symmetry function parameters
        Rij, Rik, Rjk: distances among three atoms i, j, k; 1D arrays of length nb_samples

    # Returns
        G_ang_ij: angular filter values; 1D numpy array of length nb_samples

    """
    cos_angle = (Rij**2 + Rik**2 - Rjk**2)/(2.0 * Rij * Rik)
    rad_filter = np.exp(-eta*(Rij + Rik + Rjk)**2)
    G_ang_ijk = 2**(1.0-zeta) * (1.0 + lambd * cos_angle)**zeta * rad_filter


    return G_ang_ijk

# symmetry function calculates a matrix of all the G values. (G -> vector)
# For all the atoms for a given input file.
# Improvement: Separate the individual symmetry function section (so easy to code)
def symmetry_function(distances, at_idx_map, Gparam_dict):
    """
    calculate symmetry functions from distances for the set of molecules

        Args:
            distances: distance values; pandas dataframe of shape (nb_samples, nb_distances)
            at_idx_map: a mapping between atom types and atom indexes; dictionary
            Gparam_dict: symmetry function parameters;
                            dictionary with 1st layer keys  = atom types,
                                2nd layer keys = symmetry function types (radial or angular)
                                values = 2D arrays of sym. function parameters of
                                shape (nb_sym_functions, nb_filter_parameters)

        Outputs:
        Gfunc_data: symmetry function values;
                        dictionary with 1st layer keys = atom types,
                            2nd layer keys = atom indexes,
                            values = 2D arrays with shape=(nb_samples, nb_sym_functions)
    """
    #pdb.set_trace()
    Nsamples = distances.shape[0]
    Gfunc_data = pd.Series([])

    # This for loop goes through elements
    # Are together
    for at_type in at_idx_map.keys():
        Gparam_rad = Gparam_dict[at_type]['rad']
        Gparam_ang = Gparam_dict[at_type]['ang']

        Gfunc_data[at_type] = pd.Series([])

        rad_count = sum([Gparam_rad[t].shape[0] for t in Gparam_rad.keys()])
        ang_count = sum([Gparam_ang[t].shape[0] for t in Gparam_ang.keys()])


        ## This for loop goes through all the atoms (belong to the same element)
        for at1 in at_idx_map[at_type]:
            Gfunc_data[at_type][at1] = np.zeros((Nsamples, rad_count + ang_count))

            G_temp_count = 0

            # radial components
            for at2_type in Gparam_rad.keys():
                comp_count =  Gparam_rad[at2_type].shape[0]
                G_temp_component = np.zeros((Nsamples, comp_count))

                for count, values in enumerate(Gparam_rad[at2_type]):
                    #pdb.set_trace()
                    for at2 in at_idx_map[at2_type][at_idx_map[at2_type]!=at1]:
                        # Problem Located: The following code does not work.
                        # The dist does not put into the allowance.
                        dist = tuple(sorted([at1, at2]))
                        #pdb.set_trace()
                        R12_array = distances[dist].values[:Nsamples]
                        # values[0] = Rs, values[1] = eta (integer, not array), values[2] = Rc (cutoff)
                        # Then Calculate the radial symmetric function -> value of G.
                        rad_temp = radial_filter(values[0], values[1], R12_array) * cutoff(values[2], R12_array)
                        G_temp_component[:,count] += rad_temp

                Gfunc_data[at_type][at1][:,G_temp_count:G_temp_count+comp_count] = G_temp_component
                G_temp_count += comp_count

            # ======================
            # angular components
            for atAatB_naive in Gparam_ang.keys():
                atAatB_type=re.findall('[A-Z][^A-Z]*', atAatB_naive)
                comp_count = Gparam_ang[atAatB_naive].shape[0]
                G_temp_component = np.zeros((Nsamples, comp_count))

                # This for loop goes through all 'HH', 'HO' combo?
                for count, values in enumerate(Gparam_ang[atAatB_naive]):
                    atA_list = at_idx_map[atAatB_type[0]][at_idx_map[atAatB_type[0]]!=at1]
                    for atA in atA_list:
                        dist_1A = tuple(sorted([at1, atA]))
                        R1A_array = distances[dist_1A].values[:Nsamples]


                        if atAatB_type[0] == atAatB_type[1]:
                            atB_list = at_idx_map[atAatB_type[1]][(at_idx_map[atAatB_type[1]]!=at1) & (at_idx_map[atAatB_type[1]]>atA)]
                        else:
                            atB_list = at_idx_map[atAatB_type[1]][(at_idx_map[atAatB_type[1]]!=at1)]

                        for atB in atB_list:
                            dist_1B = tuple(sorted([at1, atB]))
                            dist_AB = tuple(sorted([atA, atB]))
                            R1B_array = distances[dist_1B].values[:Nsamples]
                            RAB_array = distances[dist_AB].values[:Nsamples]

                            if np.any(R1B_array == 0):
                                import pdb; pdb.set_trace()
                            if np.any(RAB_array == 0):
                                import pdb; pdb.set_trace();


                            ang_temp = angular_filter(R1A_array, R1B_array, RAB_array, values[0], values[1], values[2]) \
                                        * cutoff(values[3], R1A_array) * cutoff(values[3], R1B_array) * cutoff(values[3], RAB_array)

                            G_temp_component[:, count] += ang_temp

                Gfunc_data[at_type][at1][:,G_temp_count:G_temp_count+comp_count] = G_temp_component
                G_temp_count += comp_count
    return Gfunc_data





"""
#####################

    Feature Array
    Functions.

#####################
"""








# Maintain: Assume that pandas Dataframe will be mutated
def feat_function(at_idx_map, md_Feat, nb_samples):
    """
    Based on the element type and the atomic index from at_idx_map, and know the number of samples (nb_samples),
    generate one feature dataset (Feat_data) for the given feature array (md_Feat)

        Args:
            at_idx_map: a mapping between atom types and atom indexes; dictionary
            md_Feat: an (pd.Series) array of all the Features for all input
                    in the shape
                    md_Feat[fileIndex][atom_idx] = Feat value
            feat_idxArr: the feat idx array for the given fit.
            nb_feat: number of feats add this time


        Output:
        Feat_data: symmetry function values;
                        dictionary with 1st layer keys = atom types,
                            2nd layer keys = atom indexes,
                            values = 2D arrays with shape=(nb_samples, nb_feat)

    """

    Feat_data = pd.Series([])

    # Maintainence:
    # Advantage of using nested pd.Series is that it is fast
    # Disadvantage is that it is harder for debugging
    # If needed, use nested dictionary and finally import into the DataFrame object.

    # This for loop goes through elements
    for at_type in at_idx_map.keys():
        Feat_data[at_type] = pd.Series([])

        ## This for loop goes through all the atoms (belong to the same element)
        for at1 in at_idx_map[at_type]:
            ## Maintain:  A Redesign of Data Structure might be / or not necessary if the No. of Atoms Varies
            ###           Assume number of atoms does not change
            ###           Assume atom indexes are the same throughout MD.
            Feat_data[at_type][at1] = np.zeros( (nb_samples, 1)) # because we are adding 1 feature at one time
            for sampleIndex in np.arange(nb_samples):
                Feat_data[at_type][at1][sampleIndex] = md_Feat[sampleIndex][at1]
    return Feat_data

## Merge the feat with the Gfunc, (or even feat with feat)
## Currently will just pass by value.
## Maintain: In the future test for using np.dataframe for Gfunc
##           Assume the number of atoms are constant.


## Assume Gfunc_data, Feat_data have same number of samples
def Gfunc_merge(Gfunc_data, Feat_data, at_idx_map, nb_samples):
    """
    Allows to merge the symmetry function vector (Gfunc_data) with the feature
    array (Feat_array) so to add more features based on the atomic index map
    (at_idx_map)
    """
    newGfunc_data = pd.Series([])
    for at_type in at_idx_map.keys():
        newGfunc_data[at_type] = pd.Series([])
        for at in at_idx_map[at_type]:
            # Maintain: Assumes that atoms are the same
            newGfunc_data[at_type][at] = np.append(Gfunc_data[at_type][at], Feat_data[at_type][at], axis=1)
    return newGfunc_data






def feat_scaling_func(Feat_data, at_idx_map, nb_feat):
    """
    Define one scaling based on the MinMaxScaler for the training set.
    Could Apply for both G-symmetry function and the Independent Features

        Args:
            Feat_data_train: Strucutre for the feat data Feat_data_train['element'][atom]
            at_idx_map: Atom Index Map
            train_idx: the indices used for the training


        Output:
            Return the Feat_scaler array
            Feat_scaler['element'][atom][Feature Number]
    """
    Feat_scaler= {}


    for at_type in at_idx_map.keys():
        Feat_scaler[at_type] = {}
        for at in at_idx_map[at_type]:
            Feat_scaler[at_type][at] = StandardScaler(with_mean=False);
            Feat_scaler[at_type][at].fit(Feat_data[at_type][at])
            Feat_data[at_type][at]= Feat_scaler[at_type][at].transform(Feat_data[at_type][at])
            #import pdb; pdb.set_trace()

    return Feat_scaler








"""
#####################

    Symmetry Function
    Debugging &
    Visualization Toolkits

#####################
"""


def sym_func_show(Gfunc_data, at_type, at, fileIndex):
    """
    Generate a plot of the symmetry function (Gfunc_data) of an atom with index (at)
    Raise Error When the array have nan.

        Args:
                Gfunc_data: Symmetry Function Vector
                at_type: str that represents element
                at: atomic index
                fileIndex: index for the file


    """

    NSamples = Gfunc_data[at_type][at].shape[0]
    NSymFunc = Gfunc_data[at_type][at].shape[1]

    print(NSamples, NSymFunc)

    x_index = np.arange(NSymFunc)

    plt.plot(x_index, Gfunc_data[at_type][at][fileIndex])
    plt.show()


def sym_func_debug(Gfunc_data):
    """
    Args:
            Gfunc_data:

    Return: the indexes of symmetry function that does not work.
    """
    pass
