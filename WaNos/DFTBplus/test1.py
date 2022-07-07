import yaml, os, shutil
from ase import Atoms
from ase.io import read


def search_string_in_file(file_name, string_to_search):

    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    
    line_number = 0
    list_of_results = []
    
    ''' Open the file in read only mode'''

    with open(file_name, 'r') as read_obj:

        '''Read all lines in the file one by one'''

        for line in read_obj:
            
            '''For each line, check if line contains the string'''
            
            line_number += 1
            if string_to_search in line:
                
                '''If yes, then add the line number & line as a tuple in the list'''
                
                list_of_results.append((line_number, line.rstrip()))
   
    '''Return list of tuples containing line numbers and lines where string is found'''
   
    return list_of_results

t1 = search_string_in_file("detailed.out","Atomic gross charges")[0][0]

structure = read("final_structure.xyz", format="xyz")
tot_atoms = structure.get_global_number_of_atoms()
print()

n = t1+1
lines = []
with open ("detailed.out", 'r') as file:
    # Read each line in loop
    charges = file.readlines()[n:n+tot_atoms]

charges_dict = {}
for ii in range(tot_atoms):
    temp_str = charges[ii]
    result = [x.strip() for x in temp_str.split(',')]
    l = result[0].split()
    charges_dict[int(l[0])]=float(l[1])

print(charges_dict)