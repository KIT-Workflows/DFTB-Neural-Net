import yaml, os
from ase.io import write, read


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

def elec_charges():

    t1 = search_string_in_file("orca.out","MULLIKEN ATOMIC CHARGES")[0][0]
    structure = read("geometry")
    tot_atoms = structure.get_global_number_of_atoms()
    n = t1+1

    with open ("orca.out", 'r') as file:
        # Read each line in loop
        charges = file.readlines()[n:n+tot_atoms]

    charge_dict = {}
    for ii in range(tot_atoms):
        temp_str = charges[ii]
        result = [x.strip() for x in temp_str.split(',')]
        l = result[0].split()
        charge_dict[int(l[0])+1]=float(l[3])

    return charge_dict


def write_energy(name_file):
    
    file = open(name_file, "r")
    for line in file: 
        if "Total Energy" in line: 
            splitted = line.split()
            energy = float(splitted[-4])
    file.close() 
    return energy

def write_units(name_file):
    
    file = open(name_file, "r")
    for line in file: 
        if "Total Energy" in line: 
            splitted = line.split()
            units = str(splitted[-3])
    file.close() 
    return units


if __name__ == '__main__':
    
    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)

    name = str(wano_file['Title'])
    
    outfile = "orca.out"    
    
    results_dict = {}
    results_dict = wano_file
    results_dict['charges'] = elec_charges()
    results_dict["orca_energy"] = write_energy(outfile)
    results_dict["orca_title"]= name
    results_dict["orca_units"]= write_units(outfile)
    with open("orca_results.yml",'w') as out:
        yaml.dump(results_dict, out,default_flow_style=False)
