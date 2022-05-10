import yaml, os


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
    results_dict["orca_energy"] = write_energy(outfile)
    results_dict["orca_structure"]= name
    results_dict["orca_units"]= write_units(outfile)
    with open("orca_results.yml",'w') as out:
        yaml.dump(results_dict, out,default_flow_style=False)