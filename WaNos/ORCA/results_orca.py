import yaml, os


def write_energy(name_file):
    
    file = open(name_file, "r")
    for line in file: 
        if "Total Energy" in line: 
            splitted = line.split()
            energy = float(splitted[-4])
    file.close() 
    
    return energy


if __name__ == '__main__':
    
    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)

    var_name = wano_file["var-name"]

    koringa_flag = wano_file["Import KORINGA"]
    
    file_outfile = "orca.out"    
    energy = write_energy(file_outfile)
    
    results_dict = {}
    results_dict["orca_energy"] = energy 
    if koringa_flag:
        with open('KORINGA.yml') as file:
            koringa_file = yaml.full_load(file)
        results_dict["orca_structure"]= koringa_file["structure"]
    
    with open("orca_results.yml",'w') as out:
        yaml.dump(results_dict, out,default_flow_style=False)
    