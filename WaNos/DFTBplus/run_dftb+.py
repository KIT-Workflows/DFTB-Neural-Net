import os,yaml,subprocess, tarfile
from ase.io import write, read
import dftbplus_functions as dftbplus

###############################################################################
###                                                                         ###
### prerequisite files: rendered_wano.yml, initial_structure.xyz (coord_0)  ###
###                                                                         ###
###############################################################################

def get_settings_from_rendered_wano():
    settings = {}
    with open('rendered_wano.yml') as infile:
        wano_file = yaml.full_load(infile)

    if wano_file['Type of calculation']['Method'] == "Single shot calculation":
        opt_options = False
    else:
        opt_options = True
    settings['title'] =wano_file['Title']
    settings['follow-up'] = wano_file['Follow-up calculation']
    settings['use old struct'] = wano_file['Molecular structure']['Use old structure']
    settings['use old charges'] = wano_file['Initial guess']['Use old charges']
    settings['charge'] = wano_file['Initial guess']['Charge']
    settings['multiplicity'] = wano_file['Initial guess']['Multiplicity']
    settings['scc'] = wano_file['DFTB options']['SCC calculation']
    settings['third'] = wano_file['DFTB options']['Third order']
    settings['max scc iter'] = wano_file['DFTB options']['Max SCC iterations']
    settings['skf'] = wano_file['DFTB options']['Slater-Koster parameters']
    settings['opt'] = opt_options
    settings['opt cyc'] = 100
    settings['opt driver'] = wano_file['Type of calculation']['Optimisation algorithm']
    settings['max opt cyc'] = wano_file['Type of calculation']['Max optimization cycles']
    settings['disp'] = wano_file['DFTB options']['Dispersion']

    settings['Simulation'] = wano_file['Type of calculation']['Method']
    settings['Functions']=wano_file['Type of calculation']['Symmetry functions file']
    settings['Model']=wano_file['Type of calculation']['Model']
    settings['Thermostat']=wano_file['Type of calculation']['Thermostat']
    settings['TimeStep']=wano_file['Type of calculation']['Time-Step fs'] # Time step for MD
    settings['MDTimeStep']=wano_file['Type of calculation']['MD-Time-Step fs'] # Time step for MD-ML
    settings['Steps']=wano_file['Type of calculation']['Steps']
    settings['MDsteps']=wano_file['Type of calculation']['MD-Steps']
    settings['InitTemp']=wano_file['Type of calculation']['Initial temperature K']
    settings['Thermostat']=wano_file['Type of calculation']['MD-Thermostat']

    return settings

def sanitize_multiplicity(multi,n_el):

    multi_new = multi
    multi_min = n_el%2+1

    if multi < 1:
        print('Attention: a multiplicity of %i is not possible.'%(multi))

    elif n_el%2 and multi%2: 
        print('Attention: a multiplicity of %i is not possible for an odd number of electrons.'%(multi))
        multi_new -= 1
    elif not n_el%2 and not multi%2: 
        print('Attention: a multiplicity of %i is not possible for an even number of electrons.'%(multi))
        multi_new -= 1

    if multi_new < multi_min: multi_new=multi_min
    if multi != multi_new: print('The multiplicity was set to %i by default'%(multi_new))
    
    return multi_new

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

    t1 = search_string_in_file("detailed.out","Atomic gross charges")[0][0]
    structure = read("initial_structure.xyz", format="xyz")
    tot_atoms = structure.get_global_number_of_atoms()
    n = t1+1

    with open ("detailed.out", 'r') as file:
        # Read each line in loop
        charges = file.readlines()[n:n+tot_atoms]

    charge_dict = {}
    for ii in range(tot_atoms):
        temp_str = charges[ii]
        result = [x.strip() for x in temp_str.split(',')]
        l = result[0].split()
        charge_dict[int(l[0])]=float(l[1])

    return charge_dict


if __name__ == '__main__':
    
    results_dict = {}
    settings = get_settings_from_rendered_wano()    

    results_dict['Slatko'] = settings['skf']
    results_dict['energy_unit'] = 'Hartree'
    results_dict['dftb_title'] = settings['title']

    struct_0_file = 'initial_structure.xyz'
    
    struct = read(struct_0_file)
    n_el = sum(struct.numbers)-settings['charge']
    settings['multiplicity'] = sanitize_multiplicity(settings['multiplicity'],n_el)

    opt_settings = settings.copy()
    settings['opt'] = False

    dftbplus.write_input(settings,struct_0_file)

    num_iter,done = 0,False
    while not done:
        done = dftbplus.run_dftb(False)
        if not done:
            num_iter += settings['max scc iter']
            if num_iter > settings['max scc iter']:
                print('SCC not converged in maximum number of iterations (%i)'%(settings['max scc iter']))
                results_dict['Converged']='No'
                break
        else: results_dict['Converged']='Yes'
                

    if settings["Simulation"] == "Structure optimisation":
        os.system('cp dftb.out dftb_0.out')
        os.system('cp charges.bin charges_0.bin')
        settings = opt_settings
        settings['use old charges'] = True
        dftbplus.write_input(settings,struct_0_file)
        num_cycles,done = 0,False    
        while not done:
            done = dftbplus.run_dftb(True)
            if not done:
                num_cycles += settings['opt cyc']
                if num_cycles > settings['max opt cyc']:
                    print('Structure optimisation not converged in maximum number of cycles (%i)'%(settings['max opt cyc']))
                    exit(0)
                if not os.path.isfile('intermediate_structure.xyz'):
                    os.rename('final_structure.xyz','intermediate_structure.xyz')
                    dftbplus.write_input(settings,'intermediate_structure.xyz')
                else: os.rename('final_structure.xyz','intermediate_structure.xyz')
    else: os.rename(struct_0_file,'final_structure.xyz')

    if settings["Simulation"] == "Machine Learning":
        with open('dftb.out') as infile:
            for line in infile.readlines():
                if line.startswith('MACHINE_LEARNING_ENERGY'):
                    results_dict['ML_correction']=float(line.split()[1])
                    break
    else: results_dict['ML_correction']=0

    with open('detailed.out') as infile:
        for line in infile.readlines():
            if line.startswith('Total energy'):
                results_dict['dftb_energy']=float(line.split()[2])
                break

    with open('dftb_plus_results.yml','w') as outfile: 
        yaml.dump(results_dict, outfile)

    output_files = ['dftb_pin.hsd','final_structure.xyz','charges.bin','rendered_wano.yml']
    for filename in output_files:
        if not os.path.isfile(filename): output_files.remove(filename)

    os.system('tar -cf results.tar.xz %s'%(' '.join(output_files)))
