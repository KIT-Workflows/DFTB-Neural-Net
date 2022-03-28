import os,yaml,subprocess
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
    
    opt_options = {wano_file['Type of calculation']['Method']:True}

    settings['follow-up'] = wano_file['Follow-up calculation']
    settings['use old struct'] = wano_file['Molecular structure']['Use old structure']
    settings['use old charges'] = wano_file['Initial guess']['Use old charges']
    settings['charge'] = wano_file['Initial guess']['Charge']
    settings['multiplicity'] = wano_file['Initial guess']['Multiplicity']
    settings['scc'] = wano_file['DFTB options']['SCC calculation']
    settings['scc iter'] = 100
    settings['max scc iter'] = wano_file['DFTB options']['Max SCC iterations']
    settings['skf'] = wano_file['DFTB options']['Slater-Koster parameters']
    settings['opt'] = opt_options
    settings['opt cyc'] = 100
    settings['opt driver'] = wano_file['Type of calculation']['Optimisation algorithm']
    settings['max opt cyc'] = wano_file['Type of calculation']['Max optimization cycles']

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

if __name__ == '__main__':
    
    struct_0_file = 'initial_structure.xyz'

    settings = get_settings_from_rendered_wano()

    if settings['follow-up']:
        os.system('mkdir old_results; tar -xf old_calc.tar.xz -C old_results')
        if settings['use old struct']: os.system('cp old_results/final_structure.xyz %s'%(struct_0_file))
        if settings['use old charges']: os.system('cp old_results(charges.bin .')
    
    struct = read(struct_0_file)
    n_el = sum(struct.numbers)-settings['charge']
    settings['multiplicity'] = sanitize_multiplicity(settings['multiplicity'],n_el)

    opt_settings = settings.copy()
    settings['opt'] = False

    print(settings)

    dftbplus.write_input(settings,struct_0_file)

    num_iter,done = 0,False
    while not done:
        done = dftbplus.run_dftb(False)
        if not done:
            num_iter += settings['scc iter']
            if num_iter > settings['max scc iter']:
                print('SCC not converged in maximum number of iterations (%i)'%(settings['max scc iter']))
                exit(0)
            if not settings['use old charges']:
                settings['use old charges'] = True
                dftbplus.write_input(settings,struct_0_file)
    
    if opt_settings['opt']:
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

    results_dict = {}
    results_dict['energy_unit'] = 'Hartree'
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
