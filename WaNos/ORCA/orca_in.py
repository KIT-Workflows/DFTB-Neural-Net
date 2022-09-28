from ase.calculators.orca import ORCA
from ase.build import molecule
from ase import io
import yaml


if __name__ == '__main__':
    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)

    def sanitize_multiplicity(multi,n_el):

        multi_new=multi
        multi_min=(((n_el+1)/2)*2)+1

        if multi < 1:
            print('Attention: a multiplicity of %i is not possible.'%(multi))
        elif n_el%2 and multi%2: 
            print('Attention: a multiplicity of %i is not possible for an odd number of electrons.'%(multi))
            multi_new-=1
        elif not n_el%2 and not multi%2: 
            print('Attention: a multiplicity of %i is not possible for an even number of electrons.'%(multi))
            multi_new-=1

        if multi_new < multi_min: multi_new=multi_min
        if multi != multi_new:
            print('The multiplicity was set to %i by default'%(multi_new))
        return int(multi_new)
    
    my_geo = io.read("geometry",format = 'xyz')
    label = wano_file["Title"]
    Functional = wano_file["Functional"]
    Basis_set = wano_file["Basis-set"]
    Charge = wano_file["Charge"]
    Multiplicity = wano_file["Multiplicity"]

    if wano_file["Functional"] == "DLPNO-CCSD(T)":
        calc = ORCA(label = 'orca',
                    maxiter = 0, 
                    orcasimpleinput = Functional + ' ' + Basis_set + ' '+Basis_set+'/C',
                    charge=Charge, mult=sanitize_multiplicity(Multiplicity,Charge),
                    task='gradient',
                    orcablocks='%pal nprocs 1 end',
                    tolerance='VeryTight')
    else:
        calc = ORCA(label = 'orca',
            maxiter = 0, 
            orcasimpleinput = Functional + ' ' + Basis_set,
            charge=Charge, mult=sanitize_multiplicity(Multiplicity,Charge),
            task='gradient',
            orcablocks='%pal nprocs 1 end',
            tolerance='VeryTight')

    my_geo.calc = calc

    calc.write_input(my_geo)
