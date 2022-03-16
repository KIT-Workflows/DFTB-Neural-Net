from ase.calculators.orca import ORCA
from ase.build import molecule
from ase import io
import yaml


if __name__ == '__main__':
    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)
    
    my_geo = io.read('geometry',format = 'xyz')
    label = wano_file["label"]
    Functional = wano_file["Functional"]
    Basis_set = wano_file["Basis-set"]
    Charge = wano_file["Charge"]
    Multiplicity = wano_file["Multiplicity"]

    #atoms = molecule('H2O')

    calc = ORCA(label = 'orca',
                maxiter = 0, 
                orcasimpleinput = Functional + ' ' + Basis_set,
                charge=Charge, mult=Multiplicity,
                task='gradient',
                orcablocks='%pal nprocs 1 end',
                tolerance='VeryTight')

    my_geo.calc = calc

    calc.write_input(my_geo)
