import yaml, os, tarfile, glob, shutil

if __name__ == '__main__':

    with open("rendered_wano.yml") as file:
        wano_file = yaml.full_load(file)

    if wano_file["Multivariable-mode"]:
        geom_structure_xyz = int(wano_file['Structures-int'])
        input_file = wano_file['Input-file']
        with open(input_file) as file:
            inp_struct = yaml.full_load(file)

        var_1 = inp_struct['iter'][geom_structure_xyz]
    else:
        geom_structure_xyz = str(wano_file['Structures-name'])
        var_1 = geom_structure_xyz
    
    fname = wano_file['Structures']

    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname)
        tar.extract(var_1)
        tar.close()
    elif fname.endswith("tar.xz"):
        tar = tarfile.open(fname)
        tar.extract(var_1)
        tar.close()    
    elif fname.endswith("tar"):
        tar = tarfile.open(fname)
        # tar.extractall()
        tar.extract(var_1)
        tar.close() 

    os.rename(var_1,'Mol_geom.xyz')
    print(fname)

    unpackmol_results = {}
    unpackmol_results["structure"] = var_1 
    with open("unpackmol_results.yml",'w') as out:
        yaml.dump(unpackmol_results, out,default_flow_style=False)

    # for filename in glob.glob(geom_structure_xyz + "*"):
    #     os.remove(filename) 
    shutil.rmtree('Structures')
    os.remove(fname)