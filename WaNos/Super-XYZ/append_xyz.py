import yaml, os, tarfile, glob, shutil
import pandas as pd

if __name__ == '__main__':

    with open("rendered_wano.yml") as file:
        wano_file = yaml.full_load(file)

    with open("input-dict.yml") as file:
        s_file = yaml.full_load(file)

    structure_file = s_file["iter"] 

    fname =  wano_file['Molecules']
    energies =  wano_file['Table energy']

    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname)
        tar.extractall('my_folder')
        tar.close()
    elif fname.endswith("tar.xz"):
        tar = tarfile.open(fname)
        tar.extractall('my_folder')
        tar.close()    
    elif fname.endswith("tar"):
        tar = tarfile.open(fname)
        tar.extractall('my_folder')
        tar.close() 


    f = pd.read_csv(energies)
    if f['orca_title'].equals(f['dftb_title']):
        f.to_csv('rearranged-table.csv')    
    else:
        dftb = pd.concat([f["dftb_energy"], f["dftb_title"]], axis=1)
        orca = pd.concat([f["orca_energy"], f["orca_title"]], axis=1)
        arr = dftb["dftb_title"].to_numpy()

        new_orca=[]
        for i in range(len(arr)):
            new_orca.append(orca[orca.eq(arr[i]).any(1)])
        merged = pd.concat(new_orca,ignore_index=True)

        rearranged_table=pd.concat([dftb["dftb_energy"], dftb["dftb_title"],merged["orca_title"],merged["orca_energy"]],axis=1)
        rearranged_table.to_csv('rearranged-table.csv')

    mylist = rearranged_table['orca_title'].tolist()
    string = 'my_folder/'
    lst_xyz=[string + structure_file[s] for s in mylist]
    with open("appended_structures.xyz", "w") as outfile:
        for filename in lst_xyz:
            with open(filename) as infile:
                for line in infile:
                    if not line.isspace():
                        outfile.write(line)

    #os.remove(fname)
    shutil.rmtree("my_folder")