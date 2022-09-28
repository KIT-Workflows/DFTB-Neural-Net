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

    if wano_file['Corrective model']=="True":
        f = pd.read_csv(energies)
        try:
            if f['orca_title'].equals(f['dftb_title']):
                f.to_csv('rearranged-table.csv',index=False)
                rearranged_table=f    
                arr=f["dftb_title"].to_numpy()
            else:
                dftb = pd.concat([f["dftb_energy"], f["dftb_title"]], axis=1)
                orca = pd.concat([f["orca_energy"], f["orca_title"]], axis=1)
                arr = dftb["dftb_title"].to_numpy()

                new_orca=[]
                for i in range(len(arr)):
                    new_orca.append(orca[orca.eq(arr[i]).any(1)])
                merged = pd.concat(new_orca,ignore_index=True)

                rearranged_table=pd.concat([dftb["dftb_energy"], dftb["dftb_title"],merged["orca_title"],merged["orca_energy"]],axis=1)

        except KeyError:
            if f['turbomole_title'].equals(f['dftb_title']):
                f.to_csv('rearranged-table.csv',index=False)
                rearranged_table=f    
                arr=f["dftb_title"].to_numpy()
            else:
                dftb = pd.concat([f["dftb_energy"], f["dftb_title"]], axis=1)
                turbomole = pd.concat([f["turbomole_energy"], f["turbomole_title"]], axis=1)
                arr = dftb["dftb_title"].to_numpy()

                new_turbomole=[]
                for i in range(len(arr)):
                    new_turbomole.append(turbomole[turbomole.eq(arr[i]).any(1)])
                merged = pd.concat(new_turbomole,ignore_index=True)

                rearranged_table=pd.concat([dftb["dftb_energy"], dftb["dftb_title"],merged["turbomole_title"],merged["turbomole_energy"]],axis=1)
            
    else:
        f = pd.read_csv(energies)
        rearranged_table=f    
        try:
            arr=f["orca_title"].to_numpy()
        except KeyError:
            arr=f["turbomole_title"].to_numpy()
        else:
            arr=f["dftb_title"].to_numpy()
    
    rearranged_table.to_csv('rearranged-table.csv',index=False)


    if wano_file['Get charges']:
        with open("asambeled.yml") as file:
            dicts = yaml.full_load(file)
        Cargas=[]
        for ele in arr:
            try:
                Cargas.append(dicts['dftb_plus_results.yml'][ele]['charges'].values())
            except KeyError:
                Cargas.append(dicts['turbomole_results.yml'][ele]['charges'].values())
            else:
                Cargas.append(dicts['orca_results.yml'][ele]['charges'].values())

        with open(r'cargas.txt', 'w') as fp:
            for i in range(0,len(Cargas)):
                for item in Cargas[i]:
                    fp.write("%s\n" % item)

    try:
        mylist = rearranged_table['orca_title'].tolist()
    except KeyError:
        mylist = rearranged_table['turbomole_title'].tolist()
    else:
        mylist = rearranged_table['dftb_title'].tolist()
    
    string = 'my_folder/'
    lst_xyz=[string + s for s in mylist]
    with open("appended_structures.xyz", "w") as outfile:
        for filename in lst_xyz:
            with open(filename) as infile:
                for line in infile:
                    if not line.isspace():
                        outfile.write(line)

    #os.remove(fname)
    shutil.rmtree("my_folder")