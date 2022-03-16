import pandas as pd

d=pd.read_csv("Table-var.csv")

dftb = pd.concat([d["dftb_energy"], d["dftb_structure"]], axis=1)
orca = pd.concat([d["orca_energy"], d["orca_structure"]], axis=1)

arr = dftb["dftb_structure"].to_numpy()

new_orca=[]
for i in range(len(arr)):
    new_orca.append(orca[orca.eq(arr[i]).any(1),header=None])