import pandas as pd
import os, yaml

def round_float_list(float_list, decimal_points):
    float_list = [round(float(item),decimal_points) for item in float_list]
    return float_list

def add_values_in_dict(sample_dict, key, list_of_values):
    ''' Append multiple values to a key in 
        the given dictionary '''
    if key not in sample_dict:
        sample_dict[key] = list()
    sample_dict[key].extend(list_of_values)
    return sample_dict

if __name__ == '__main__':
    # Input data
    decimal_points = 4
    Table_dict = {}
    var_list = []
    name_dif_files = []
    del_files = []
    data = {}
    
    df = pd.DataFrame()

    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)

    for jj in range(len(wano_file["Files"])):
        Search_parameter_list = []
        for ii in range(len(wano_file["Files"][jj]["Search-Parameters"])):
            Search_parameter_list.append(wano_file["Files"][jj]["Search-Parameters"][ii]["var"])

        Search_file = wano_file["Files"][jj]["Search-in-file"]
        Search_file_list = []

        for file1 in os.listdir():
            if file1.endswith(Search_file):
                Search_file_list.append(file1)
        if jj == 0:
            name_dif_files.extend(Search_file_list)

        del_files.extend(Search_file_list)
        var_list.extend(Search_parameter_list)    

        temp_len = 0
        
        print(var_list)
        for parameter_var in Search_parameter_list:
            temp_dict = {}
            Table_var = []
            for file in Search_file_list:
                with open(file) as f_file:
                    n_file = yaml.full_load(f_file)
                temp_var1 = n_file[parameter_var]
                Table_var.append(temp_var1)
            temp_dict[parameter_var] = Table_var
            data.update(temp_dict)
        df = df.from_dict(data, orient='columns')
    
    ''' Sorted by the first parameter ''' 
    
    df.sort_values(by=var_list[0], ascending=False, na_position='first', inplace=True)
    df.to_csv('Table-var.csv')
    data = df.to_dict(orient="list")

    try:   
        with open("Table-dict.yml",'w') as out:
            yaml.dump(data, out,default_flow_style=False)
    
    except IOError:
        print("I/O error") 

    if wano_file["Delete-Files"]:
        for file in del_files:
            if os.path.exists(file):
                os.remove(file)