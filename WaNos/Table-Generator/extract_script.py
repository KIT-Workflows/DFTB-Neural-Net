import pandas as pd
import os, yaml, h5py
import numpy as np

def save_dict_to_hdf5(dic, filename):

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def load_dict_from_hdf5(filename):

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')



def recursively_save_dict_contents_to_group( h5file, path, dic):

    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")        

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        #print(key,item)
        key = str(key)
        if isinstance(item, list):
            item = np.array(item)
            #print(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, and numpy.float64 types
        if isinstance(item, (np.int64, np.float64, str, np.float, float, np.float32,int)):
            #print( 'here' )
            h5file[path + key] = item
            if not h5file[path + key].value == item:
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save numpy arrays
        elif isinstance(item, np.ndarray):            
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype('|S9')
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key].value, item):
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        # other types cannot be saved and will result in an error
        else:
            #print(item)
            raise ValueError('Cannot save %s type.' % type(item))

def recursively_load_dict_contents_from_group( h5file, path): 

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans            


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

    files_ext = []

    for jj in range(len(wano_file["Files"])):
        Search_parameter_list = []
        for ii in range(len(wano_file["Files"][jj]["Search-Parameters"])):
            Search_parameter_list.append(wano_file["Files"][jj]["Search-Parameters"][ii]["var"])

        Search_file = wano_file["Files"][jj]["Search-in-file"]
        files_ext.append(Search_file)

        Search_file_list = []

        for file1 in os.listdir():
            if file1.endswith(Search_file):
                Search_file_list.append(file1)
        if jj == 0:
            name_dif_files.extend(Search_file_list)

        del_files.extend(Search_file_list)
        var_list.extend(Search_parameter_list)    

        temp_len = 0
        
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

    #files_ext
    all_data_dict = {}
    prefixed = []
    if wano_file["Assemble-files (HDF5)"]:
        for var in files_ext:
            temp_list = [filename for filename in os.listdir('.') if filename.endswith(var)]
            prefixed.append(temp_list)

        int_n = 0

        for var_ext in files_ext:
            #print(int_n)
            all_data_dict[var_ext] = {}
            for var_file in prefixed[int_n]:
                with open(var_file) as file:
                    temp_file = yaml.full_load(file)            
                #print([value for key, value in temp_file.items() if 'title' in key.lower()])
                temp_var1 = [value for key, value in temp_file.items() if 'title' in key.lower()][0] #temp_file["dftb_title"]
                all_data_dict[var_ext][str(temp_var1)] = temp_file 
            int_n += 1 

    with open('all_data.yml','w') as outfile: 
        yaml.dump(all_data_dict, outfile)


    if wano_file["Delete-Files"]:
        for file in del_files:
            if os.path.exists(file):
                os.remove(file)
