import matplotlib.pyplot as plt
import yaml
import  numpy as np

def find_by_key(data, target):
    for key, value in data.items():
        if isinstance(value, dict):
            yield from find_by_key(value, target)
        elif key == target:
            yield value

def call_find_by_key(data, target):

    y = []
    x = None
    
    for x in find_by_key(data, target):
        y.append(x) 
    
    return x

if __name__ == '__main__':

    with open('Table-dict.yml') as file:
        Table_file = yaml.full_load(file)
    
    a_exp = 2.462 # Angst
    c_exp = 6.707 # Angst
    
    delta_a = 100*((np.array(Table_file["a"]) - a_exp)/a_exp)
    delta_c = 100*((np.array(Table_file["c"]) - c_exp)/c_exp)

    encut_key = "ENCUT"
    kpt_key = "NKPTS"

    if encut_key in Table_file:
        x_key = encut_key
    else:
        x_key = kpt_key
    
    x_var = Table_file[x_key]
    

    with open("output_dict.yml",'w') as out:
        yaml.dump(Table_file, out,default_flow_style=False)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x_var, delta_a,'-ob')
    #axs[0].set_xlim(0, 2)
    axs[0].set_xlabel(x_key)
    axs[0].set_ylabel('$\Delta a(\%)$')
    #axs[0].grid(True)

    axs[1].plot(x_var, delta_c,'-or')
    axs[1].set_ylabel('$\Delta c(\%)$')
    axs[1].set_xlabel(x_key)
    
    fig.tight_layout()
    fig.savefig('Fig1.png')

    # fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    # ax.plot([0,1,2], [10,20,3])
    # fig.savefig('Fig1.png')   # save the figure to file
    # plt.close(fig)    