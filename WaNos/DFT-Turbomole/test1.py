 
# results_dict = {}

# with open('eiger.out') as infile: 
#         content = infile.readlines()
    
# temp_lst = content[12].split()

# if temp_lst[5] == 'H':
#     results_dict['homo'] = float(temp_lst[4])
#     results_dict['lumo'] = float(content[13].split()[4])
# else:
#     results_dict['homo'] = float(temp_lst[5])
#     results_dict['lumo'] = float(content[13].split()[5])
# results_dict['homo-lumo gap'] = float(content[14].split()[2])

# # with open('1eiger.out') as infile:
# #     #results_dict['homo-lumo gap'] = float(infile.readlines()[14].split()[2])
# #     #results_dict['lumo'] = float(infile.readlines()[13].split()[5])
# #     results_dict['homo'] = float(infile.readlines()[12].split()[5])

# print(results_dict)

import numpy as np

def get_hyper_polarizability():
    with open("escf.out", "r") as infile:
        escf_data = infile.readlines()

    # Electronic dipole hyperpolarizability
    begin_hyper_pol = int([i for i, line in enumerate(escf_data) if "Electronic dipole hyperpolarizability" in line][0]) + 4
    # read beta_x
    beta = np.zeros((3,3,3))
    lines = escf_data[begin_hyper_pol:begin_hyper_pol+9]
    line_split = [line.split() for line in lines]
    for i in range(3):
        beta[i,0,0] = float(line_split[0][i+(i+1)])
        beta[i,1,0] = float(line_split[1][i+(i+1)])
        beta[i,2,0] = float(line_split[2][i+(i+1)])
        beta[i,0,1] = float(line_split[3][i+(i+1)])
        beta[i,1,1] = float(line_split[4][i+(i+1)])
        beta[i,2,1] = float(line_split[5][i+(i+1)])
        beta[i,0,2] = float(line_split[6][i+(i+1)])
        beta[i,1,2] = float(line_split[7][i+(i+1)])
        beta[i,2,2] = float(line_split[8][i+(i+1)])
    return beta

a = get_hyper_polarizability()

print(a)

# with open("control", "r") as f:
#     lines = f.readlines()
# with open("1control", "w") as f:
#     for line in lines:
#         if line.strip("\n") != "$end":
#             f.write(line)
#     f.write('aaaaaa\n')
#     f.write('$end')