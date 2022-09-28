import numpy as np
import matplotlib.pyplot as plt


def get_pe_arr(traj):
    """Read the ase trajectory and plot the energy

    """
    n_samples = len(traj)
    ke_arr = np.zeros(n_samples)
    pe_arr = np.zeros(n_samples)
    for i in np.arange(n_samples):
        ke_arr[i] = traj[i].get_potential_energy()
        pe_arr[i] = traj[i].get_kinetic_energy()

    return ke_arr, pe_arr


def get_pe_arr_batch(traj, idx, idx_size):
    """Get the Potential Energy and Kinetic Energy from the samples
    for a batch.

    """
    ke_arr = np.zeros(idx_size)
    pe_arr = np.zeros(idx_size)
    for i in idx:
        ke_arr[i] = traj[i].get_potential_energy()
        pe_arr[i] = traj[i].get_kinetic_energy()

    return ke_arr, pe_arr



def calc_pe_comp(traj, calc1, calc2, calc3, start, end, interval=1):
    """Using the given calculator (calc), calculate the potential energy
    of given trajectory from index start to end (not include end)

    """
    if interval <= 0:
        raise ValueError("Interval should be greater than 0")

    if start <= 0 or end > len(traj):
        raise ValueError("Start and End is invalid")



    arr_len = int((end-start) / interval )
    pe_arr_1 = np.zeros(arr_len)
    ke_arr_1 = pe_arr_1.copy()
    pe_arr_2 = pe_arr_1.copy()
    ke_arr_2 = pe_arr_1.copy()
    pe_arr_3 = pe_arr_1.copy()
    ke_arr_3 = pe_arr_2.copy()

    count = 0
    for idx in range(start, end, interval):
        temp_atom = traj[idx].copy()
        temp_atom.set_calculator(calc1)
        pe_arr_1[count] = temp_atom.get_potential_energy()
        ke_arr_1[count] = temp_atom.get_kinetic_energy()


        temp_atom.set_calculator(calc2)
        pe_arr_2[count] = temp_atom.get_potential_energy()
        ke_arr_2[count] = temp_atom.get_kinetic_energy()

        temp_atom.set_calculator(calc3)
        pe_arr_3[count] = temp_atom.get_potential_energy()
        ke_arr_3[count] = temp_atom.get_kinetic_energy()
        count += 1

    return pe_arr_1, pe_arr_2, pe_arr_3, ke_arr_1, ke_arr_2, ke_arr_3



def print_md_energy(ke_arr_input, pe_arr_input, start_step):
    """Print the MD energy values from start_step

            Args:
                ke_arr_input: np.array of all the kinetic energy results.
                pe_arr_input: np.array of all the potential energy results.
                start_step: the step that started to display. All steps
                            before are ignored.
                            Used to ignored the initial steps for equilibrium.

            Outputs:
                No. Plot the graph.


    Comments:
    ke and pe array should have the same size.

    """


    n_samples = len(ke_arr_input)

    if start_step > n_samples:
        raise ValueError("print_md_energy: start_step is more than the md step")
    index_arr_input = np.arange(n_samples)
    index_arr = index_arr_input[start_step:-1]
    ke_arr = ke_arr_input[start_step:-1]
    pe_arr = pe_arr_input[start_step:-1]


    total_arr = ke_arr + pe_arr

    ke_fig = plt.figure()
    plt.plot(index_arr, ke_arr, label='Kinetic Energy (eV)')
    plt.legend()
    plt.xlabel("Step ")
    plt.ylabel("Energy (eV)")
    plt.show()

    pe_fig = plt.figure()
    plt.plot(index_arr, pe_arr, label='Potential Energy (eV)')
    plt.legend()
    plt.xlabel("Step ")
    plt.ylabel("Energy (eV)")
    plt.show()


    total_fig = plt.figure()
    plt.plot(index_arr, total_arr, label='Total Energy (eV)')
    plt.legend()
    plt.xlabel("Step ")
    plt.ylabel("Energy (eV)")
    plt.show()



def plot_energy(pe_arr_input, start_step, title, xlabel='Step', ylabel='Energy (eV)' ):
    """Print the MD energy values from start_step

            Args:
                pe_arr_input: np.array of all the potential energy results.
                start_step: the step that started to display. All steps
                            before are ignored.
                            Used to ignored the initial steps for equilibrium.

            Outputs:
                No. Plot the graph.


    Comments:
    ke and pe array should have the same size.

    """


    n_samples = len(pe_arr_input)

    if start_step > n_samples:
        raise ValueError("print_md_energy: start_step is more than the md step")
    index_arr_input = np.arange(n_samples)
    index_arr = index_arr_input[start_step:-1]
    pe_arr = pe_arr_input[start_step:-1]




    pe_fig = plt.figure()
    plt.plot(index_arr, pe_arr, label='Potential Energy (eV)')
    plt.legend()
    plt.title(title)
    plt.xlabel("Step ")
    plt.ylabel("Energy (eV)")
    plt.show()


def printenergy(a):
    """Print the Kinetic Energy and Potential Energy during the ASE Molecular
    Dynamics Simulation.

    """
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
