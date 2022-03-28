from ase.io import write, read
import subprocess

def run_dftb(opt):
    with open('dftb.out','w') as dftb_out:
        dftb_process = subprocess.Popen('dftb+',stdout=dftb_out,stderr=subprocess.PIPE)
        out, err = dftb_process.communicate()
        
    done=False
    keyword={True:'Geometry converged',False:'Total Energy'}
    with open('dftb.out') as infile:
        for line in reversed(infile.readlines()):
            if keyword[opt] in line:
                done=True
                break
    return done

def write_input(settings,struct_file):
    struct = read(struct_file)
    bool2yn = {True:'Yes',False:'No'}
    sk_dict = {'3ob':'3ob-3-1','mio':'mio-1-1'}
    max_ang_mom = {}
    max_ang_mom['3ob'] = {"Br":"d","C":"p","Ca":"p","Cl":"d","F":"p","H":"s","I":"d","K":"p","Mg":"p","N":"p","Na":"p","O":"p","P":"d","S":"d","Zn":"d"}
    max_ang_mom['mio'] = {"C":"p","H":"s","N":"p","O":"p","P":"p","S":"p"}


    dftb_in = ""
    dftb_in += "Geometry = xyzFormat {\n  <<< \"%s\"\n}\n\n"%(struct_file)

    dftb_in += "Hamiltonian = DFTB {\n"
    if settings['scc']:
        dftb_in += "  Scc = Yes\n"
        dftb_in += "  MaxSCCIterations = %i\n"%(settings['scc iter'])
        dftb_in += "  Mixer = Broyden {}\n" #add different charge mixers as options in the .xml file
        dftb_in += "  ReadInitialCharges = %s\n"%(bool2yn[settings['use old charges']])
    dftb_in += "  Charge = %f\n"%(float(settings['charge']))

    #if settings['multiplicity'] > 1:
    #    dftb_in+="  SpinPolarisation = Colinear {\n"
    #    dftb_in+="    UnpairedElectrons = %f\n"%(float(settings['multiplicity']-1))
    #    dftb_in+="  }\n"

    dftb_in += "  SlaterKosterFiles = Type2FileNames {\n"
    ## hard coded location of skf-files on int-nano with dftb+ in qn0453's home
    dftb_in += "    Prefix = \"/home/ws/gt5111/DFTB+/dftbplus/external/slakos/origin/%s/\"\n"%(sk_dict[settings['skf']])
    dftb_in += "    Separator = \"-\"\n"
    dftb_in += "    Suffix = \".skf\"\n"
    dftb_in += "    LowerCaseTypeName = No\n"
    dftb_in += "  }\n"
    dftb_in += "  MaxAngularMomentum {\n"
    for element in list(dict.fromkeys(struct.get_chemical_symbols())):
        dftb_in += "    %s = \"%s\"\n"%(element,max_ang_mom[settings['skf']][element])
    dftb_in += "  }\n"
    dftb_in += "}\n\n"

    if settings['opt']:
        dftb_in += 'Driver = %s {\n'%(settings['opt driver'])
        dftb_in += '  MovedAtoms = 1:-1\n'
        dftb_in += '  MaxForceComponent = 1E-3\n'
        dftb_in += '  MaxSteps = %i\n'%(settings['opt cyc'])
        dftb_in += '  OutputPrefix = \"final_structure\"\n'
        dftb_in += '}\n\n'

    dftb_in += "Options {}\n\n"
    dftb_in += "Analysis {\n  CalculateForces = Yes\n}\n\n"
    dftb_in += "ParserOptions {\n  ParserVersion = 7\n}\n"

    with open('dftb_in.hsd','w') as outfile:
        outfile.write(dftb_in)




