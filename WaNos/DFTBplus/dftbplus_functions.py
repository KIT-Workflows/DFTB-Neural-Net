import os,yaml,subprocess, tarfile
from ase.io import write, read
from ase.data import atomic_numbers
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

def symfunc(filename):
  a_file = open(filename)
  lines = a_file.readlines()
  return [line.strip() for line in lines]

def write_input(settings,struct_file):
    struct = read(struct_file)
    bool2yn = {True:'Yes',False:'No'}
    sk_dict = {'3ob':'3ob-3-1','mio':'mio-1-1'}
    max_ang_mom = {}
    max_ang_mom['3ob'] = {"Br":"d","C":"p","Ca":"p","Cl":"d","F":"p","H":"s","I":"d","K":"p","Mg":"p","N":"p","Na":"p","O":"p","P":"d","S":"d","Zn":"d"}
    max_ang_mom['mio'] = {"C":"p","H":"s","N":"p","O":"p","P":"p","S":"p"}
    Hubbard = {}
    Hubbard['3ob'] = {"Br":-0.0573,"C":-0.1492,"Ca":-0.0340,"Cl":-0.0697,"F" :-0.1623,"H" :-0.1857,"I" :-0.0433,"K" :-0.0339,"Mg" :-0.02,"N" :-0.1535,"Na" :-0.0454,"O" :-0.1575,"P" :-0.14,"S" :-0.11,"Zn" :-0.03}

    dftb_in = ""
    dftb_in += "Geometry = xyzFormat {\n  <<< \"%s\"\n}\n\n"%(struct_file)

    dftb_in += "Hamiltonian = DFTB {\n"
    if settings['scc']:
        dftb_in += "  Scc = Yes\n"
        dftb_in += "  MaxSCCIterations = %i\n"%(settings['scc iter'])
        dftb_in += "  Mixer = Broyden {}\n" #add different charge mixers as options in the .xml file
        dftb_in += "  ReadInitialCharges = %s\n"%(bool2yn[settings['use old charges']])
    dftb_in += "  Charge = %f\n"%(float(settings['charge']))

    if settings['third']:
        dftb_in += "  HCorrection = Damping {\n"
        dftb_in += "  Exponent = 4\n"
        dftb_in += "  }\n"
        dftb_in += "  ThirdOrder = Yes\n"
        dftb_in += "  HubbardDerivs {\n" 
        for element in list(dict.fromkeys(struct.get_chemical_symbols())):
            dftb_in += "    %s = %s\n"%(element,Hubbard[settings['skf']][element])#add different charge mixers as options in the .xml file
        dftb_in += "  }\n"

    dftb_in += "  SlaterKosterFiles = Type2FileNames {\n"
    ## hard coded location of skf-files on int-nano with dftb+ in qn0453's home
    dftb_in += "    Prefix = \"/home/ws/qs7669/Slatko/%s/\"\n"%(sk_dict[settings['skf']])
    dftb_in += "    Separator = \"-\"\n"
    dftb_in += "    Suffix = \".skf\"\n"
    dftb_in += "    LowerCaseTypeName = No\n"
    dftb_in += "  }\n"
    dftb_in += "  MaxAngularMomentum {\n"
    for element in list(dict.fromkeys(struct.get_chemical_symbols())):
        dftb_in += "    %s = \"%s\"\n"%(element,max_ang_mom[settings['skf']][element])
    dftb_in += "  }\n"

    if settings['disp'] == "D3":
        dftb_in+= "    Dispersion = SimpleDftD3 {\n"
        dftb_in+= "    a1 = 0.5719\n"
        dftb_in+= "     a2 = 3.6017\n"
        dftb_in+= "     s6 = 1.0\n"
        dftb_in+= "     s8 = 0.5883\n"
        dftb_in+= "     }\n\n"
    if settings['disp'] == "LennardJones":
        dftb_in+= "Dispersion = LennardJones {\n"
        dftb_in+= "Parameters = UFFParameters {}\n"
        dftb_in+=  "              }\n\n"

    if settings["Simulation"] == "Single shot calculation":
        dftb_in+="  }\n"

    if settings["Simulation"] == "Machine Learning":
        fname =  settings['Model']
        if fname.endswith("tar"):
            tar = tarfile.open(fname)
            tar.extractall()
            tar.close()

        dftb_in+="  MachineLearning = NeuralNet {\n"
        dftb_in+="     SymmetryFunctions {\n"
        dftb_in+="     Neighboursearching = Yes\n"
        dftb_in+="     AtomicNumber = {\n"
        for element in list(dict.fromkeys(struct.get_chemical_symbols())):
            dftb_in+="    %s = %s\n"%(element, atomic_numbers[element])
        dftb_in+="  }\n"
        dftb_in+="  \n".join(map(str, symfunc(settings['Functions'])))
        dftb_in+="\n"
        dftb_in+="  }\n"
        dftb_in+=" NeuralNetworkFiles = Type2Filenames {\n"
        dftb_in+="    Prefix = \"Model/\"\n"
        dftb_in+="    Suffix = \"-subnet.param\"\n"
        dftb_in+="  }\n"
        dftb_in+="  }\n"
        dftb_in += "}\n\n"

    if settings["Simulation"] == "Structure optimisation":
        dftb_in+='}\n\n'
        dftb_in += 'Driver = %s {\n'%(settings['opt driver'])
        dftb_in += '  MovedAtoms = 1:-1\n'
        dftb_in += '  MaxForceComponent = 1E-3\n'
        dftb_in += '  MaxSteps = %i\n'%(settings['opt cyc'])
        dftb_in += '  OutputPrefix = \"final_structure\"\n'
        dftb_in += '}\n\n'

    if settings["Simulation"] == "Molecular Dynamics":
        dftb_in+='}\n\n'
        dftb_in+='Driver = VelocityVerlet{\n'
        dftb_in+='  TimeStep [fs] = %f\n'%(settings['TimeStep'])
        dftb_in+='  Thermostat = %s{\n'%(settings['Thermostat'])
        dftb_in+='      InitialTemperature [Kelvin] = %i\n'%(settings['InitTemp'])
        dftb_in+="  }\n"
        dftb_in+='  Steps = %i\n'%(settings['Steps'])
        dftb_in+='  MovedAtoms = 1:-1\n'
        dftb_in+='  MDRestartFrequency = 100\n'
        dftb_in+='}\n\n'
        
    
    if settings["Simulation"] == "MD-Machine Learning":
        fname =  settings['Model']
        if fname.endswith("tar"):
            tar = tarfile.open(fname)
            tar.extractall()
            tar.close()

        dftb_in+="  MachineLearning = NeuralNet {\n"
        dftb_in+="     SymmetryFunctions {\n"
        dftb_in+="     Neighboursearching = Yes\n"
        dftb_in+="     AtomicNumber = {\n"
        for element in list(dict.fromkeys(struct.get_chemical_symbols())):
            dftb_in+="    %s = %s\n"%(element, atomic_numbers[element])
        dftb_in+="  }\n"
        dftb_in+="  \n".join(map(str, symfunc(settings['Functions'])))
        dftb_in+="\n"
        dftb_in+="  }\n"
        dftb_in+=" NeuralNetworkFiles = Type2Filenames {\n"
        dftb_in+="    Prefix = \"Model/\"\n"
        dftb_in+="    Suffix = \"-subnet.param\"\n"
        dftb_in+="  }\n"
        dftb_in+="  }\n"

        dftb_in+="}\n\n"

        dftb_in+='Driver = VelocityVerlet{\n'
        dftb_in+='  TimeStep [fs] = %f\n'%(settings['MDTimeStep'])
        dftb_in+='  Thermostat = %s{\n'%(settings['Thermostat'])
        dftb_in+='      InitialTemperature [Kelvin] = %i\n'%(settings['InitTemp'])
        dftb_in+="  }\n"
        dftb_in+='  Steps = %i\n'%(settings['MDsteps'])
        dftb_in+='  MovedAtoms = 1:-1\n'
        dftb_in+='  MDRestartFrequency = 100\n'
        dftb_in+='}\n\n'

    

    dftb_in += "Options {}\n\n"
    dftb_in += "Analysis {\n  CalculateForces = Yes\n}\n\n"
    dftb_in += "ParserOptions {\n  ParserVersion = 7\n}\n"

    with open('dftb_in.hsd','w') as outfile:
        outfile.write(dftb_in)







