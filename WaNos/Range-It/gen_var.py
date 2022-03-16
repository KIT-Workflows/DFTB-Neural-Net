import numpy as np
from os import listdir
from os.path import isfile, join
import sys, os, tarfile, shutil, yaml


if __name__ == '__main__':
  ## inputs #############################
  with open('rendered_wano.yml') as file:
          wano_file = yaml.full_load(file)
  #######################################

  outdict = {}
  outdict["iter"] = []
  if wano_file["Float"]:
      var_begin = wano_file["VarF-begin"]
      var_end = wano_file["VarF-end"]
      var_npoints = wano_file["N-points"]
      
      step = (var_end-var_begin)/var_npoints
      for i in range(0,var_npoints):
          z_0 = var_begin + step*i
          z_0 = round(z_0,3)
          outdict["iter"].append(float(z_0))    
  elif wano_file["Int"]:
      var_begin = wano_file["VarI-begin"]
      var_end = wano_file["VarI-end"]
      step = wano_file["Step"]
      if step == 0:
          print( "N_points must fit in the interval" )
          sys.exit()
          
      for z_0 in range(var_begin,var_end, step):
          #z_0 = var_begin + int(step*i)
          outdict["iter"].append(z_0)
  elif wano_file["Structures"]:
    if os.path.isfile("tarfile.tar"):
      print ("File exist")
    else:
      print ("File not exist")

    file = tarfile.open('tarfile.tar', 'r')
    file.extractall()
    sub_folder = os.path.commonprefix(file.getnames())
    if os.path.isdir(sub_folder):
      onlyfiles = [sub_folder + '/'+ f for f in listdir(sub_folder) if isfile(join(sub_folder, f))]
      outdict["iter"] = onlyfiles #file.getnames()
      shutil.rmtree(sub_folder)
    else:
      outdict["iter"] = file.getnames()
  
  with open("output_dict.yml",'w') as out:
      yaml.dump(outdict, out,default_flow_style=False)

  #print(type(f_file["iter"]))
  # with open("output_dict.yml",'w') as out:
  #     yaml.safe_dump(outdict, out)