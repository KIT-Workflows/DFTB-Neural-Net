import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import  numpy as np
import pandas as pd
import pickle


if __name__ == '__main__':

        with open("rendered_wano.yml") as file:
                wano_file = yaml.full_load(file)

        geom_filename = wano_file['Geometries']
        neurons=wano_file['Neurons']
        act_funct=wano_file['Activation function']
        layerss=wano_file['Hidden Layers']
        lr=wano_file['Learning rate']


        data_energy = wano_file['Energies'] 
        data_set = pd.read_csv(data_energy)

        ref_orca_file = wano_file['Ref-Energy'] 
        ref = pd.read_csv(ref_orca_file)

        if wano_file['Corrective model']:
                ref_dftb_file = wano_file['Ref-Energy-DFTB'] 
                ref_dftb = pd.read_csv(ref_dftb_file)
                try:
                        data_set['Delta'] = (data_set['orca_energy'] - data_set['dftb_energy'] + ref_dftb['dftb_energy'].sum() - ref['orca_energy'].sum())*627.5
                except KeyError:
                        data_set['Delta'] = (data_set['turbomole_energy'] - data_set['dftb_energy'] + ref_dftb['dftb_energy'].sum() - ref['turbomole_energy'].sum())*627.5
        
                plt.figure(figsize=(10,8), dpi=300)
                p = sns.displot(data=data_set['Delta'], kde=True)
                plt.xlabel('$\Delta E_{ab initio-DFTB}$ [kcal/mol]',fontsize = 16)
                plt.ylabel('Number of Structures',fontsize = 16)
                plt.title('Dataset histogram', fontsize=20)
                plt.tick_params(axis='both', labelsize=14)
                plt.tight_layout()
                plt.savefig('Histogram.png')

                plt.figure(figsize=(10,8), dpi=300)
                plt.scatter(((data_set['dftb_energy']-ref_dftb['dftb_energy'].sum())*627.5), data_set['Delta'])
                plt.title('Correlation plot', fontsize=20)
                plt.xlabel('DFTB energy [kcal/mol]',fontsize = 16)
                plt.ylabel('$\Delta E_{ab initio-DFTB}$ [kcal/mol]',fontsize = 16)
                plt.tick_params(axis='both', labelsize=14)
                plt.savefig('Delta_correlation.png')

                plt.figure(figsize=(10,8), dpi=300)
                try:
                        plt.scatter(((data_set['dftb_energy']-ref_dftb['dftb_energy'].sum())*627.5), (data_set['orca_energy']-ref_dftb['orca_energy'].sum())*627.5)
                except KeyError:
                        plt.scatter(((data_set['dftb_energy']-ref_dftb['dftb_energy'].sum())*627.5), (data_set['turbomole_energy']-ref_dftb['turbomole_energy'].sum())*627.5)
                plt.title('Correlation plot', fontsize=20)
                plt.xlabel('DFTB energy [kcal/mol]',fontsize = 16)
                plt.ylabel('ab initio energy [kcal/mol]',fontsize = 16)
                plt.tick_params(axis='both', labelsize=14)
                plt.savefig('Correlation.png')

                y_pred=pickle.load( open( "Model/y_pred.pkl", "rb" ))
                y_obs=pickle.load( open( "Model/y_obs.pkl", "rb" ))
                x2 = pd.Series(pickle.load( open( "Model/y_pred.pkl", "rb" ) ), name="Predicted")
                x1 = pd.Series(pickle.load( open( "Model/y_obs.pkl", "rb" ) ), name="Observed")
                x = np.random.uniform(min(x1),max(x1),size=100)
                plt.figure(figsize=(10,8), dpi=300)
                plt.plot(x, x, dashes=[10, 5, 20, 5],color='navy',linewidth=1)
                plt.scatter(x2, x1)
                plt.title('Test Set fitting of $\Delta E_{ab initio-DFTB}$ ', fontsize=20)
                plt.ylabel('Reference energy [kcal/mol]', fontsize = 16)
                plt.xlabel('Prediction energy [kcal/mol]',fontsize = 16)
                plt.tick_params(axis='both', labelsize=14)
                plt.savefig('TestSet.png')
        
        else:
                try:
                        data_set['Delta'] = (data_set['orca_energy'] - ref['orca_energy'].sum())*627.5
                except KeyError:
                        data_set['Delta'] = (data_set['turbomole_energy'] - ref['turbomole_energy'].sum())*627.5
                else:
                        data_set['Delta'] = (data_set['dftb_energy'] - ref['dftb_energy'].sum())*627.5

                plt.figure(figsize=(10,8), dpi=300)
                p = sns.displot(data=data_set['Delta'], kde=True)
                plt.xlabel('Energy [kcal/mol]',fontsize = 16)
                plt.ylabel('Number of Structures',fontsize = 16)
                plt.title('Dataset histogram', fontsize=20)
                plt.tick_params(axis='both', labelsize=14)
                plt.tight_layout()
                plt.savefig('Histogram.png')

                y_pred=pickle.load( open( "Model/y_pred.pkl", "rb" ))
                y_obs=pickle.load( open( "Model/y_obs.pkl", "rb" ))
                x2 = pd.Series(pickle.load( open( "Model/y_pred.pkl", "rb" ) ), name="Predicted")
                x1 = pd.Series(pickle.load( open( "Model/y_obs.pkl", "rb" ) ), name="Observed")
                x = np.random.uniform(min(x1),max(x1),size=100)
                plt.figure(figsize=(10,8), dpi=300)
                plt.plot(x, x, dashes=[10, 5, 20, 5],color='navy',linewidth=1)
                plt.scatter(x2, x1)
                plt.title('Test Set fitting of energy', fontsize=20)
                plt.ylabel('Reference energy [kcal/mol]', fontsize = 16)
                plt.xlabel('Prediction energy [kcal/mol]',fontsize = 16)
                plt.tick_params(axis='both', labelsize=14)
                plt.savefig('TestSet.png')


