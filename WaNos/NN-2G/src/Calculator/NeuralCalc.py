"""
Module: NNCalculator

Purpose: For the Defining of a base class used for defining other neural Network
Calculators.


"""
import os
import numpy as np
import pandas as pd
import h5py
import src.Calculator.ModelIO
import sys
import src.Calculator.ForceNum

from ase.calculators.calculator import Calculator, all_changes
from src.Utils.DirNav import get_project_dir


class NeuralCalc(Calculator):
    """
    An Abstract Base Class that can be generalized for any neural network calculator.
    Cannot used directly to genreate instancesself.
    This is an ASE calculator, can be used to predict energy, force, stress
    for ase.atoms.
    """
    def __init__(self, calc, **kwards):
        pass


    def initialize(self, atoms):

        """
        Convert the atom into the input

        """
        pass

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes= all_changes):
        pass



    """
    ##########

    Methods for Importing the Neural Network Model.

    ##########
    """

    def import_model(self, save_dir):
        """
        Import the model from the file saved in the save_dir
        """

        pass

    def update_model(self):
        """
        Update the model from the trained Neural Network Model from
        a model when running (Not FileIO)
            Args:
                model: keras.models
        See NeuralCalc.get_model function.
        """
        pass


    ### Export the model used by the instance
    def export_model(self):
        """
        Export the model to the given path as a .h5 file.
        """
        pass


    def get_model(self):
        """
        Return the model when running .
        See NeuralCalc.update_model function.
        """
        pass
