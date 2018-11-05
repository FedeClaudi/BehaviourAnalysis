from Tables_definitions import *

import datajoint as dj
import os


class PopulateDatabase:
    def __init__(self):
        """
        Collection of methods to populate the different 


        """
        print("""
        Ready to populate database. Available classes:
                * Mouse
                * Experiment
                * Surgery
                * Manipulation
                * Session
            Updating SESSION will also update:
                * NeuralRecording
                * BehaviourRecording
                * BehaviourTrial""")

    @staticmethod
    def mouse(filepath):
        print(" Update MOUSE table from excel file.")
        print(""" 
        Table definition:
            {}""".format(Mouse.definition))


    @staticmethod
    def experiment(filepath):
        print(" Update EXPERIMENT table from excel file.")
        print(""" 
        Table definition:
            {}""".format(Experiment.definition))

    @staticmethod
    def surgery(filepath):
        print(" Update SURGERY table from excel file.")
        print(""" 
        Table definition:
            {}""".format(Surgery.definition))

    @staticmethod
    def manipulation(filepath):
        print(" Update MANIPULATION table from excel file.")
        print(""" 
        Table definition:
            {}""".format(Manipulation.definition))

    @staticmethod
    def session(filepath):
        print(" Update SESSION table from excel file.")
        print(""" 
        Table definition:
            {}
        With subclasses:
            {}
            {}
            {}""".format(Session.definition, NeuronalRecording.definition, BehaviourRecording.definition,
                         BehaviourTrial.definition))




