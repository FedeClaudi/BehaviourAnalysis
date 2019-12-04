import sys
sys.path.append("./")
from Utilities.constants import *

maze_designs = {0:"three_arms", 
                1:"asymmetric_long", 
                2:"asymmetric_mediumlong", 
                3:"asymmetric_mediumshort", 
                4:"symmetric", 
                -1:"nan"}


maze_names = {"maze1":"asymmetric_long", 
                "maze2":"asymmetric_mediumlong", 
                "maze3":"asymmetric_mediumshort", 
                "maze4":"symmetric"}
maze_names_r = {"asymmetric_long":"maze1", 
                "asymmetric_mediumlong":"maze2", 
                "asymmetric_mediumshort":"maze3", 
                "symmetric":"maze4"}

