# %%
import sys
sys.path.append('./')

from Utilities.imports import *

import control

# %%
"""
The system's readout is a linear combination of the activity of the two hemispheres of the GRN. 

y = 0: no turning
y > 0: turn right
y < 1: turn left
"""

# %%
# Define matrices
ymtx = [1, -1] # readout -> sum of GRN activity
grn_input = np.array([[0, 1, 1, 1], [1, 0, 1, 1]]) # controlateral proj from SC to GRN and  bilateral proj from MOS
sc_input = np.array([[1, 0], [0, 1]]) # bilateral proj from MOS to SC

grn_recurr = np.array([[1, 0], [0, 1]])

# Define variables
pos = [0, 0]
ori = 0

# Define functions
def move(pos, ori):
    x = np.cos(ori)
    y = np.sin(ori)
    return [pos[0]+x, pos[1]+y]


def turn(ori, input): 
    return ori + np.matmul(ymtx, input)

def reset():
    return [0, 0], 0


# %%

# %% Simulate
pos, ori = reset()

nsteps = 100
ix = np.arange(nsteps)
L_mos = np.sin(np.pi*ix/float(nsteps/2))
R_mos = np.sin(np.pi*ix/float(nsteps/2))

inputs = list(np.vstack((L_mos, R_mos)).T)

X, Y = [0], [0]
for inp in inputs:
    # Compute mos input to SC
    mos_to_sc = np.matmul(sc_input, inp)

    # Compute MOS input to GRN
    inp_to_grn = np.matmul(grn_input, list(inp)+list(mos_to_sc))

    # Compute how GRN state results in turning
    ori = turn(ori, inp_to_grn)
    pos = move(pos, ori)
    X.append(pos[0]); Y.append(pos[1])

f, ax = create_figure(subplots=False, figsize=(5, 5))
ax.scatter(X, Y, c=np.arange(len(X)))





# %%
# Check controllability 
C = control.ctrb(grn_recurr, grn_input)
rank = np.linalg.matrix_rank(C)

"""
    A system is only controllable if the rank of C is equal to N [the dimension of the the state (2 in this case)]
"""
if rank == 2:  
    print("system is controllable")
else:
    print("system is not controllable")



# %%
