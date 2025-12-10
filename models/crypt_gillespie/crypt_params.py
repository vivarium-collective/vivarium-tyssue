# cell types of the model:pc,gc,ent,dcs
cell_types = ("sc", "pc", "ent", "gc", "dcs")

# Construct a dictionary which convert cell kind to integer to store the in a numpy array format
cell_to_int = {j: i for i, j in enumerate(cell_types)}
int_to_cell = {i: j for i, j in enumerate(cell_types)}

######################################
# JUMPS
# the jump types of the model (in terms of identity of the new cell, whether it is a differentiation
# or a division)
jump_types = ("sc", "pc", "gc", "ex", "ent")

# Construct a dictionary which convert jump type to integer to store them in a numpy array format
jump_to_int = {i: x for x, i in enumerate(jump_types)}
int_to_jump = {x: i for x, i in enumerate(jump_types)}
