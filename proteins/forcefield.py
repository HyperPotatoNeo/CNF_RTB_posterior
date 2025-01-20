from openmm.app import PDBFile, Simulation
from openmm import LangevinIntegrator
from openmm.unit import kelvin, picosecond, femtosecond

from openmm import app 

from openmmforcefields.generators import SystemGenerator

from torchdrug import utils 

# Load the PDB file
pdb_filename = utils.download("https://files.rcsb.org/download/2LWZ.pdb", "./")
#pdb_filename = 'template.pdb'  # PDB file for the protein's topology
#pdb_filename = './protein_8.pdb'
pdb = PDBFile(pdb_filename) # check pbd, does it have Hydrogens? I think yes but double check

# Define force fields (ff14SB for proteins and TIP3P for water model)
# you don t need water, most likely you don't have them in 
forcefields = ['amber/protein.ff14SB.xml'] #, 'amber/tip3p_standard.xml'] 

# Generate the OpenMM system with the forcefields
# C: this will automatically define a small forcefield register too , it fine
# C: for your things you can also do: forcefield = ForceField(forcefields); forcefield.create_system(pdb.topology)

#system_generator = SystemGenerator(forcefields=forcefields)

system_generator = SystemGenerator(
    forcefields=['amber/protein.ff14SB.xml'],
    nonbondedMethod=app.NoCutoff
)

system = system_generator.create_system(pdb.topology)

# Define an integrator (I think we need this just as a placeholder) C: yes
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2*femtosecond)

# Create a simulation context
simulation = Simulation(pdb.topology, system, integrator) # C: you might want to define the platform if cpu or gpu here 

# --------
# if we have positions from FoldFlow2 (given a fixed topology), we can use directly the positions from the list
# of residues (openmm accepts Numpy) C: watchout that openmm works in nanometers

positions = pdb.positions  # or what's in output from FoldFlow2

# Set positions from the PDB file
simulation.context.setPositions(pdb.positions)

# Compute potential energy
state = simulation.context.getState(getEnergy=True)
potential_energy = state.getPotentialEnergy()

# Print the potential energy
print(f"Potential Energy: {potential_energy}") #C: kJ/mol here
