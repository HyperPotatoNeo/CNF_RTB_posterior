from openmm.app import PDBFile, Simulation, ForceField, NoCutoff
from openmm import LangevinIntegrator, Platform
from openmm.unit import kelvin, picosecond, femtosecond, nanometer
from openmm import NonbondedForce

# Load the PDB file
pdb_filename = './2lwz.pdb'
pdb = PDBFile(pdb_filename)

# Create system directly with ForceField
forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=NoCutoff  # Disable periodic boundary conditions
)

# Modify the nonbonded force parameters directly
for force in system.getForces():
    if isinstance(force, NonbondedForce):
        force.setNonbondedMethod(NonbondedForce.NoCutoff)
        force.setCutoffDistance(1.0*nanometer)  # Set a smaller cutoff

# Define integrator
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2*femtosecond)

# Create simulation with explicit platform choice
platform = Platform.getPlatformByName('CPU')  # or 'CUDA' for GPU
simulation = Simulation(pdb.topology, system, integrator, platform)

# Set positions
simulation.context.setPositions(pdb.positions)

# Compute potential energy
state = simulation.context.getState(getEnergy=True)
potential_energy = state.getPotentialEnergy()

print(f"Potential Energy: {potential_energy}")