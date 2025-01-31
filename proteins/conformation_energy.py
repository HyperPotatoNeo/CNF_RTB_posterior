
import sys 
sys.path.append('..')

import torch 


from prot_utils import * 

from openmm.app import PDBFile, Simulation, ForceField, NoCutoff, Modeller
from openmm import LangevinIntegrator, Platform
from openmm.unit import kelvin, picosecond, femtosecond, nanometer
from openmm import NonbondedForce


import pymol 
from pymol import cmd 
from PIL import Image 

class ConfEnergy():
    def __init__(self, device):
        self.device = device 
        self.protein_type = True 


    # get samples and save as WandB images
    def get_prot_image(self, samples):
        B = samples["prot_traj"].shape[1]

        imgs = []

        for i in range(B):
            sample = samples["prot_traj"][0, i, ...]
            
            pdb_str = prot_to_pdb(sample)
            fname = write_pdb(pdb_str, save_path = "./temp.pdb")
            
            print(fname)
            cmd.load(fname) #"./protein_2.pdb")

            cmd.show("cartoon")               # Show cartoon representation

            # highlight secondary structures
            # First calculate secondary structure if not already assigned
            cmd.dss()  # Determines secondary structure

            # Color by secondary structure
            cmd.color("blue", "ss h")        # Color helices blue
            cmd.color("red", "ss s")     # Color sheets red
            cmd.color("green", "ss l+''")   # Color loops green

            cmd.show("sticks", "organic")    # Show ligands as sticks
            cmd.zoom()                       # Zoom to molecule

            cmd.png("temp.png", width=800, height=800)
            
            # load .png  
            img = Image.open("temp.png")
            
            imgs.append(img)

            # Cleanup
            os.remove("temp.png")
            cmd.delete('all')

        return imgs 

    def eval_energy_from_pdb(self, pdb_fname):
        pdb = PDBFile(pdb_fname)

        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens()

        # Create system directly with ForceField
        forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
        system = forcefield.createSystem(
            modeller.topology,
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
        simulation = Simulation(modeller.topology, system, integrator, platform)

        # Set positions
        simulation.context.setPositions(modeller.positions)

        # Compute potential energy
        state = simulation.context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()

        return potential_energy

    def __call__(self, sample):
        pdb_strs = batch_to_pdb(sample)
        
        B = len(pdb_strs)
        energies = torch.zeros((B,)).to(self.device)

        for i in range(B):
            pdb_str = pdb_strs[i]

            fname = write_pdb(pdb_str, save_path = "./temp.pdb")
            en = self.eval_energy_from_pdb(fname)
            energies[i] = en 

        log_r = -energies 

        return log_r 