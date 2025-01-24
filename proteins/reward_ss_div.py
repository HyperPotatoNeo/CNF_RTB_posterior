import sys 
sys.path.append('..')

from io import StringIO  
import tempfile 

import numpy as np 
import torch 

import mdtraj as md 
from rdkit import Chem

from tools.analysis import utils as au
from tools.analysis import metrics 

import torchdrug as td
from torchdrug import models, core, tasks, data, transforms , datasets, layers
from torchdrug.layers import geometry
from prot_utils import * 
import model 
from torch.optim import lr_scheduler 

Registry.register("StepLR")(lr_scheduler.StepLR)

truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])


import pymol 
from pymol import cmd 
from PIL import Image 


class FoldClassifier2OLD():
    def __init__(self, device):
        self.device = device 
        args, vars = parse_args()
        self.cfg = load_config(args.config, vars)

        #self.cfg = get_conf("./proteins/gearnet_edge_ieconv.yaml")
        
        #self.cfg = load_config("./proteins/gearnet_edge_ieconv.yaml")
        dataset = core.Configurable.load_config_dict(self.cfg.dataset)
        self.solver, self.scheduler = build_downstream_solver(self.cfg, dataset)

        self.solver.model = self.solver.model.to(self.device)
        print("self.solver: ", self.solver)
        print("self.solver.model (syn with task): ", self.solver.model)

        #self.task = core.Configurable.load_config_dict(self.cfg.task)


        #print("self.task: ", self.task.graph_construction_model)
        #self.task.load()
        #print("self.task: ", self.task)

        self.transforms = transform 
        #self.solver = core.Engine(self.task, train_set = None, valid_set =None, test_set = None, optimizer=None)
    
    def to_pdb(self, sample_out):
        return prot_to_pdb(sample_out["prot_traj"][0])

    def to_protein_ds(self, sample_pdb):
        mol = Chem.MolFromPDBBlock(sample_pdb)
        return data.Protein.from_molecule(mol, atom_feature="position", bond_feature="length", residue_feature="symbol")

    def __call__(self, sample):
        #sample = sample.to(self.device)
        print('sample device: ', sample['graph'].device)
        print("model device: ", self.solver.model.device)
        with torch.no_grad():
            #sample = self.transforms(sample)
            #batch = {"graph": sample}
            #batch = self.transforms(batch)
            #sample = self.to_protein_ds(sample)
            pred = self.solver.model.predict(sample) #_and_target(sample)
            
        return pred

    
def to_pdb(sample_out):
    traj_shape = sample_out["prot_traj"].shape

    print("traj_shape: ", traj_shape)

    # there is a batch size: (shape [T,B,N, 37, 3])
    if len(traj_shape) > 4:
        pdb_str = [] # list of pdb strs 

        for i in range(traj_shape[1]):
            sample = sample_out["prot_traj"][0, i, ...]
            pdb_str_i = prot_to_pdb(sample)
            pdb_str.append(pdb_str_i)
    else:
        pdb_str = prot_to_pdb(sample_out["prot_traj"][0])
        pdb_str = [pdb_str]
    return pdb_str

def to_protein_ds_batch(pdb_str):

    if isinstance(pdb_str, str):
        mol = to_protein_ds(pdb_str)
        mols = [mol]

    elif isinstance(pdb_str, list):
        #print("to protrein ds batch detected a list!")
        mols = []
        for sample_pdb in pdb_str:
            mol = to_protein_ds(sample_pdb)
            mols.append(mol)

    return mols     
        
def to_protein_ds(sample_pdb):
    #mol = Chem.MolFromPDBBlock(sample_pdb)
    #return data.Protein.from_molecule(mol, atom_feature="position", bond_feature="length", residue_feature="symbol")

    try:
        mol = Chem.MolFromPDBBlock(sample_pdb, sanitize=False)
        if mol is None:
            raise ValueError("Invalid PDB block")
    
        # Sanitize the molecule to fix valence issues
        Chem.SanitizeMol(mol)
    
        return data.Protein.from_molecule(mol, atom_feature="position", bond_feature="length", residue_feature="symbol")
    except Chem.rdchem.KekulizeException as e:
        print(f"Error processing molecule: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None


class FoldClassifier():
    def __init__(self, device, target_class = 0):
        self.device = device 
        
        self.protein_type = True 
        
        self.target_class = target_class

        self.graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                    edge_layers=[geometry.SequentialEdge(max_distance=2), 
                                                                 geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5)],
                                                    edge_feature="gearnet")
        
        self.model = model.GearNetIEConv(input_dim = 21,
                                              embedding_dim = 512,
                                              hidden_dims = [512, 512, 512, 512, 512, 512],
                                              batch_norm = True,
                                              concat_hidden = True,
                                              short_cut = True,
                                              readout = 'sum',
                                              num_relation = 7,
                                              edge_input_dim = 59,
                                              num_angle_bin = 8,
                                              layer_norm = True,
                                              dropout = 0.2,
                                              use_ieconv = True)
        
        
        self.task = tasks.PropertyPrediction(model = self.model, 
                                             graph_construction_model= self.graph_construction_model,
                                             num_mlp_layer=3,
                                             mlp_batch_norm=True,
                                             mlp_dropout=0.5,
                                             criterion = 'ce',
                                             metric = ('acc'),
                                             num_class = 1195)
        self.ckpt_file = "./proteins/gearnet_ckpt/fold_mc_gearnet_edge_ieconv.pth" 

        state_dict = torch.load(self.ckpt_file, map_location=torch.device(self.device))

        self.init_task_mlp()

        self.task.load_state_dict(state_dict['model'])
        self.task = self.task.to(self.device)
        self.task.model.eval() 
        self.task.mlp.eval()
        print("\n\n Loaded model from: ", self.ckpt_file)

        self.transforms = transforms.ProteinView(view='residue')

    def init_task_mlp(self):
        mean = [torch.zeros(1,)]
        std = [torch.ones(1,)]
        weight = [torch.ones(1,)]

        self.task.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.task.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.task.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))

        hidden_dims = [self.task.model.output_dim] * (self.task.num_mlp_layer - 1)
        self.task.mlp = layers.MLP(self.task.model.output_dim, hidden_dims + [sum(self.task.num_class)],
                            batch_norm=self.task.mlp_batch_norm, dropout=self.task.mlp_dropout)


        return 
    
    def to_pdb(self, sample_out):
        traj_shape = sample_out["prot_traj"].shape

        print("traj_shape: ", traj_shape)

        # there is a batch size: (shape [T,B,N, 37, 3])
        if len(traj_shape) > 4:
            pdb_str = [] # list of pdb strs 

            for i in range(traj_shape[1]):
                sample = sample_out["prot_traj"][0, i, ...]
                pdb_str_i = prot_to_pdb(sample)
                pdb_str.append(pdb_str_i)
        else:
            pdb_str = prot_to_pdb(sample_out["prot_traj"][0])

        return pdb_str
    
    def to_protein_ds_batch(self, pdb_str):

        if isinstance(pdb_str, str):
            mol = self.to_protein_ds(pdb_str)
            mols = [mol]

        elif isinstance(pdb_str, list):
            #print("to protrein ds batch detected a list!")
            mols = []
            for sample_pdb in pdb_str:
                mol = self.to_protein_ds(sample_pdb)
                mols.append(mol)

        return mols     
            
    def to_protein_ds(self, sample_pdb):
        #mol = Chem.MolFromPDBBlock(sample_pdb)
        #return data.Protein.from_molecule(mol, atom_feature="position", bond_feature="length", residue_feature="symbol")

        try:
            mol = Chem.MolFromPDBBlock(sample_pdb, sanitize=False)
            if mol is None:
                raise ValueError("Invalid PDB block")
        
            # Sanitize the molecule to fix valence issues
            Chem.SanitizeMol(mol)
        
            return data.Protein.from_molecule(mol, atom_feature="position", bond_feature="length", residue_feature="symbol")
        except Chem.rdchem.KekulizeException as e:
            print(f"Error processing molecule: {e}")
            return None
        except ValueError as e:
            print(f"Error: {e}")
            return None


    # sample input must be directly from foldflow2 model output 
    def __call__(self, sample):
        #print("Sample: ", sample)

        sample = self.to_pdb(sample)
        sample = self.to_protein_ds_batch(sample)

        none_mask = np.array([x is None for x in sample])       
        filtered = [mol for mol in sample if mol is not None]

        print("len samples: ", len(sample))

        logr = torch.zeros(len(sample),).to(self.device)

        #reward for none molecules
        logr[none_mask] = -10.0 

        # no valid molecules
        if not filtered:
            return logr 

        #if sample is None:
        #    print("Invalid molecule!, reward is -10!")
        #    logr = -10.0*torch.ones((1,))
        #    #print("Shape (invalid) logr: ", logr.shape)

        #    return logr 

        #sample = sample.to(self.device)
        #print("sample (prepack): ", sample) 
        filt_sample = data.Protein.pack(filtered).to(self.device)
        filt_sample.view = 'residue'

        with torch.no_grad():
           
            
            #pred = self.task.predict(sample) #_and_target(sample)
            
            graph = self.task.graph_construction_model(filt_sample)
            #print("Intermediate graph: ", graph)

            output = self.task.model(graph, graph.node_feature.float())
            #print("output: ", output)
            #print("output graph feature shape: ", output['graph_feature'].shape)
            #print("self.task output_dim: ", self.task.model.output_dim)

            pred = self.task.mlp(output["graph_feature"])
            #print("pred: ", pred)
            #print("pred shape: ", pred.shape)

            pred = torch.log_softmax(pred, dim = -1)
            target_logit = pred[:, self.target_class]
        #print("Target logit: ", target_logit)
        #print("Shape logr: ", target_logit.shape)
        
        #print("logr shape: ", logr.shape)

        logr[~none_mask] = target_logit

        print("Log reward: ", logr)
        #print("logr shape: ", logr.shape)

        return logr
 
    # sample must be of form dict, with key 'graph' pointing to protein object 
    def forward_graph(self, sample):
        sample['graph'] = sample['graph'].to(self.device)

        with torch.no_grad():
            graph = self.task.graph_construction_model(sample['graph'])
            #print("Intermediate graph: ", graph)

            output = self.task.model(graph, graph.node_feature.float())
            #print("output: ", output)
            #print("output graph feature shape: ", output['graph_feature'].shape)
            #print("self.task output_dim: ", self.task.model.output_dim)

            pred = self.task.mlp(output["graph_feature"])
            #print("pred: ", pred)
            #print("pred shape: ", pred.shape)
        return pred
    
    # get samples and save as WandB images
    def get_prot_image(self, samples):
        #print("sample shape: ", samples["prot_traj"].shape)
        B = samples["prot_traj"].shape[1]

        imgs = []

        print("batch size: ", B)

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

# reward counting amount of helices in sampled protein
class SheetPercentReward():
    def __init__(self, device):
        self.device = device
        self.eps = 1e-5
        self.protein_type = True 

    def load_traj(self, pdb_path):
        traj = md.load(pdb_path)
        return traj
    
    def calc_mdtraj_metrics(self, traj):
        
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_coil_percent = np.mean(pdb_ss == "C")
        pdb_helix_percent = np.mean(pdb_ss == "H")
        pdb_strand_percent = np.mean(pdb_ss == "E")
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
        pdb_rg = md.compute_rg(traj)[0]
        return {
            "non_coil_percent": pdb_ss_percent,
            "coil_percent": pdb_coil_percent,
            "helix_percent": pdb_helix_percent,
            "strand_percent": pdb_strand_percent,
            "radius_of_gyration": pdb_rg,
        }

    
    # get samples and save as WandB images
    def get_prot_image(self, samples):
        B = samples["prot_traj"].shape[1]

        imgs = []

        print("batch size: ", B)

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

    def calc_ss_percentages(self, traj):
        """
        Calculate secondary structure percentages from PDB string
        
        Args:
            pdb_string: PDB format structure as string
        
        Returns:
            dict with helix, sheet, and coil percentages
        """
        
        # Calculate secondary structure
        # Returns 'H' for helix, 'E' for sheet, 'C' for coil
        ss = md.compute_dssp(traj, simplified=True)
        
        # Calculate percentages
        total_residues = ss.shape[1]
        helix_percent = np.mean(ss == 'H')
        sheet_percent = np.mean(ss == 'E')
        coil_percent = np.mean(ss == 'C')
        
        return {
            'helix': helix_percent,
            'sheet': sheet_percent,
            'coil': coil_percent,
            'structured': helix_percent + sheet_percent  # Total structured content
        }


    def get_percents(self, pdb_str_list, *args):
        batch_size = len(pdb_str_list)
        print("pdb batch size: ", batch_size)

        sheet_percents = torch.zeros((batch_size,)).to(self.device)

        # sample is sample_out 
        with torch.no_grad():
            for i, pdb_str in enumerate(pdb_str_list):
                with open(f"./protein_{i}.pdb", "w") as f:
                    f.write(pdb_str)
                pdb_filename = "./protein_{}.pdb".format(i)
                # Create temporary trajectory
                traj = md.load_pdb(pdb_filename)
            
                # Calculate percentages
                ss_metrics = self.calc_ss_percentages(traj)
            
                # Convert to tensor
                helix_percent = torch.tensor(ss_metrics["helix"]).to(self.device)
                sheet_percent = torch.tensor(ss_metrics["sheet"]).to(self.device)
                
                print("{} helix percent: {}".format(i, helix_percent))
                print("{} sheet percent: {}".format(i, sheet_percent))

                sheet_percents[i] = sheet_percent #helix_percent
                #sheet_percents.append(sheet_percent)

            return sheet_percents #, sheet_percents

      # sample input must be directly from foldflow2 model output 
    def __call__(self, sample):

        sample = to_pdb(sample)
        
        with torch.no_grad():
           percents = self.get_percents(sample)
           logr = torch.log(percents + self.eps)

        print("Log reward: ", logr)
        print("logr shape: ", logr.shape)

        return logr