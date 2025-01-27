import yaml 
import os
import numpy as np 
import torch 
import re

import jinja2 
from jinja2 import meta

import easydict
from foldflow.data import protein 
from omegaconf import DictConfig, OmegaConf

from torchdrug import models, core, tasks, data, transforms, utils
from torchdrug.utils import comm
import logging 

import argparse
from torch.optim import lr_scheduler 

from torchdrug.core import Registry

#Registry.register("StepLR")(lr_scheduler.StepLR)

logger =  logging.getLogger(__name__)

def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars

"""
def visualize_pdb(pdb_string, output_path="protein.png"):
    # Initialize PyMOL
    pymol.finish_launching()
    
    # Load structure from string
    cmd.read_pdbstr(pdb_string, "protein")
    
    # Style options
    cmd.show("cartoon")  # Ribbon representation
    cmd.color("spectrum")  # Color by chain
    cmd.set("ray_shadows", 0)  # Disable shadows for speed
    
    # Save image
    cmd.png(output_path, width=1000, height=1000, ray=1)
    cmd.delete("all")
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=False, default = "./proteins/gearnet_edge_ieconv.yaml")
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, default="null")
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars

def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg

def build_downstream_solver(cfg, dataset):
    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    if cfg.task['class'] == 'MultipleBinaryClassification':
        cfg.task.task = [_ for _ in range(len(dataset.tasks))]
    else:
        cfg.task.task = dataset.tasks
    task = core.Configurable.load_config_dict(cfg.task)

    cfg.optimizer.params = task.parameters()        
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    if "scheduler" not in cfg:
        scheduler = None
    
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        cfg.engine.scheduler = scheduler

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if "lr_ratio" in cfg:
        cfg.optimizer.params = [
            {'params': solver.model.model.parameters(), 'lr': cfg.optimizer.lr * cfg.lr_ratio},
            {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer
    elif "sequence_model_lr_ratio" in cfg:
        assert cfg.task.model["class"] == "FusionNetwork"
        cfg.optimizer.params = [
            {'params': solver.model.model.sequence_model.parameters(), 'lr': cfg.optimizer.lr * cfg.sequence_model_lr_ratio},
            {'params': solver.model.model.structure_model.parameters(), 'lr': cfg.optimizer.lr},
            {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer

    if scheduler is not None:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        solver.scheduler = scheduler

    if cfg.get("checkpoint") is not None:
        solver.load(cfg.checkpoint)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        
        print("model_dict keys: ", model_dict['model'].keys())
        
        #new_state_dict = {}
        #for k, v in model_dict['model'].items():
        #    new_key = k.replace('module.', '')  # Adjust this based on the actual prefix
        #    if 'model.' in k:
        #        new_key = k.replace('model.', '')
        #        new_state_dict[new_key] = v
            

        #task.model.load_state_dict(new_state_dict)
        #task.model 
        solver.model.load_state_dict(model_dict['model'])
    return solver, scheduler

def get_conf(dir):
    with open(dir, 'r') as file:
        yaml_content = yaml.safe_load(file)

    # Convert YAML content to DictConfig
    conf = OmegaConf.create(yaml_content)

    return conf 

def create_full_prot(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    aatype=None,
    b_factors=None,
):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def write_pdb(pdb_str, save_path = "./temp.pdb"):
    with open(save_path, "w") as f:
        f.write(pdb_str)
        f.write("END")
    
    return save_path 


def prot_to_pdb(
    prot_pos: np.ndarray,
    aatype: np.ndarray = None,
    overwrite=False,
    no_indexing=False,
    b_factors=None,
):
    if b_factors is None:
        flow_mask = np.ones(prot_pos.shape[0])
        
        b_factors = np.tile((flow_mask * 100)[:, None], (1, 37))
        
    if prot_pos.ndim == 4:
        len = prot_pos.shape[1]

        for t, pos37 in enumerate(prot_pos):
            atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
            prot = create_full_prot(
                pos37, atom37_mask, aatype=aatype, b_factors=b_factors
            )
            pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
            
    elif prot_pos.ndim == 3:
        atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
        prot = create_full_prot(
            prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors
        )
        pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
        
    else:
        raise ValueError("prot_pos must have 3 or 4 dimensions")

    return pdb_prot 


def batch_to_pdb(sample_out):
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

"""

import pymol 
from pymol import cmd 
from PIL import Image 

# get samples and save as WandB images
def get_prot_image(samples):
    B = samples["prot_traj"].shape[0]

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

    return imgs 
"""