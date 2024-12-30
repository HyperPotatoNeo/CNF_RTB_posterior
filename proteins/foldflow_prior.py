import sys 
sys.path.append('..')

import os
import subprocess 
import numpy as np
import pandas as pd 
import torch
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ['EOMSTATS_BACKEND'] = 'pytorch'

from omegaconf import DictConfig, OmegaConf
import yaml 

import tree
from typing import Dict, Optional
import esm 
import GPUtil

from biotite.sequence.io import fasta

from foldflow.data import utils as du
from tools.analysis import utils as au
from tools.analysis import metrics 

from foldflow.models.ff2flow.flow_model import FF2Model
from foldflow.models.ff2flow.ff2_dependencies import FF2Dependencies

from runner import train 

def get_conf(dir):
    with open(dir, 'r') as file:
        yaml_content = yaml.safe_load(file)

    # Convert YAML content to DictConfig
    conf = OmegaConf.create(yaml_content)

    return conf 

def get_device(_infer_conf):
    if torch.cuda.is_available():
        if _infer_conf.gpu_id is None:
            available_gpus = "".join(
                [str(x) for x in GPUtil.getAvailable(order="memory", limit=8)]
            )
            device = f"cuda:{available_gpus[0]}"
        else:
            device = f"cuda:{_infer_conf.gpu_id}"
    else:
        device = "cpu"
    return device 

class FoldFlowModel():
    def __init__(self, device, num_steps = 50):
        torch.set_float32_matmul_precision("medium")
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = True
        
        #os.chdir('../')
        self.device = device 

        conf = get_conf("./proteins/inference_conf.yaml")

        # set to default val 
        conf.inference.flow.num_t = num_steps

        # Prepare configs.
        self._conf = conf
        self._infer_conf = conf.inference
        self._fm_conf = self._infer_conf.flow
        self._sample_conf = self._infer_conf.samples

        self._rng = np.random.default_rng(self._infer_conf.seed)

        # Set model hub directory for ESMFold.
        torch.hub.set_dir(self._infer_conf.pt_hub_dir)
        
        _weights_path = self._infer_conf.weights_path
        
        # Load models and experiment
        weights_pkl = du.read_pkl(
                _weights_path, use_torch=True, map_location=device
        )

        deps = FF2Dependencies(conf)
        self.model = FF2Model.from_ckpt(weights_pkl, deps)
        self.flow_matcher = deps.flow_matcher
        self.exp = train.Experiment(conf=self._conf, model=self.model)  

        self.model = self.model.to(device)
        self.model.eval()
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(device)

    #################################
    # sampling and inference
    #################################   
    def sample(self, sample_length: int, context: Optional[torch.Tensor] = None):
            """Sample based on length.

            Args:
                sample_length: length to sample

            Returns:
                Sample outputs. See train.inference_fn.
            """
            # Process motif features.
            res_mask = np.ones(sample_length)
            fixed_mask = np.zeros_like(res_mask)
            aatype = torch.zeros(sample_length, dtype=torch.int32)
            chain_idx = torch.zeros_like(aatype)

            # Initialize data
            ref_sample = self.flow_matcher.sample_ref(
                n_samples=sample_length,
                as_tensor_7=True,
            )
            res_idx = torch.arange(1, sample_length + 1)
            init_feats = {
                "res_mask": res_mask,
                "seq_idx": res_idx,
                "fixed_mask": fixed_mask,
                "torsion_angles_sin_cos": np.zeros((sample_length, 7, 2)),
                "sc_ca_t": np.zeros((sample_length, 3)),
                "aatype": aatype,
                "chain_idx": chain_idx,
                **ref_sample,
            }
            # Add batch dimension and move to GPU.
            init_feats = tree.map_structure(
                lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
            )
            init_feats = tree.map_structure(lambda x: x[None].to(self.device), init_feats)

            # Run inference
            sample_out = self.exp.inference_fn(
                init_feats,
                num_t=self._fm_conf.num_t,
                min_t=self._fm_conf.min_t,
                aux_traj=True,
                noise_scale=self._fm_conf.noise_scale,
                context=context,
            )
            return tree.map_structure(lambda x: x[:, 0], sample_out)

    #input tensor of init feats like in sample method
    def __call__(self, init_feats, context = None):
        # Run inference
        sample_out = self.exp.inference_fn(
            init_feats,
            num_t=self._fm_conf.num_t,
            min_t=self._fm_conf.min_t,
            aux_traj=True,
            noise_scale=self._fm_conf.noise_scale,
            context=context,
        )
        return tree.map_structure(lambda x: x[:, 0], sample_out)
    
    def sample_prior(self, sample_length: int, ref_sample = None):
        # Process motif features.
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)
        aatype = torch.zeros(sample_length, dtype=torch.int32)
        chain_idx = torch.zeros_like(aatype)

        if ref_sample is None:
            # Initialize data
            ref_sample = self.flow_matcher.sample_ref(
                n_samples=sample_length,
                as_tensor_7=True,
            )
        else:
            assert len(ref_sample) == sample_length , "Length of ref_sample must be equal to sample_length"
            
        res_idx = torch.arange(1, sample_length + 1)
        init_feats = {
            "res_mask": res_mask,
            "seq_idx": res_idx,
            "fixed_mask": fixed_mask,
            "torsion_angles_sin_cos": np.zeros((sample_length, 7, 2)),
            "sc_ca_t": np.zeros((sample_length, 3)),
            "aatype": aatype,
            "chain_idx": chain_idx,
            **ref_sample,
        }
        
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(lambda x: x[None].to(self.device), init_feats)

        return init_feats

    def so3_unif_prior(self, sample_length):
        return self.flow_matcher.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
    
    #########################################
    # Evaluation and Saving
    #########################################
    def save_traj(
        self,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        flow_mask: np.ndarray,
        output_dir: str,
    ):
        """Writes final sample and reverse flow matching trajectory.

        Args:
            bb_prot_traj: [T, N, 37, 3] atom37 sampled flow matching states.
                T is number of time steps. First time step is t=eps,
                i.e. bb_prot_traj[0] is the final sample after reverse flow matching.
                N is number of residues.
            x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
            aatype: [T, N, 21] amino acid probability vector trajectory.
            res_mask: [N] residue mask.
            flow_mask: [N] which residues are flowed.
            output_dir: where to save samples.

        Returns:
            Dictionary with paths to saved samples.
                'sample_path': PDB file of final state of reverse trajectory.
                'traj_path': PDB file os all intermediate flowed states.
                'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
            b_factors are set to 100 for flowed residues and 0 for motif
            residues if there are any.
        """

        # Write sample.
        flow_mask = flow_mask.astype(bool)
        sample_path = os.path.join(output_dir, "sample")
        prot_traj_path = os.path.join(output_dir, "bb_traj")
        x0_traj_path = os.path.join(output_dir, "x0_traj")

        # Use b-factors to specify which residues are flowed.
        b_factors = np.tile((flow_mask * 100)[:, None], (1, 37))

        sample_path = au.write_prot_to_pdb(
            bb_prot_traj[0], sample_path, b_factors=b_factors
        )
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj, prot_traj_path, b_factors=b_factors
        )
        x0_traj_path = au.write_prot_to_pdb(x0_traj, x0_traj_path, b_factors=b_factors)
        return {
            "sample_path": sample_path,
            "traj_path": prot_traj_path,
            "x0_traj_path": x0_traj_path,
        }

    def run_self_consistency(
        self,
        decoy_pdb_dir: str,
        reference_pdb_path: str,
        motif_mask: Optional[np.ndarray] = None,
    ):
        """Run self-consistency on design proteins against reference protein.

        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """

        # Run PorteinMPNN
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen(
            [
                "python",
                f"{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py",
                f"--input_path={decoy_pdb_dir}",
                f"--output_path={output_path}",
            ]
        )
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            "python",
            f"{self._pmpnn_dir}/protein_mpnn_run.py",
            "--out_folder",
            decoy_pdb_dir,
            "--jsonl_path",
            output_path,
            "--num_seq_per_target",
            str(self._sample_conf.seq_per_sample),
            "--sampling_temp",
            "0.1",
            "--seed",
            str(self._infer_conf.seed),
            "--batch_size",
            "1",
        ]
        if self._infer_conf.gpu_id is not None:
            pmpnn_args.append("--device")
            pmpnn_args.append(str(self._infer_conf.gpu_id))
        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._log.info(f"Failed ProteinMPNN. Attempt {num_tries}/5 {e}")
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            "seqs",
            os.path.basename(reference_pdb_path).replace(".pdb", ".fa"),
        )

        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results = {
            "tm_score": [],
            "sample_path": [],
            "header": [],
            "sequence": [],
            "rmsd": [],
        }
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results["motif_rmsd"] = []
        esmf_dir = os.path.join(decoy_pdb_dir, "esmf")
        os.makedirs(esmf_dir, exist_ok=True)

        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats("sample", reference_pdb_path)
        for i, (header, string) in enumerate(fasta_seqs.items()):
            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f"sample_{i}.pdb")
            _ = self.run_folding(string, esmf_sample_path)
            esmf_feats = du.parse_pdb_feats("folded_sample", esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats["aatype"])

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = metrics.calc_tm_score(
                sample_feats["bb_positions"],
                esmf_feats["bb_positions"],
                sample_seq,
                sample_seq,
            )
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats["bb_positions"], esmf_feats["bb_positions"]
            )
            if motif_mask is not None:
                sample_motif = sample_feats["bb_positions"][motif_mask]
                of_motif = esmf_feats["bb_positions"][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(sample_motif, of_motif)
                mpnn_results["motif_rmsd"].append(motif_rmsd)
            mpnn_results["rmsd"].append(rmsd)
            mpnn_results["tm_score"].append(tm_score)
            mpnn_results["sample_path"].append(esmf_sample_path)
            mpnn_results["header"].append(header)
            mpnn_results["sequence"].append(string)

        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, "sc_results.csv")
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output


def main():
    flow_prior = FoldFlowModel(device = "cuda:0", num_steps = 50)

    print("Loaded!")

    sample_out = flow_prior.sample(sample_length = 1)

    print("Sampled!")
    print("sample_out: ", sample_out)

if __name__ == "__main__":
    main()