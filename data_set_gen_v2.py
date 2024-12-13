import torch
from torch_geometric.data import Data, Dataset
from ase.neighborlist import neighbor_list
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from ase import Atoms

for i in range(19, 20):
    class SMILESGraphDataset(Dataset):
        def __init__(self, csv_file, large_cutoff, dup=128):
            super().__init__(None, transform=None, pre_transform=None)

            # Load the CSV file
            self.dataframe = pd.read_csv(csv_file)

            # Extract SMILES strings and bioactivity data from the dataframe
            self.smiles_list = self.dataframe['smiles'].tolist()

            # Get the bioactivity columns (all columns except 'smiles')
            self.bioactivity_columns = [col for col in self.dataframe.columns if col != 'smiles']

            # Initialize dataset list
            self.dataset = []

            for index, row in self.dataframe.iterrows():
                y_value = row[self.bioactivity_columns].iloc[i]

                if (y_value == 0.0 or y_value == 1.0):
                    print(y_value)
                    smiles = row['smiles']

                    # Convert SMILES to RDKit molecule
                    mol = Chem.MolFromSmiles(smiles)
                    mol = Chem.AddHs(mol)
                    print(i, f"Processing SMILES: {smiles}")

                    try:
                        AllChem.EmbedMolecule(mol)
                        AllChem.UFFOptimizeMolecule(mol)
                        atom_positions = mol.GetConformer().GetPositions()

                        # Extract bioactivity data for this molecule
                        #y_value = row[self.bioactivity_columns].fillna(0).values  # Replace NaNs with zeros if needed

                        # Create a PyTorch Geometric Data object
                        data = Data(
                            x=torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]).long(),
                            pos=torch.tensor(atom_positions).float(),
                            numbers=torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]).long(),
                            y=torch.tensor(y_value).float().unsqueeze(0),  # *25 Unsqueezing to add a batch dimension
                        )

                        # Create atomic graph
                        data = self.atomic_graph_ase(data, cutoff=large_cutoff)

                        # Duplicate the data if needed
                        for _ in range(dup):
                            self.dataset.append(data.clone())

                    except ValueError as e:
                        print(f"Skipping molecule with SMILES {smiles}: {str(e)}")
                        continue

        def len(self):
            return len(self.dataset)

        def get(self, idx):
            return self.dataset[idx]

        def atomic_graph_ase(self, data, cutoff):
            # Create atomic graph using ASE primitive neighbor list
            atom_numbers = data.x.numpy()
            positions = data.pos.numpy()
            ase_atoms = Atoms(numbers=atom_numbers, positions=positions)
            i, j, D = neighbor_list('ijD', a=ase_atoms, cutoff=cutoff)
            data.edge_index = torch.tensor(np.stack((i, j))).long()
            data.edge_attr = torch.tensor(D).float()
            return data

    # Path to the CSV file containing SMILES strings and bioactivity data
    #csv_file = '/Users/wilsonschwegler/Desktop/Internship/es9b04833_si_003.csv'
    #
    ## Specify other parameters
    #large_cutoff = 100
    #dup = 1
    #
    ## Create the dataset
    #
    ## Create the dataset/
    #dataset = SMILESGraphDataset(csv_file, large_cutoff, dup)
    #
    ## Save the dataset
    #torch.save(dataset, 'smiles_graph_dataset_target_'+str(i)+'_FINAL.pt')
    #3
