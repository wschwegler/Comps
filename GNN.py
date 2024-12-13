import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from graphite.nn import periodic_radius_graph
from tqdm.notebook import tqdm
import torch
from ase.neighborlist import primitive_neighbor_list
import graphite
from torch import nn
from graphite.nn.basis import Bessel
from graphite.nn.models.e3nn_nequip import NequIP
from functools import partial
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from data_set_gen_v2 import SMILESGraphDataset
from torch.utils.data import Subset


class PeriodicStructureDataset(Dataset):
    def __init__(self, atoms_list, y_list, large_cutoff, dup=1):
        super().__init__()

        unique_numbers = np.concatenate([np.unique(atoms.numbers) for atoms in atoms_list])
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.onehot_encoder.fit(unique_numbers.reshape(-1, 1))

        self.dataset = []
        for atoms, y in tqdm(zip(atoms_list, y_list), desc='Convert to PyG data'):
            z = self.onehot_encoder.transform(atoms.numbers.reshape(-1, 1))
            data = Data(
                x=torch.tensor(atoms.numbers).long(),
                y=y,
                pos=torch.tensor(atoms.positions).float(),
                cell=torch.tensor(np.array(atoms.cell)).float(),
                numbers=torch.tensor(atoms.numbers).long(),
                box=torch.diagonal(torch.tensor(np.array(atoms.cell))).float(),
                z=torch.tensor(atoms.numbers).long()
            )
            data = self.atomic_graph_ase(data, cutoff=large_cutoff)
            for _ in range(dup):
                self.dataset.append(data.clone())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def atomic_graph_ase(self, data, cutoff):
        i, j, D = primitive_neighbor_list('ijD', pbc=[True] * 3, cell=data.cell.numpy(), positions=data.pos.numpy(),
                                          cutoff=cutoff)
        data.edge_index = torch.tensor(np.stack((i, j))).long()
        data.edge_attr = torch.tensor(D).float()
        return data


#Y_test = torch.tensor(Y_test)
#
#length = len(Y_test)
#train_dataset = torch.load('/Users/wilsonschwegler/PycharmProjects/pythonProject7/train_dataset_1_V2.pt')
#test_dataset = torch.load('/Users/wilsonschwegler/PycharmProjects/pythonProject7/test_dataset_1_V2.pt')
#val_dataset = torch.load('/Users/wilsonschwegler/PycharmProjects/pythonProject7/val_dataset_1_V2.pt')

dataset = torch.load('smiles_graph_dataset_target_1_FINAL.pt')


dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


#minority_indices = []
#majority_indices = []
#for i in range(0, len(train_dataset)):
#    if(train_dataset[i].y.item() == 0.0):
#       majority_indices.append(i)
#    else:
#        minority_indices.append(i)
#
#
#weight_0 = dataset_size/(2*len(majority_indices))
#weight_1 = dataset_size/(2*len(minority_indices))
#weights = torch.tensor([weight_0, weight_1])
#print(weights)
#print(1)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


if __name__ == "__main__":

    PIN_MEMORY = False


    class PeriodicStructureDataset(Dataset):
        def __init__(self, atoms_list, y_list, large_cutoff, dup=1):
            super().__init__()

            unique_numbers = np.concatenate([np.unique(atoms.numbers) for atoms in atoms_list])
            self.onehot_encoder = OneHotEncoder(sparse_output=False)
            self.onehot_encoder.fit(unique_numbers.reshape(-1, 1))

            self.dataset = []
            for atoms, y in tqdm(zip(atoms_list, y_list), desc='Convert to PyG data'):
                z = self.onehot_encoder.transform(atoms.numbers.reshape(-1, 1))
                data = Data(
                    x=torch.zeros(len(atoms.numbers)).long(),
                    y=y,
                    z=torch.tensor(z).float(),
                    pos=torch.tensor(atoms.positions).float(),
                    cell=torch.tensor(np.array(atoms.cell)).float(),
                    numbers=torch.tensor(atoms.numbers).long(),
                    box=torch.diagonal(torch.tensor(np.array(atoms.cell))).float()
                )
                data = self.atomic_graph_ase(data, cutoff=large_cutoff)
                for _ in range(dup):
                    self.dataset.append(data.clone())

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.dataset[idx]

        def atomic_graph_ase(self, data, cutoff):
            i, j, D = primitive_neighbor_list('ijD', pbc=[True] * 3, cell=data.cell.numpy(), positions=data.pos.numpy(),
                                              cutoff=cutoff)
            data.edge_index = torch.tensor(np.stack((i, j))).long()
            data.edge_attr = torch.tensor(D).float()
            return data


    data_list=[]
    for i in range (0, len(train_dataset)):
        data_list.append(train_dataset[i].y.item())
    

    weights = compute_class_weight(class_weight="balanced", classes=np.unique(data_list), y=data_list)

    print(weights)

    weights = torch.tensor(weights, dtype=torch.float32)


    class MyModel(pl.LightningModule):
        def __init__(self, num_species, val_loader):
            super().__init__()
            self.num_species = num_species
            self.init_embed = InitialEmbedding(num_species)
            self.val_loader = val_loader
            self.model = NequIP(
                init_embed=self.init_embed,
                irreps_node_x='8x0e',
                irreps_node_z='8x0e',
                irreps_hidden='8x0e + 8x1e + 8x2e',
                irreps_edge='1x0e+1x1e+1x2e',
                irreps_out='1x0e',  # predict var and mu
                num_convs=1,
                radial_neurons=[16, 64],
                num_neighbors=6,
            )

        def forward(self, data):
            data = data.to(self.device)
            return self.model(data)

        def training_step(self, batch, batch_idx):
            data = batch
            data = data.to(self.device, non_blocking=PIN_MEMORY)

            disp_true = data.y
            mu = self.forward(data)
            mu_sum = mu.sum(dim=0)
            # loss = F.mse_loss(mu.float(), disp_true.float())
            #loss = F.l1_loss(mu_sum.float().squeeze(0), data.y[0].float())
            weights_per_sample = weights[data.y[0].int()]
            loss = F.binary_cross_entropy(torch.sigmoid(mu_sum.float().squeeze(0)), data.y[0].float(), weight=weights_per_sample)
            loss_list.append(loss)
            self.log('train_loss', loss, batch_size=data.size(0))  # Log the loss for visualization
            return loss

        def validation_step(self, batch, batch_idx):
            data = batch
            data = data.to(self.device)
            # Perform the forward pass
            mu = self.forward(data)
            mu_sum = mu.sum(dim=0)
            # Calculate the loss
            # loss = F.mse_loss(mu.float(), data.y.float())
            #loss = F.l1_loss(mu_sum.float().squeeze(0), data.y[0].float())
            weights_per_sample = weights[data.y[0].int()]
            loss = F.binary_cross_entropy(torch.sigmoid(mu_sum.float().squeeze(0)), data.y[0].float(), weight=weights_per_sample)
            loss_list.append(loss)
            # Log the loss
            self.log('val_loss', loss)
            return loss

        def test_step(self, batch, batch_idx):
            data = batch
            data = data.to(self.device)
            mu = self.forward(data)
            mu_sum = mu.sum(dim=0)
            #loss = F.mse_loss(mu.float(), data.y.float())
            #loss = F.l1_loss(mu_sum.float().squeeze(0), data.y[0].float())
            weights_per_sample = weights[data.y[0].int()]
            loss = F.binary_cross_entropy(torch.sigmoid(mu_sum.float().squeeze(0)), data.y[0].float(), weight=weights_per_sample)
            self.log('test_loss', loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.RAdam(self.parameters(), lr=2e-4)


        # Define the InitialEmbedding class
    class InitialEmbedding(nn.Module):
        def __init__(self, num_species):
            super().__init__()
            self.embed_node_x = nn.Embedding(num_species, 8)
            self.embed_node_z = nn.Embedding(num_species, 8)
            self.embed_edge = partial(bessel, start=0.0, end=5, num_basis=16)

        def forward(self, data):
            # Embed node
            h_node_x = self.embed_node_x(data.x)
            h_node_z = self.embed_node_z(data.x)

            # Embed edge
            h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))

            data.h_node_x = h_node_x
            data.h_node_z = h_node_z
            data.h_edge = h_edge
            return data



    loss_list = []
    #loss_list2 = []
    ## Calculate split sizes
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    ## Create an instance of your LightningModule
    forward_model = MyModel(num_species=54, val_loader=val_loader)
    forward_trainer = pl.Trainer(strategy='ddp',accelerator='cpu', devices=1, max_epochs=340)
    # if needed, resume_training='forward_model.ckpt'

    #checkpoint = torch.load('pfas-forward_model_target_0.ckpt')
    #print(checkpoint['epoch'])
    # Train the model
    #forward_trainer.fit(forward_model, train_loader, val_loader)
    forward_trainer.fit(forward_model, train_loader, val_loader,ckpt_path='pfas-forward_model_target_1.ckpt')
    print(forward_model)
    forward_model.eval()  # Switch to evaluation mode , ckpt_path='forward_model.ckpt'

    val_loss = forward_trainer.test(forward_model, val_loader)
    print(val_loss)
    print(f"Validation Loss: {val_loss}")

    # Save the model checkpoint
    checkpoint_path = 'pfas-forward_model_target_1.ckpt'
    forward_trainer.save_checkpoint(checkpoint_path)



'''
    sample = test_dataset[0]

    print("Sample:", sample)

    if(torch.sum(forward_model.forward(sample) < 0, dim=0)):
        print(0)

    print(len(test_dataset))

    tested_data = []
    actual_data = []
    for i in range(0, len(test_dataset)):
        #actual_data.append(int(test_dataset[i].y))
        #print(int(test_dataset[i].y))
        print(torch.sum(forward_model.forward(test_dataset[i]), dim=0))
        if(int(test_dataset[i].y) == 50):
            actual_data.append(1)
        else:
            actual_data.append(0)
        if (torch.sum(forward_model.forward(test_dataset[i]), dim=0) < 25):
            tested_data.append(0)
        else:
            tested_data.append(1)
            print(1)
'''
'''
    for l in range (0, len(actual_data)):
        print(actual_data[i])
    print("tested")
    for q in range(0, len(test_dataset)):
        print(tested_data[q])
'''

    #from sklearn.metrics import f1_score

    #f1 = f1_score(actual_data, tested_data)
    #print("F1 Score", f1)
    #from sklearn.metrics import accuracy_score

    # Assuming actual_data and tested_data are your arrays/lists
    #accuracy = accuracy_score(actual_data, tested_data)
    #print("Accuracy:", accuracy)

    #print(classification_report(actual_data, tested_data))

    # Save the model checkpoint
    #checkpoint_path = 'pfas-forward_model_1.ckpt'
    #forward_trainer.save_checkpoint(checkpoint_path)

    # forward_model = MyModel.load_from_checkpoint('/Users/schwegler2/pfas-forward_model.ckpt', num_species=10, val_loader=val_loader)
