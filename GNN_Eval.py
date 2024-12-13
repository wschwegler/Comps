import numpy as np
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from graphite.nn import periodic_radius_graph
from tqdm.notebook import tqdm
import torch
from ase.neighborlist import primitive_neighbor_list
import graphite
from torch import nn
from graphite.nn.basis import bessel
from graphite.nn.models.e3nn_nequip import NequIP
from functools import partial
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch.utils.data import Subset
from data_set_gen_v2 import SMILESGraphDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt





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

target = 19

#Y_test = torch.tensor(Y_test)
#
#length = len(Y_test)
dataset = torch.load('/Users/wilsonschwegler/PycharmProjects/comps/smiles_graph_dataset_target_'+str(target)+'_FINAL.pt')

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(len(train_dataset))

minority_indices = []
majority_indices = []
for i in range(0, len(train_dataset)):
    if(train_dataset[i].y.item() == 0.0):
       majority_indices.append(i)
    else:
        minority_indices.append(i)

minority_sampled_indices = np.random.choice(minority_indices, size=int(len(majority_indices)*0.5), replace=True)

balanced_indices = np.concatenate([minority_sampled_indices, majority_indices])

np.random.shuffle(balanced_indices)

weight_0 = dataset_size/(2*len(majority_indices))
print(weight_0)
weight_1 = dataset_size/(2*len(minority_indices))
print(weight_1)
weights = torch.tensor([weight_0, weight_1])

balanced_dataset = Subset(train_dataset, balanced_indices)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))





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


    #data_list=[]
    #for i in range (0, len(train_dataset)):
    #    data_list.append(train_dataset[i].y.item())
    #print(data_list)

    #weights = compute_class_weight(class_weight="balanced", classes=np.unique(data_list), y=data_list)

    #print(weights)

    #weights = torch.tensor(weights, dtype=torch.float32)


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
            # loss = F.mse_loss(mu.float(), data.y.float())
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
    loss_list2 = []
    ## Calculate split sizes
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)


    #forward_model = MyModel.load_from_checkpoint('/Users/wilsonschwegler/Desktop/pastcheckpoints/7/pfas-forward_model_target_'+str(target)+'.ckpt', num_species=54, val_loader=val_loader)
    forward_model = MyModel.load_from_checkpoint(
        '/Users/wilsonschwegler/test_pfas-forward_model_1.ckpt',
        num_species=54, val_loader=val_loader)
    sample = test_dataset[0]

    print("Sample:", sample)

    if (torch.sum(forward_model.forward(sample) < 0, dim=0)):
        print(0)

    print(len(test_dataset))

    tested_data = []
    actual_data = []
    for i in range(0, len(test_dataset)):
        # actual_data.append(int(test_dataset[i].y))
        print(int(test_dataset[i].y))
        print(torch.sum(forward_model.forward(test_dataset[i]), dim=0))
        # print(int(test_dataset[i].y))
        if (int(test_dataset[i].y) == 1):
            actual_data.append(1)
        else:
            actual_data.append(0)
        if (torch.sum(forward_model.forward(test_dataset[i]), dim=0) < 0.5):
            tested_data.append(0)
        else:
            tested_data.append(1)
            print(torch.sum(forward_model.forward(test_dataset[i])))
            print(1)

    '''
    for l in range (0, len(actual_data)):
        print(actual_data[i])
    print("tested")
    for q in range(0, len(test_dataset)):
        print(tested_data[q])
    '''

    from sklearn.metrics import f1_score, classification_report

    f1 = f1_score(actual_data, tested_data)
    print("F1 Score", f1)
    from sklearn.metrics import accuracy_score

    # Assuming actual_data and tested_data are your arrays/lists
    accuracy = accuracy_score(actual_data, tested_data)
    print("Accuracy:", accuracy)

    print(classification_report(actual_data, tested_data))
    i = 10
    cm = confusion_matrix(actual_data, tested_data, labels=[0.0, 1.0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[0.0, 1.0])
    disp.plot()
    plt.show()










