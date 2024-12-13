import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import math
import matplotlib.patches as patches

warnings.filterwarnings("ignore")
'''
#reads the csv file to a pd data frame (df)
df = pd.read_csv("/Users/wilsonschwegler/Desktop/Internship/es9b04833_si_004.csv")

#Create a FingerprintGenerator object
fpgen = AllChem.GetRDKitFPGenerator()

#create a list of SMILES (simplified molecular-input line entry system) strings from the df data frame
smiles_list=df["smiles"].values.tolist()

#Construct a rdkit.Chem.rdchem.Mol object from each SMILES string
mol_list=[Chem.MolFromSmiles(x) for x in smiles_list]

#convert rdkit.Chem.rdchem.Mol objects to fingerprint bit vectors
fps = [fpgen.GetFingerprint(x) for x in mol_list]

#create a list of np arrays of Morgan Fingerprints for each molecule
fp = []
for i in range(len(mol_list)):
    fp.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol_list[i], radius=2, nBits=1024)))

#set the column 'fingerprint' to the array fp
df["fingerprint"]= fp

#Pull all of the relevant columns from the dataframe
column_names = [col for col in df.columns if col not in ["smiles", "fingerprint"]]

#set X to the column 'fingerprint' (input)
X = df["fingerprint"].copy()

#set Y to the columns in column names (output)
Y = df[column_names].copy()

#Convert X from a pandas series to a np array
fp_list = [np.array(fp) for fp in X]
X = np.vstack(fp_list)


Y_np = df[column_names].to_numpy()
'''

X = np.load('/Users/wilsonschwegler/PycharmProjects/comps/X_train_target_7.npy')
Y = np.load('/Users/wilsonschwegler/PycharmProjects/comps/Y_train_target_7.npy')

#Principal Component Analysis: dimensionality reduction method
#Reduce the number of varuables of a data set, while preserving as much information as possible
pca = PCA(n_components = 2)

#[0, 1, 0, ..., 1] -> [0.543, 0.867]
X_dimension_reduced = pca.fit_transform(X)

#Plot the dimension reduced data
plt.scatter(X_dimension_reduced[:,0],X_dimension_reduced[:,1])


#Intertia measures how well a data set was clustered
#A good model is one with a lot inertia and number of clusters
#Find the optimal number of clusters by plotting the inertias and using the elbow method
inertias = []



#Outline data points that are bioactive with red
color_array=[]

for i in range (0,len(Y)):
    if Y[i] == 1:
        color_array.append('orange')
    else:
        color_array.append('black')





# Plot the data colored with their clusters
# plt.scatter(X_dimension_reduced[:,0], X_dimension_reduced[:,1], c=kmeans.labels_)
plt.title("Target 7")

plt.scatter(X_dimension_reduced[:, 0], X_dimension_reduced[:, 1], c=color_array)

plt.show()