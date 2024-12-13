import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
'''
import warnings
target = 25
warnings.filterwarnings("ignore", category=Warning)

#reads the csv file to a pd data frame (df)
df = pd.read_csv("/Users/wilsonschwegler/Desktop/Internship/es9b04833_si_003_2.csv")

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
    fp.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol_list[i], radius=3, nBits=1024)))

#set the column 'fingerprint' to the array fp
df["fingerprint"]= fp
df = df.fillna(-1)

#Pull all of the relevant columns from the dataframe
column_names = [col for col in df.columns if col not in ["smiles", "fingerprint"]]

#set X to the column 'fingerprint' (input)
X = df["fingerprint"].copy()

#set Y to the columns in column names (output)
Y = df[column_names].copy()

#Convert X from a pandas series to a np array
fp_list = [np.array(fp) for fp in X]
X = np.vstack(fp_list)

Y_np = Y.to_numpy()
print(len(Y_np))
print(len(X))
X_final = []
Y_final = []
for i in range(len(Y_np)):
    if (Y_np[i][target] == 0.0 or Y_np[i][target] == 1.0):
        X_final.append(X[i])
        Y_final.append(Y_np[i][target])
    else:
        pass


print(len(Y_final))
print(len(X_final))






#create the training and test split (train = 85% test = 15%)
X_train, X_test, Y_train, Y_test = train_test_split(X_final,
                                                    Y_final,
                                                    train_size=.85,
                                                    random_state=25)

np.save('X_train_target_' + str(target) + '.npy', X_train)
np.save('X_test_target_' + str(target) + '.npy', X_test)
np.save('Y_train_target_' + str(target) + '.npy', Y_train)
np.save('Y_test_target_' + str(target) + '.npy', Y_test)
/Users/wilsonschwegler/PycharmProjects/comps/X_test_target_4.npy
'''
target = 22
#X_train = np.load('/Users/wilsonschwegler/PycharmProjects/pythonProject7/X_train_target_'+str(target)+'.npy')
#X_test = np.load('/Users/wilsonschwegler/PycharmProjects/pythonProject7/X_test_target_'+str(target)+'.npy')
#Y_train = np.load('/Users/wilsonschwegler/PycharmProjects/pythonProject7/Y_train_target_'+str(target)+'.npy')
#Y_test = np.load('/Users/wilsonschwegler/PycharmProjects/pythonProject7/Y_test_target_'+str(target)+'.npy')


X_train = np.load('/Users/wilsonschwegler/PycharmProjects/comps/X_train_target_'+str(target)+'.npy')
X_test = np.load('/Users/wilsonschwegler/PycharmProjects/comps/X_test_target_'+str(target)+'.npy')
Y_train = np.load('/Users/wilsonschwegler/PycharmProjects/comps/Y_train_target_'+str(target)+'.npy')
Y_test = np.load('/Users/wilsonschwegler/PycharmProjects/comps/Y_test_target_'+str(target)+'.npy')

data_list=[]
for i in range (0, len(X_train)):
    data_list.append(Y_train[i])

#weights = compute_class_weight(class_weight="balanced", classes=np.unique(data_list), y=data_list)

#print(weights)

#weights = torch.tensor(weights, dtype=torch.float32)
classes = np.unique(Y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=Y_train)

# Create a dictionary that maps class labels to their corresponding weights
class_weight_dict = dict(zip(classes, weights))


print(len(Y_train))
print(len(X_train))
X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")

#for i in range(len(Y_train)):
    #print(Y_train[i])
print(Y_train[0])
print(Y_train[1])
print(Y_train[2])



# Define the model
model = Sequential([
    Dense(1024, input_dim=1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(600, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(200, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.summary()

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy", "KLDivergence"])

# Train the model
history = model.fit(X_train, Y_train, class_weight=class_weight_dict, epochs=130, batch_size=32, validation_split=0.2)

# Save the final model
model.save('NN_target_'+str(target)+'.keras')

#model = load_model('NN_target_'+str(target)+'.keras')




Y_pred_test = model.predict(X_test)
Y_pred_fin = []
for i in range (0, len(Y_pred_test)):
    if(Y_pred_test[i] >= 0.5):
        Y_pred_fin.append(1)
    else:
        Y_pred_fin.append(0)

print(classification_report(Y_test, Y_pred_fin))



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


Y_pred_test = model.predict(X_test)
Y_pred_fin = []
for i in range (0, len(Y_pred_test)):
    if(Y_pred_test[i] >= 0.5):
        Y_pred_fin.append(1)
    else:
        Y_pred_fin.append(0)

print(classification_report(Y_test, Y_pred_fin))

cm = confusion_matrix(Y_test, Y_pred_fin, labels=[0.0, 1.0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0.0, 1.0])
disp.plot()
plt.show()

