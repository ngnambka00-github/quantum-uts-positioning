from copy import deepcopy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from zmq import device

from quantum_model import QuantumLayer

# define global variable
N_INPUTS = 4
N_QUBITS = 1
N_OUT=2
N_SHOTS=512

class MeanSquareError(nn.Module):
    def __init__(self):
        super(MeanSquareError, self).__init__()

    def forward(self, A, B):
        errA = torch.zeros(A.shape[0])
        for i in range(A.shape[0]):
            errA[i] = torch.linalg.norm(A[i] - B[i])

        return errA.mean()


class UTSDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train).float()
        self.y_data = torch.from_numpy(y_train).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# define netword has quantum_layer
class Net(nn.Module):
    def __init__(self, quantum=True, num_classes=N_OUT):
        super(Net, self).__init__()
        self.init_weight()

        self.quantum = quantum
        self.fc_in = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, N_INPUTS),
            nn.BatchNorm1d(N_INPUTS),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classical = nn.Linear(N_INPUTS, num_classes)
        
        self.quantum_layer = QuantumLayer.apply
        self.pre_out = nn.Linear(N_INPUTS * (2 ** N_QUBITS), N_INPUTS)
        self.pre_bn = nn.BatchNorm1d(N_INPUTS)
        self.fc_out = nn.Linear(N_INPUTS, num_classes)
        

    def init_weight(self):
        for m in self.modules():
          if m.__class__.__name__.startswith('Linear'):
              torch.nn.init.xavier_normal_(m.weight)
              m.bias.data.fill_(0.0001)

    def forward(self, x):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.quantum:
            x = self.fc_in(x) # vao 64 -> ra n_inputs
            quantum_out = torch.cat([self.quantum_layer(x_i) for x_i in x])
            quantum_out = quantum_out.to(device)

            x = self.pre_out(quantum_out.float())
            x = F.relu(self.pre_bn(x))
            x = F.relu(self.fc_out(x))
        else:
            x = self.fc_in(x)
            x = self.classical(x)

        return x

def train_model_deep(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, data_path=None, num_epochs=25):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_model_state = deepcopy(model.state_dict())
    best_loss = 1000000.0
    
    statistic_loss = {
        "train": [], 
        "val": []
    }
    
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(f"Epoch {epoch} -> phase {phase}: ")

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            count = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                count += 1
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    print(f"\tBatch: {count}/{phase}: ")
                    
                    start = time.time()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    end = time.time()
                    print(f"\tTime forward: {end - start}")

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()

                        start = time.time()
                        loss.backward()
                        end = time.time()
                        print(f"\tTime backward: {end - start}")

                        start = time.time()
                        optimizer.step()
                        end = time.time()
                        print(f"\tTime update weight: {end - start}")

                    print()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            statistic_loss[phase].append(epoch_loss)
            print(f"\t\tEpoch loss [{epoch}] / phase_{phase}: {epoch_loss}")

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print(f"\t\t\tBEST LOST: {epoch_loss}")
                best_loss = epoch_loss
                best_model_state = deepcopy(model.state_dict())

    # model.load_state_dict(best_model_state)
    # save best_model_state
    if not (data_path is None):
        torch.save(best_model_state, data_path["save_model_path"])
        np.savez(data_path["loss_train"], np.array(statistic_loss["train"]))
        np.savez(data_path["loss_val"], np.array(statistic_loss["val"]))

    # return thong ke loss (train, val) | lowest_loss | best model state
    return statistic_loss, best_loss, best_model_state


# if __name__ == "__main__":
#     net = Net(quantum=True)
#     a = torch.rand((8, 64))
#     out = net(a)
#     out.shape