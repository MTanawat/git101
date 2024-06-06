import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

from GCN.data_processing import Data_Loader
from GCN.graph import Graph
from GCN.sgcn_lstm_pytorch_cbam_eca import SgcnLstm
from GCN.structure_learner import EdgeStructureLearner
from torchsummary import summary
from thop import profile
import argparse
output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('cuda.is_available', torch.cuda.is_available())

# Create the parser
my_parser = argparse.ArgumentParser(description='List of argument')

# Add the arguments
my_parser.add_argument('--ex', type=str, default='Kimore_ex5',
                       help='the name of the exercise.', required=True)

my_parser.add_argument('--lr', type=float, default=0.0001,
                       help='initial learning rate for optimizer.')

my_parser.add_argument('--epoch', type=int, default=1000,
                       help='number of epochs to train.')

my_parser.add_argument('--batch_size', type=int, default=10,
                       help='training batch size.')

args = my_parser.parse_args()

# Seed for reproducibility
random_seed = 20
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# import the whole dataset
data_loader = Data_Loader(args.ex)
#print('x_train',data_loader.scaled_x_joint.shape,'handcrafted', data_loader.scaled_x_handcrafted.shape, 'x_train',data_loader.scaled_y_joint.shape)
# import the graph data structure
graph = Graph(len(data_loader.body_part))

# Split the data into training and validation sets while preserving the distribution
train_x_joint, test_x_joint, train_y, test_y = train_test_split(
    data_loader.scaled_x, data_loader.scaled_y, test_size=0.2, random_state=random_seed)

# Convert to PyTorch tensors
train_x_joint, test_x_joint = torch.tensor(train_x_joint, dtype=torch.float64).to(output_device), torch.tensor(test_x_joint, dtype=torch.float64).to(output_device)
#train_x_handcrafted, test_x_handcrafted = torch.tensor(train_x_handcrafted, dtype=torch.float64).to(output_device), torch.tensor(test_x_handcrafted, dtype=torch.float64).to(output_device)
train_y, test_y = torch.tensor(train_y, dtype=torch.float64).to(output_device), torch.tensor(test_y, dtype=torch.float64).to(output_device)
print('train_x_joint',train_x_joint.shape)
# Train the algorithm
num_nodes = len(data_loader.body_part)
A = graph.AD
A2 = graph.AD2 

model = SgcnLstm(len(data_loader.body_part)).to(output_device).double()
print(model)
criterion = nn.SmoothL1Loss(beta=1)
#criterion = torch.nn.MSELoss(size_average=False)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
edge_learner = EdgeStructureLearner(num_nodes, dim=3, initial_adjacency=A)
edge_learner2 = EdgeStructureLearner(num_nodes, dim=3, initial_adjacency=A2)
print('edge_learner >>>>',edge_learner )
edge_learner.to('cuda')
edge_learner2.to('cuda')
# Lists to store training and testing loss for plotting
train_loss_list = []
test_loss_list = []

# Train the model
for epoch in range(args.epoch):
    optimizer.zero_grad()
    edge_learner.adj = nn.Parameter(edge_learner.adj.to(output_device))
    edge_learner2.adj = nn.Parameter(edge_learner2.adj.to(output_device))
    
    # Forward pass and compute training loss
    output = model(train_x_joint, train_y, edge_learner.adj,
                   edge_learner2.adj, graph.bias_mat_1.to(output_device), graph.bias_mat_2.to(output_device))
    loss = criterion(output, train_y)
    
    # Backward pass and optimization step
    loss.backward()
    optimizer.step()

    # Record training loss for plotting
    train_loss_list.append(loss.item())
    
    # Evaluate model on test data to compute test loss
    with torch.no_grad():
        test_output = model(test_x_joint, test_y, graph.AD.to(output_device), graph.AD2.to(output_device), graph.bias_mat_1.to(output_device), graph.bias_mat_2.to(output_device))
        test_loss = criterion(test_output, test_y)
        test_loss_list.append(test_loss.item())

    # Print training and testing loss for current epoch
    print(f'Epoch [{epoch + 1}/{args.epoch}], Training Loss: {loss.item()}, Testing Loss: {test_loss.item()}')


plt.plot(train_loss_list, label='Training Loss')
plt.plot(test_loss_list, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Over Epochs')
plt.legend()
plt.savefig('training_loss_plot_ad.jpg')
plt.show()

# Test the model
with torch.no_grad():
    y_pred = model(test_x_joint, test_y, graph.AD.to(output_device), graph.AD2.to(output_device), graph.bias_mat_1.to(output_device), graph.bias_mat_2.to(output_device))

y_pred = torch.tensor(data_loader.sc2.inverse_transform(y_pred.cpu().numpy()), dtype=torch.float64)
test_y = torch.tensor(data_loader.sc2.inverse_transform(test_y.cpu().numpy()), dtype=torch.float64)

print('predict', y_pred)
print('label', test_y)
test_y_np = test_y.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()

# Stack test_y and y_pred horizontally
data_to_save = np.hstack((test_y_np, y_pred_np))

# Save the stacked data to a CSV file
np.savetxt("labels_predictions.csv", data_to_save, delimiter=",", fmt='%.6f')

def mean_absolute_deviation(y_true, y_pred):
    # Calculate absolute deviations
    abs_deviation = torch.abs(y_pred - y_true)
    
    # Compute mean absolute deviation
    mean_abs_deviation = torch.mean(abs_deviation).item()
    
    return mean_abs_deviation
    
    
def root_mean_square_error(y_true, y_pred):
    # Calculate squared errors
    squared_errors = (y_pred - y_true) ** 2
    
    # Compute mean squared error using PyTorch
    mean_squared_error = torch.mean(squared_errors).item()
    
    # Calculate RMS error (square root of MSE)
    rms_error = sqrt(mean_squared_error)
    
    return rms_error


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mad = mean_absolute_deviation(test_y, y_pred)
rms_dev = root_mean_square_error(test_y, y_pred)
mape = mean_absolute_percentage_error(test_y, y_pred)
print('Mean absolute deviation:', mad)
print('RMS deviation:', rms_dev)
print('MAPE: ', mape)
