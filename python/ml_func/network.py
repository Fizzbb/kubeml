""" Example file that will be run in the functions """

import time
# misc
from typing import Dict

import ml_dataset
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tdata
import torchvision.transforms as transforms
# import the utils and dataset
import train_utils
# Flask and logging
from flask import current_app, jsonify

# params of the training
train_params: train_utils.TrainParams = None

# Set some global stuff
tensor_dict: Dict[str, torch.Tensor] = dict()  # Tensor to accumulate the gradients


# Define the network that we'll use to train
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def create_model():
    """Creates the model used to train the network

    For this example we'll be using the simple model from the MNIST examples
    (https://github.com/pytorch/examples/blob/master/mnist/main.py)
    """

    def init_weights(m: nn.Module):
        """Initialize the weights of the network"""
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    # Create the model and initialize the weights
    model = Net()

    # If the task is initializing the layers do so
    if train_params.task == 'init':
        current_app.logger.info('Initializing layers...')
        model.apply(init_weights)

    return model


def train(model: nn.Module, device,
          train_loader: tdata.DataLoader,
          optimizer: torch.optim.Optimizer) -> float:
    """Loop used to train the network"""
    model.train()
    loss = None
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        # Here save the gradients to publish on the database
        train_utils.update_tensor_dict(model, tensor_dict)
        optimizer.step()

        if batch_idx % 10 == 0:
            current_app.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                1, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()


def validate(model, device, val_loader: tdata.DataLoader) -> (float, float):
    """Loop used to validate the network"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)

    accuracy = 100. * correct / len(val_loader.dataset)
    current_app.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return accuracy, test_loss


# The function that will be run by default when the fission func is invoked
# TODO fill the rest of the function once we know how to load the data in Kubernetes
def main():
    global train_params

    start = time.time()
    current_app.logger.info(f'Started serving request')

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_app.logger.info('Running on device', device)

    # 1) Parse args to see the kind of task we have to do
    train_params = train_utils.parse_url_args()

    # Create the transformation
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # build the model
    model: nn.Module = create_model()

    # If we just want to init save the model and return
    if train_params.task == 'init':
        # Save the models and return the weights
        train_utils.save_model_weights(model, train_params)
        return f'Model saved, layers are {[name for name, layer in model.named_children() if hasattr(layer, "bias")]}'

    # For training or validation we need to
    # 1) create the dataset
    # 2) load the model weights
    # 3) train or validate
    # (if we train) publish the gradients on the cache
    dataset = ml_dataset.MnistDataset(func_id=train_params.func_id, num_func=train_params.N,
                                      task=train_params.task, transform=transf)
    train_utils.load_model_weights(model, train_params)
    # TODO receive the batch size through the api call
    loader = tdata.DataLoader(dataset, batch_size=128)
    current_app.logger.info(f'built dataset of size {dataset.data.shape} task is {train_params.task}')

    # If we want to validate we call test, if not we call train, we return the stats from the
    if train_params.task == 'val':
        acc, loss = validate(model, device, loader)
        current_app.logger.info(f"""Task is validation, received parameters are 
                funcId={train_params.func_id}, N={train_params.N}, task={train_params.task}, 
                psId={train_params.ps_id}, psPort={train_params.ps_port}
                completed in {time.time() - start}""")
        return jsonify(accuracy=acc, loss=loss)

    elif train_params.task == 'train':
        # TODO make this lr this also a parameter
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss = train(model, device, loader, optimizer)

        # After training save the gradients
        train_utils.save_gradients(tensor_dict, train_params)
        current_app.logger.info(f"""Task is training, received parameters are 
                funcId={train_params.func_id}, N={train_params.N}, task={train_params.task}, 
                psId={train_params.ps_id}, psPort={train_params.ps_port}
                completed in {time.time() - start}""")
        return jsonify(loss=loss)