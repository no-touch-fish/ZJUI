import torch
import torch.nn as nn


def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    # raise NotImplementedError("You need to write this part!")
    model = nn.Sequential(
        nn.Linear(2,3),
        nn.Sigmoid(),
        nn.Linear(3,5)
    )
    return model

def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    # raise NotImplementedError("You need to write this part!")
    loss = nn.CrossEntropyLoss()
    return loss


class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=5, stride=1, padding=2)

        self.fc1 = nn.Linear(9 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 5)
        self.relu = nn.ReLU()
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        x = x.view(-1,3,31,31)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 9 * 7 * 7)
        x = self.relu(self.fc1(x))
        y = self.fc2(x)
        return y
        ################## Your Code Ends here ##################


def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer
    model = NeuralNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.03,weight_decay=0.02)
    for epoch in range(epochs):
        for features, labels in train_dataloader:
            y_prec = model(features)
            loss = loss_fn(y_prec,labels)
            optimizer.zero_grad()   # Clear previous gradients, will see more about this later
            loss.backward()         # back propagation
            optimizer.step()
    
    ################## Your Code Ends here ##################

    return model
