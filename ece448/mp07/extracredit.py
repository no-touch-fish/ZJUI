import torch, random, math, json
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE=torch.float32
DEVICE=torch.device("cpu")

###########################################################################################
def trainmodel():

    model = torch.nn.Sequential(torch.nn.Flatten(), 
                                torch.nn.Linear(in_features=8*8*15, out_features=1)
                                # torch.nn.ReLU(),
                                # torch.nn.Linear(in_features=250, out_features=1)
                                )
    

    model[1].weight.data = initialize_weights()
    model[1].bias.data = torch.zeros(1)

    trainset = ChessDataset(filename='extracredit_train.txt')
    validset = ChessDataset(filename='extracredit_validation.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=1000, shuffle=True)
    lr = 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    min_valid_loss = np.inf

    for epoch in range(100):
        model.train()
        for x,y in validloader:
            # x has shape [batchsize, 15, 8, 8]
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            result = model(x)
            loss = loss_func(result, y)
            loss.backward()
            optimizer.step()
        
        valid_loss = 0.0
        model.eval()
        for x,y in trainloader:
            # x has shape [batchsize, 15, 8, 8]
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            result = model(x)
            loss = loss_func(result, y)
            valid_loss += loss.item()
        
        if min_valid_loss > valid_loss:
            torch.save(model, 'model_ckpt.pkl')
            min_valid_loss = valid_loss


###########################################################################################
if __name__=="__main__":
    trainmodel()
    
    