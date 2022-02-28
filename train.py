import torch.nn as nn
import torch.optim as optim

def Train(epochs, lr, Net, train_set, device):
    Net.to(device)
    Loss = nn.NLLLoss()
    optimizer = optim.Adam(Net.parameters(), lr)
    train_losses = []
    for i in range(epochs):
        Net.train()
        totalLoss = 0
        for iter, (x, y) in enumerate(train_set):
            predicted = Net(x.cuda())
            loss = Loss(predicted, y.cuda())
            Net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
        print("Epoca: " + str(i+1) + " Loss: " + str(totalLoss/len(train_set)))
        train_losses.append(totalLoss/len(train_set))
    
    return train_losses, Net