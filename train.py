import torch 
from torch.utils.data import DataLoader
from data import Monecular
from model import AutoEncoder, VariationalAutoEncoder


def train(model, data_loader, optimzizer, epochs):
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        for i, (feature, _) in enumerate(train_loader):
            prediction = model(feature)
            loss = criterion(feature, prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'model.pth.tar')

dataset = Monecular('sample_data1.csv')
train_loader = DataLoader(dataset)

ae_model = AutoEncoder(260, 128, 32)
vae_model = VariationalAutoEncoder(260, 128, 32)

optimizer = torch.optim.SGD(ae_model.parameters(), lr=0.1, momentum=0.8)

epochs = 5

train(ae_model, train_loader, optimizer, epochs)