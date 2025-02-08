import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter
# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(1, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64,128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64,32)
        self.lin6 = nn.Linear(32,2)




    def forward(self, x):
        x.view(-1,2)
        x = self.relu(self.lin1(x)).to(device)
        x = self.relu(self.lin2(x)).to(device)
        x = self.relu(self.lin3(x)).to(device)
        x = self.relu(self.lin4(x)).to(device)
        x = self.relu(self.lin5(x)).to(device)
        x = self.lin6(x).to(device)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Generate synthetic data
n_samples1 = 1000
t_train = np.linspace(0, 1000, n_samples1).reshape(-1, 1)
n_samples2 = 1000
t_test = np.linspace(0, 1000, n_samples2).reshape(-1, 1)
data_test = pd.read_csv(r"D:\pycharmproject\second try\1000idiq.csv", header=None)
x = np.array(data_test.values)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
train_data = x
test_data = x
x_train = torch.tensor(train_data, dtype=torch.float32).to(device)
x_test = torch.tensor(test_data,dtype= torch.float32).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32).to(device)
t_test = torch.tensor(t_test,dtype=torch.float32).to(device)


# Define the model, loss function, and optimizer
model = MLP()
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
n_epochs = 2000
for epoch in range(n_epochs):
    model.train().to(device)
    optimizer.zero_grad()
    outputs = model(t_train).to(device)
    loss = criterion(outputs, x_train).to(device)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
# Predict with the trained model
model.eval().to(device)
with torch.no_grad():
    x_pred = model(t_test).to(device)

fig = plt.figure()

ax = fig.add_subplot(111,projection = '3d')

# Make data.

data_we =pd.read_csv(r"D:\pycharmproject\second try\we1000.csv", header=None)
we = np.array(data_we.values)
we = torch.from_numpy(we).to(device)
obs = torch.cat((we, x_pred), 1).to(device)
'''sequence = list(range(0, 2000))
t =torch.tensor(sequence,dtype=float)
times = t/10000
times = times.unsqueeze(1)
obs = torch.cat((times, x_real), 1)'''
obsc = obs.cpu()
z = np.array([o.detach().numpy() for o in obsc])
z = np.reshape(z, [-1, 3])
X = z[:, 0]
Y = z[:, 1]
Z = z[:, 2]
# Plot the surface.

surf = ax.plot_trisurf(X,Y,Z, cmap=cm.coolwarm,

                       linewidth=0, antialiased=False)

# Customize the z axis.

ax.set_zlim(-1, 1)


ax.zaxis.set_major_locator(LinearLocator(5))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


