import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mpl.rcParams['text.usetex'] = True
class ODE_CNN(nn.Module):
    def __init__(self):
        super(ODE_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=1)
        self.fc1 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(2)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Generate synthetic data
n_samples1 = 400
t_train = np.linspace(0, 400, n_samples1).reshape(-1, 1)
n_samples2 = 400
t_test = np.linspace(0, 400, n_samples2).reshape(-1, 1)
data_test = pd.read_csv(r"D:\pycharmproject\second try\400-800.csv", header=None)
x = np.array(data_test.values)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
train_data = x
test_data = x
x_train = torch.tensor(train_data, dtype=torch.float32).to(device)
x_test = torch.tensor(test_data,dtype= torch.float32).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32).to(device)
t_test = torch.tensor(t_test,dtype=torch.float32).to(device)


model = ODE_CNN()
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


n_epochs = 3000
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
    predictions = model(t_test).to(device)
# Ensure both 'x' and 'predictions' are aligned
x_test_actual = scaler.inverse_transform(x_test.cpu().numpy())  # Inverse scale to get original values
predictions = scaler.inverse_transform(predictions.cpu().numpy())

# Compute metrics on the entire dataset
mae = mean_absolute_error(x_test_actual, predictions)
mse = mean_squared_error(x_test_actual, predictions)
rmse = sqrt(mse)
r2 = r2_score(x_test_actual, predictions, multioutput='uniform_average')

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)


'''fig = plt.figure()

ax = fig.add_subplot(111,projection = '3d')

# Make data.

data_we =pd.read_csv(r"D:\pycharmproject\second try\we400-800.csv", header=None)
we = np.array(data_we.values)
we = torch.from_numpy(we).to(device)
obs = torch.cat((we, predictions), 1).to(device)

obsc = obs.cpu()
z = np.array([o.detach().numpy() for o in obsc])
z = np.reshape(z, [-1, 3])
X = z[:, 0]
Y = z[:, 1]
Z = z[:, 2]
# Plot the surface.
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 12,
        }
ax.set_xlabel(' $\omega_m$(rpm)',fontdict=font)
ax.set_ylabel('$i_d$(A)',fontdict=font)
ax.set_zlabel('$i_q$(A)',fontdict=font)
plt.rc('font',family='Times New Roman')

surf = ax.plot_trisurf(X,Y,Z, cmap=cm.coolwarm,

                       linewidth=0, antialiased=False)

# Customize the z axis.

ax.set_zlim(-1, 1)


ax.zaxis.set_major_locator(LinearLocator(5))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.

fig.colorbar(surf, shrink=0.5, aspect=5,pad = 0.2)
plt.savefig(r"CNN.svg", format="svg")
plt.show()'''