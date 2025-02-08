import torch
import numpy as np
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib as mpl
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
mpl.rcParams['text.usetex'] = True
def plot_3D(predictions = None):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    predictions = predictions.squeeze(1)
    # Make data.
    data_we = pd.read_csv(r"D:\pycharmproject\second try\x1000.csv", header=None)
    we = np.array(data_we.values)
    we = torch.from_numpy(we).to(device)
    obs = torch.cat((we, predictions), 1).to(device)
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
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 12,
            }
    ax.set_xlabel(' $\omega_m$(rpm)', fontdict=font)
    ax.set_ylabel('$i_d$(A)', fontdict=font)
    ax.set_zlabel('$i_q$(A)', fontdict=font)
    plt.rc('font', family='Times New Roman')

    surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm,

                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_zlim(-1, 1)

    ax.zaxis.set_major_locator(LinearLocator(5))

    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.

    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
    plt.savefig(r"tansformer.svg", format="svg")
    plt.show()
def plot_trajectories(fig, obs=None, noiseless_traj=None, times=None, trajs=None, save=None, title=None):
        fig = plt.figure()
        mpl.rcParams['axes.unicode_minus'] = False
        font ={'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 12,
        }
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('$i_d$[A]', fontdict=font)
        ax.set_ylabel('$i_q$[A]', fontdict=font)
        plt.legend(loc='best')
        obsc = obs.cpu()
        z = np.array([o.detach().numpy() for o in obsc])
        z = np.reshape(z, [-1, 2])
        ax.plot(z[:, 0], z[:, 1], color='greenyellow', alpha=0.5,label='prediction')
        trajsc = trajs.cpu()
        z = np.array([o.detach().numpy() for o in trajsc])
        z = np.reshape(z, [-1, 2])
        ax.plot(z[:, 0], z[:, 1], color='k', alpha=0.3,label='target')
        plt.legend(loc='best')
        # time.sleep(0.1)
        plt.savefig(r"jieyue.svg", format="svg")
        plt.show()

class ODEF(nn.Module):
    def forward_with_grad(self, z, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]
        out = self.forward(z)

        a = grad_outputs
        adfdz, *adfdp = torch.autograd.grad(
            # concatenating tuples
            (out,), (z,) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0) # unsqueeze(0) add dimension 1 to the position 0
            adfdp = adfdp.expand(batch_size, -1) / batch_size # passing -1 does not change dimension in that position
        return out, adfdz, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func, ode_solve, STEP_SIZE):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            # initialize z to len of time and type of z0
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            # solving throughout time
            for i_t in range(time_len - 1):
                # z0 updated to next step
                z0 = ode_solve(z0, torch.abs(t[i_t+1]-t[i_t]), func, STEP_SIZE)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        ctx.ode_solve = ode_solve
        ctx.STEP_SIZE = STEP_SIZE
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)
        ode_solve = ctx.ode_solve
        STEP_SIZE = ctx.STEP_SIZE

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdp = func.forward_with_grad(z_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            return torch.cat((func_eval, -adfdz, -adfdp), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i).view(bs, n_dim)
                # Compute direct gradients
                dLdz_i = dLdz[i_t]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z)), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, torch.abs(t_i-t[i_t-1]), augmented_dynamics, -STEP_SIZE)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]

            # Adjust adjoints
            adj_z += dLdz_0
        #print("\nreturned grad:\n", adj_p)
        return adj_z.view(bs, *z_shape), None, adj_p, None, None, None

class NeuralODE(nn.Module):
    def __init__(self, func, ode_solve, STEP_SIZE):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func
        self.ode_solve = ode_solve
        self.STEP_SIZE = STEP_SIZE

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0).to(device)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func, self.ode_solve, self.STEP_SIZE).to(device)
        if return_whole_sequence:
            return z
        else:
            return z[-1]


def RK(z0, n_steps, f, h):
    '''
    4th Order Runge Kutta Numerical Solver
    Input:
      z0: initial condition
      t0: initial time (not actual time, but the index of time)
      n_steps: the number of steps to integrate
      f: vector field
      h: step size
    Return:
      z: the state after n_steps
    '''
    z = z0
    for i in range(int(n_steps)):
        k1 = h * f(z)
        k2 = h * f(z + 0.5 * k1)
        k3 = h * f(z + 0.5 * k2)
        k4 = h * f(z + k3)

        z = z + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z
def sample_and_grow(ode_train, true_sampled_traj, true_sampled_times, epochs,
                    lr, hybrid, lookahead, loss_arr, plot_freq=50, save_path=None):
    """
    The main training loop

    :param ode_train: the ode to be trained
    :param true_sampled_traj: sampled observations (training data)obs
    :param true_sampled_times: sampled time stamps
    :param epochs: the total number of epochs to train
    :param lookahead: lookahead
    :param loss_arr: array where the training losses are stored
    :param plot_freq: frequency of which the trajectories are plotted
    :return: None
    """
    plot_title = "Epoch: {0} Loss: {1:.3e} Sim Step: {2} \n No. of Points: {3} Lookahead: {4} LR: {5}"
    optimizer = torch.optim.Adam(ode_train.parameters(), lr=lr)#
    n_segments = len(true_sampled_traj-200)#10000
    fig = plt.figure()

    for i in range(epochs):
        # Train Neural ODE
        true_segments_list = []
        for j in range(0, n_segments - lookahead + 1, 1):#(0,9999,1)
            true_sampled_segment = true_sampled_traj[j:j + lookahead]
            true_segments_list.append(true_sampled_segment)

        all_init = true_sampled_traj[:n_segments-lookahead+1] .to(device) # the initial condition for each segment([:9999
        true_sampled_time_segment = torch.tensor(np.arange(lookahead)).to(device)  # the times step to predict

        # predicting
        z_ = ode_train(all_init, true_sampled_time_segment, return_whole_sequence=True).to(device)
        z_ = z_.view(-1, 2).to(device)
        obs_ = torch.cat(true_segments_list, 1).to(device)
        obs_ = obs_.view(-1, 2).to(device)

        # computing loss
        loss = F.mse_loss(z_, obs_).to(device)
        loss_arr.append(loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        print(plot_title.format(i, loss.item(), ode_train.STEP_SIZE, n_segments,
                                lookahead - 1, lr))
    data = pd.read_csv(r"D:\pycharmproject\second try\1000idiq.csv", header=None)
    x1 = np.array(data.values)
    scaler = MinMaxScaler()
    x1 = scaler.fit_transform(x1)
    x_test = torch.from_numpy(x1[:, None, :])  # [len, 1, dim]
    obs_test = x_test.to(device)
    obs_test = Tensor.float(obs_test.detach()).to(device)  # [len, 1, dim]
    sequencetest = list(range(0, 1000))
    t_test = torch.tensor(sequencetest, dtype=float)
    time = t_test.to(device)
    time = time.detach().to(device)
    z_p = ode_train(obs_test[0], time, return_whole_sequence=True)
    predictions = z_p.detach()
    x_test = x_test.squeeze(1)
    predictions = predictions.squeeze(1)
    plot_trajectories(fig=fig,obs=predictions, trajs=x_test)
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


'''      if i % plot_freq == 0:
            # saving model
            if save_path is not None:
                CHECKPOINT_PATH = save_path + "Lorenz_" + name + ".pth"
                torch.save({'ode_train': ode_train, 'ode_true': ode_true, 'loss_arr': loss_arr},
                           CHECKPOINT_PATH)

            # computing trajectory using the current model
            z_p = ode_train(true_sampled_traj[0], true_sampled_times, return_whole_sequence=True)
            # plotting
            plot_trajectories(fig, obs=[true_sampled_traj], noiseless_traj=[true_sampled_traj],
                              times=[true_sampled_times], trajs=[z_p[:int(true_sampled_times[-1])]],
                              save=None, title=True)
            print(plot_title.format(i, loss.item(), ode_train.STEP_SIZE, n_segments,
                                    lookahead - 1, lr))
'''
class PINNKNODE(ODEF):
    """
    KNODE combining incorrectly SINDy-identified lorenz system and a neural network
    """
    def __init__(self):
        super(PINNKNODE, self).__init__()
        self.lin1 = nn.Linear(2, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64,128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64,32)
        self.lin6 = nn.Linear(32,2)


        self.relu = nn.ReLU()
#x(id,iq,we)p(ud,uq)
    def forward(self, x):

        x = self.relu(self.lin1(x)).to(device)
        x = self.relu(self.lin2(x)).to(device)
        x = self.relu(self.lin3(x)).to(device)
        x = self.relu(self.lin4(x)).to(device)
        x = self.relu(self.lin5(x)).to(device)
        x = self.lin6(x).to(device)

        return x
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    print("CUDA is available! Using GPU.")
else:
    device = torch.device("cpu")           # a CPU device object
    print("CUDA is not available. Using CPU.")
training_solver = RK
training_step_size = 0.0001
ode_train = NeuralODE(PINNKNODE(), training_solver, training_step_size).to(device)
hybrid = False
loss_arr = []
save_path = None
sampling_rate = training_step_size * 1  # seconds per instance i.e. 1/Hz, assumed to be lower than simulation rate
t0 = 0  # start point (index of time)
N_POINTS = 1000  # Number of times the solver steps. total_time_span = N_POINTS * simulation_step_size
data_test = pd.read_csv(r"D:\pycharmproject\second try\800idiq.csv", header=None)
x = np.array(data_test.values)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_real = torch.from_numpy(x[:, None, :])#[len, 1, dim]
obs =x_real.to(device)
sequence = list(range(0, 1000))
t =torch.tensor(sequence,dtype=float)
times = t.to(device)
obs = Tensor.float(obs.detach()) .to(device) # [len, 1, dim]
times = times.detach().to(device)
EPOCHs = 1000  # No. of epochs to train
LOOKAHEAD = 16  # lookahead
name = "lookahead_" + str(LOOKAHEAD - 1)
LR = 0.006  # learning rate
sample_and_grow(ode_train, obs, times, EPOCHs, LR, hybrid, LOOKAHEAD, loss_arr, plot_freq=200)
