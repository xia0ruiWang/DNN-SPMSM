def plot_trajectories(fig, obs=None, noiseless_traj=None,times=None, trajs=None, save=None, title=None):
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    times = times.unsqueeze(1)
    times = times.unsqueeze(2)
    if title is not None:
      ax1.set_title(' Predicted Trajectory of Id \n')
      ax2.set_title(' Predicted Trajectory of Iq \n' )
    if obs is not None:
      obs = torch.cat((times, obs), 2)
      obsc = obs.cpu()
      z = np.array([o.detach().numpy() for o in obsc])
      z = np.reshape(z, [-1, 3])
      ax1.scatter(z[:, 1],z[:, 2],  marker='.', color='k', alpha=0.5, linewidths=0, s=45)

    if trajs is not None:
      trajs = torch.cat((times, trajs), 2)
      trajsc = trajs.cpu()
      z = np.array([o.detach().numpy() for o in trajsc])
      z = np.reshape(z, [-1, 3])
      ax1.plot(z[:, 1],z[:, 2], color='r', alpha=0.3)
    # time.sleep(0.1)
    plt.show()