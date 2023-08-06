import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from torch.utils.data import Dataset, DataLoader, random_split
import gym
import pybullet_data
import pybullet_envs
import os
import sys
import argparse

# parser = argparse.ArgumentParser(description='Neural ODE arguments')

# # Add arguments
# parser.add_argument('--mode', type=str, help='Inference or Train', required=True)
# parser.add_argument('--model', type=str, help='NeuralODE, NeuralODEResidual, or ResidualDynamics', required=True)
# parser.add_argument('--solver', type=str, help='Solver: dopri5, euler, or dopri8', required=True)

# # Parse arguments
# args = parser.parse_args()

# # Get the values of the arguments
# mode = args.mode
# model = args.model
# solver = args.solver

def collect_data_random(env, num_trajectories=1000, trajectory_length=50):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: PyBullet Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = []
    for i in range(num_trajectories):

        state = env.reset()
        states = np.zeros((trajectory_length, env.observation_space.shape[0]), dtype=np.float32)
        next_states = np.zeros((trajectory_length, env.observation_space.shape[0]), dtype=np.float32)
        actions = np.zeros((trajectory_length, 1), dtype=np.float32)
        states_diff = np.zeros((trajectory_length, env.observation_space.shape[0]), dtype=np.float32)

        for t in range(trajectory_length):

          action = env.action_space.sample()
          next_state, _, done, _ = env.step(action)
          states[t] = state
          actions[t] = action
          next_states[t] = next_state

          # Call differentiator to get the first order differential of states only, give current action and current state as the input
          states_diff[t] = dynamics_diff(state,action)


          if not done:
            state = next_state
            continue
            
          if done:
            state = env.reset()

        trajectory = {'states': states, 'actions': actions, 'next_states':next_states}
        collected_data.append(trajectory)

    return collected_data

def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:
Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here

    dataset = SingleStepDynamicsDataset(collected_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # ---
    return train_loader, val_loader

class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}, u_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
     'next_action': u_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
     u_{t+1}: torch.float32 tensor of shape (action_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] # Correct : 10

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state, next_action).
        The class description has more details about the format of this data sample.
        """
        
        traj_idx = item // self.trajectory_length
        #print("traj: ", traj_idx)
        #t = item % self.trajectory_length

        x_t = self.data[traj_idx]['states']
        u_t = self.data[traj_idx]['actions']
        x_tp1 = self.data[traj_idx]['next_states']
        #u_tp1 = self.data[traj_idx]['actions'][t + 1] if t < self.trajectory_length - 1 else np.zeros_like(u_t)
        #x_t_diff = self.data[traj_idx]['states_diff']


        sample = {
            'state': x_t,
            'action': u_t,
            'next_state': x_tp1
        }

        return sample


# Write a function to get the first order differentiation of the states
def dynamics_diff(state,action):

    x, theta, x_dot, theta_dot = state[0], state[1], state[2], state[3]
    u = action
    g = 9.81
    mc = 1
    mp = 0.1
    l = 0.5

    theta_ddot = (g * np.sin(theta) - np.cos(theta) * ((u + mp * l * (theta_dot**2) * np.sin(theta)) / (mc + mp))) / (l * ((4/3) - (mp * ((np.cos(theta))**2)) / (mc + mp)))
    x_ddot = (u + mp * l * ((theta_dot**2) * np.sin(theta) - theta_ddot * np.cos(theta))) / (mc + mp)

    dx = x_dot
    dtheta = theta_dot
    dx_dot = x_ddot
    dtheta_dot = theta_ddot
    
    dzdt = np.array([dx, dtheta, dx_dot, dtheta_dot]) 

    return dzdt


class NeuralODEModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralODEModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = 100

        # Define the architecture of the model
        self.net = nn.Sequential(
            nn.Linear(self.input_dim , hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, self.output_dim)
        )

        

    def forward(self, x,t):
      x = t
      dx_dt = self.net(x)

      return dx_dt

class ResidualNeuralODEModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualNeuralODEModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = 100

        # Define the architecture of the model
        self.net = nn.Sequential(
            nn.Linear(self.input_dim , hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, self.output_dim)
        )

        

    def forward(self, x,t):
      x = t
      dx_dt = self.net(x)

      return dx_dt

# class ResidualNeuralODEModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ResidualNeuralODEModel, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         hidden_dim = 512

#         # Define the architecture of the model
#         self.net = nn.Sequential(
#             nn.Linear(self.input_dim , hidden_dim),
#             nn.Softplus(),
#             nn.Linear(hidden_dim, 256),
#             nn.Softplus(),
#             nn.Linear(256, 128),
#             nn.Softplus(),
#             nn.Linear(128, 64),
#             nn.Softplus(),
#             nn.Linear(64, self.output_dim)
#         )
#         # self.fc1 = nn.Linear(input_dim, 100)
#         # self.fc2 = nn.Linear(100,100)
#         # self.fc6 = nn.Linear(100, output_dim)
#         # self.activation = nn.ReLU()
        

#     def forward(self, x,t):
#       x = t
#       residual = self.net(x) #Predicted residual

#       next_z = x + residual

#       return next_z

class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = 100

        # Define the architecture of the model
        self.net = nn.Sequential(
            nn.Linear(self.input_dim , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        # ---

    def forward(self, x):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        
        residual = self.net(x)
        next_state = torch.add(x,residual)
        return next_state

class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, predicted_state, true_state):
        mse_loss = nn.MSELoss()
        L1 = mse_loss(predicted_state[:,0], true_state[:,0])
        L2 = mse_loss(predicted_state[:,1], true_state[:,1])
        L3 = mse_loss(predicted_state[:,2], true_state[:,2])
        L4 = mse_loss(predicted_state[:,3], true_state[:,3])
        return L1*100. + L2 + L3*100. + L4

def train(train_loader,val_loader,num_epochs,lr,horizon,time_interval,model_type,optim,solver='dopri5'):
  if model_type == "NeuralODE":
    model = NeuralODEModel(4+1,4+1)
  elif model_type == "ResidualNeuralODE":
    model = ResidualNeuralODEModel(4+1,4+1)
  elif model_type == "ResidualDynamics":
    model = ResidualDynamicsModel(4+1,4+1)
  if optim == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  if optim == "SGD-M":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9)
  if optim == "RMS-Prop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
  loss_func = PoseLoss() # ONLY IF ODE TODO: Needs if condition

  train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, loss_func, num_epochs, horizon, time_interval,solver,model_type=model_type)

  
  # plot train loss and test loss:
  plot_losses(train_losses, val_losses)

  return model, train_losses, val_losses

def train_model(model, train_dataloader, val_dataloader, optimizer, loss_func, num_epochs, horizon, time_interval,solver,model_type):
    """
    Trains the given model for `num_epochs` epochs. Use Adam as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    train_losses = []
    val_losses = []
    timer = 50
    #torch.manual_seed(42)
    
    for epoch_i in range(num_epochs):
        train_loss_i = None
        val_loss_i = None
        c = torch.rand(1)

        train_loss_i = train_step(model=model, train_loader=train_dataloader, optimizer=optimizer, loss_func=loss_func, horizon=horizon, time_interval = time_interval,solver=solver,model_type=model_type)
        val_loss_i = val_step(model=model, val_loader=val_dataloader, loss_func=loss_func, horizon=horizon, time_interval = time_interval,solver=solver,model_type=model_type)
        pr = timer + c.item()
	
        print(f"Epoch {epoch_i+1}/{num_epochs}: train loss={train_loss_i:.4f}, val loss={val_loss_i:.4f} Elapsed seconds left : {pr:.4f}")
        timer -= 1
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

    return train_losses, val_losses

def train_step(model, train_loader, optimizer, loss_func, horizon, solver,time_interval,model_type) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0. 
    t = torch.arange(0, (horizon+1)*time_interval,time_interval) # TODO: Make this general based on horizon

    T = horizon
    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        state = batch['state'][i,:T] # (T,4)
        action = batch['action'][i,:T] # (T,1)
        next_state = batch['next_state'][i,:T] # (T,4)
        
        #input = torch.cat((state, action), dim=-1) # SHAPE: (T,5)
              
        # Loop through every action sequence in action and compute the next_state 
        next_state_pred = torch.zeros_like(state)
        state_t = state[0] # First state always initial state
        for j in range(len(action)):
          action_t = action[j]
          model_input_t = torch.cat((state_t,action_t),dim=-1)
          t_span = torch.tensor([t[j],t[j+1]])
          
          if model_type=="NeuralODE":  
            next_state_t = odeint(model, model_input_t, t_span, method = solver, atol=1e-7,rtol=1e-5)[-1] # Predicted Residual
            next_state_pred[j] = next_state_t[:4]
            state_t = next_state_pred[j]
          elif model_type=="ResidualNeuralODE":
            next_state_t = odeint(model, model_input_t, t_span, method = solver, atol=1e-7,rtol=1e-5)[-1] # Predicted Residual
            next_state_pred[j] = next_state_t[:4] 
            state_t = next_state_pred[j]
          elif model_type=="ResidualDynamics":
            next_state_t = model(model_input_t)
            next_state_pred[j] = next_state_t[:4] 
            state_t = next_state_pred[j]

        if model_type=="NeuralODE":         
          loss = loss_func(next_state_pred[:,:4], next_state) # COMPARE WITH THE NEXT_STATE
        elif model_type=="ResidualNeuralODE":
          residual_truth = next_state - state
          residual_pred = next_state_pred[:,:4] - state
          loss = loss_func(residual_pred[:,:4], residual_truth)
        elif model_type=="ResidualDynamics":
          residual_truth = next_state - state
          residual_pred = next_state_pred[:,:4] - state
          loss = loss_func(residual_pred[:,:4], residual_truth)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)

def val_step(model, val_loader, loss_func, horizon,solver,time_interval,model_type='NeuralODE') -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0. 
    #t = torch.arange(0, num_traj+1,1) # num_traj+1
    t = torch.arange(0, (horizon+1)*time_interval,time_interval)
    T = horizon

    model.eval()
    # ---
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            state = batch['state'][i,:T] # (T,4)
            #print(state.shape)
            action = batch['action'][i,:T] # (T,1)
            next_state = batch['next_state'][i,:T] # (T,4)
            
            # Loop through every action sequence in action and compute the next_state 
            next_state_pred = torch.zeros_like(state)
            state_t = state[0]
            for j in range(len(action)):
              action_t = action[j]
              model_input_t = torch.cat((state_t,action_t),dim=-1)
              #ONLY FOR ODE# # TODO: Add if statement
              t_span = torch.tensor([t[j],t[j+1]])
              #t_span = torch.arange(t[j],t[j+1],(t[j+1] - t[j])/horizon) # len = horizon
              
              if model_type=="NeuralODE":  
                next_state_t = odeint(model, model_input_t, t_span, method = solver, atol=1e-7,rtol=1e-5)[-1]
                next_state_pred[j] = next_state_t[:4]
                state_t = next_state_pred[j]
              elif model_type=="ResidualNeuralODE":
                next_state_t = odeint(model, model_input_t, t_span, method = solver, atol=1e-7,rtol=1e-5)[-1]
                next_state_pred[j] = next_state_t[:4] 
                state_t = next_state_pred[j]
              elif model_type=="ResidualDynamics":
                next_state_t = model(model_input_t)
                next_state_pred[j] = next_state_t[:4] 
                state_t = next_state_pred[j]

            if model_type=="NeuralODE":         
              loss = loss_func(next_state_pred[:,:4], next_state) # COMPARE WITH THE NEXT_STATE
            elif model_type=="ResidualNeuralODE":
              residual_truth = next_state - state
              residual_pred = next_state_pred[:,:4] - state
              loss = loss_func(residual_pred[:,:4], residual_truth)
            elif model_type=="ResidualDynamics":
              residual_truth = next_state - state
              residual_pred = next_state_pred[:,:4] - state
              loss = loss_func(residual_pred[:,:4], residual_truth)
            elif model_type=="ResidualDynamics":
              residual_truth = next_state - state
              residual_pred = next_state_pred[:,:4] - state
              loss = loss_func(residual_pred[:,:4], residual_truth)

            val_loss += loss.item()
    return val_loss/len(val_loader)

def plot_traj(model_pred,states_pybullet,T):
  # Plot and compare - They should be indistinguishable 
  t = torch.arange(0, T, 1)
  fig, axes = plt.subplots(2, 2, figsize=(8, 8))
  axes[0][0].plot(t[:T],model_pred[:, 0], label='model')
  axes[0][0].plot(t[:T],states_pybullet[:, 0], '--', label='pybullet')
  axes[0][0].title.set_text('x')
	# min_ = min(model_pred[:, 0].any(),states_pybullet[:, 0].any())
  max_ = max(model_pred[:, 0].any(),states_pybullet[:, 0].any())
  #axes[0][0].set_ylim([-0.01, max_])



  axes[0][1].plot(t[:T],model_pred[:, 1], label='model')
  axes[0][1].plot(t[:T],states_pybullet[:, 1], '--', label='pybullet')
  axes[0][1].title.set_text('theta')
	# min_ = min(model_pred[:, 1].any(),states_pybullet[:, 1].any())
	# max_ = max(model_pred[:, 1].any(),states_pybullet[:, 1].any())
	# axes[0][1].set_ylim([min_, max_])


  axes[1][0].plot(t[:T],model_pred[:, 2], label='model')
  axes[1][0].plot(t[:T],states_pybullet[:, 2], '--', label='pybullet')
  axes[1][0].title.set_text('x_dot')
	# min_ = min(model_pred[:, 2].any(),states_pybullet[:, 2].any())
	# max_ = max(model_pred[:, 2].any(),states_pybullet[:, 2].any())
	# axes[1][0].set_ylim([min_, max_])

  axes[1][1].plot(t[:T],model_pred[:, 3], label='model')
  axes[1][1].plot(t[:T],states_pybullet[:, 3], '--', label='pybullet')
  axes[1][1].title.set_text('theta_dot')
	# min_ = min(model_pred[:, 3].any(),states_pybullet[:, 3].any())
	# max_ = max(model_pred[:, 3].any(),states_pybullet[:, 3].any())
	# axes[0][0].set_ylim([min_, max_])

  axes[0][0].legend() 
  axes[0][1].legend()
  axes[1][0].legend()
  axes[1][1].legend()
  plt.tight_layout()

  plt.show()

def get_inference(model_pt,model_type,control_sequence,T,solver='dopri5') -> float:

  state_dict = torch.load(model_pt)
  if model_type=="NeuralODE":
    model = NeuralODEModel(4+1, 4+1)  # replace with your actual model class
  if model_type=="ResidualNeuralODE":
    model = ResidualNeuralODEModel(4+1, 4+1) 
  if model_type=="ResidualDynamics":
    model = ResidualDynamicsModel(4+1, 4+1) 
  model.load_state_dict(state_dict)
  t = torch.arange(0, T, 1)
  #control_sequence = torch.zeros_like(t, dtype=torch.float32)
  # Initialize the environment and collect data
  env = pybullet_envs.make('CartPoleBulletEnv-v1')
  env.action_space = gym.spaces.Box(low=-20.0, high=20.0, shape=(1,))
  x_lims = [-5, 5]  # TODO: Verify that they are the correct limits
  theta_lims = [-np.pi, np.pi]
  x_dot_lims = [-10, 10]
  theta_dot_lims = [-5 * np.pi, 5 * np.pi]
  env.state_space = gym.spaces.Box(low=np.array([x_lims[0], theta_lims[0], x_dot_lims[0], theta_dot_lims[0]]),
                              high=np.array([x_lims[1], theta_lims[1], x_dot_lims[1], theta_dot_lims[
                                  1]]))
  start_state=env.reset()
  states_pybullet = np.zeros((T, 4))
  states_pybullet[0] = start_state
  for t in range(1,T):
      states_pybullet[t] = env.step(control_sequence.numpy()[t-1])[0]
  #print("PyBullet Trajectory: \n", states_pybullet)

  next_state_pred_list = torch.zeros((T,4))
  next_state_pred_list[0] = torch.from_numpy(start_state).to(dtype=torch.float32)
  with torch.no_grad():
      model.eval()
      t = torch.arange(0, (T+1)*0.1,0.1)
      
      state_t = torch.from_numpy(start_state).to(dtype=torch.float32)
      for i in range(1,T):
        action_t = control_sequence[i-1]
        t_span = torch.tensor([t[i],t[i+1]])
        model_input_t = torch.cat((state_t,action_t),dim=-1)
        if model_type=="NeuralODE" or model_type=="ResidualNeuralODE":
          next_state_pred_list[i] = odeint(model,model_input_t,t_span,method=solver,atol=1e-4,rtol=1e-4)[-1][:4]
          state_t = next_state_pred_list[i]
        if model_type=="ResidualDynamics":
          next_state_t = model(model_input_t)
          next_state_pred_list[i] = next_state_t[:4] 
          state_t = next_state_pred_list[i]


      #print("Model Output: \n",next_state_pred_list)

  model_pred = next_state_pred_list.numpy()
  plot_traj(model_pred,states_pybullet,T)
  return states_pybullet,model_pred

# convert back to numpy for plotting


# if __name__ == '__main__':
#   batch_size = 500
#   collected_data = np.load('collected_data_20.npy', allow_pickle=True)
#   if mode == 'train':
#       print('Training mode selected')
#       # Code for training
#       train_loader, val_loader = process_data_single_step(collected_data, batch_size=batch_size)
#       # Else if it is --inference do this:
#       model = NeuralODEModel(4+1,4+1)
#       trained_model = train_ode(model,train_loader,val_loader,num_epochs=2,lr=1e-3,horizon=4,time_interval = 0.1, solver='dopri15') # Include time interval
#   elif mode == 'inference':
#       print('Inference mode selected')
#       # Code for inference
#       # Initialize the environment and collect data
#       env = pybullet_envs.make('CartPoleBulletEnv-v1')
#       env.action_space = gym.spaces.Box(low=-20.0, high=20.0, shape=(1,))
#   else:
#       print('Invalid mode selected. Please choose either "train" or "inference".')

#   if model == 'NeuralODE':
#       print('NeuralODE model selected')
#       # Code for NeuralODE model
#   elif model == 'NeuralODEResidual':
#       print('NeuralODEResidual model selected')
#       # Code for NeuralODEResidual model
#   elif model == 'ResidualDynamics':
#       print('ResidualDynamics model selected')
#       # Code for ResidualDynamics model
#   else:
#       print('Invalid model selected. Please choose either "NeuralODE", "NeuralODEResidual", or "ResidualDynamics".')

#   if solver == 'dopri5':
#       print('Dormand-Prince 5 solver selected')
#       # Code for dopri5 solver
#   elif solver == 'euler':
#       print('Euler solver selected')
#       # Code for euler solver
#   elif solver == 'dopri8':
#       print('Dormand-Prince 8 solver selected')
#       # Code for dopri8 solver
#   else:
#       print('Invalid solver selected. Please choose either "dopri5", "euler", or "dopri8".')

#     batch_size = 500
#     # Check if args has --training
#     collected_data = np.load(os.path.join(GOOGLE_DRIVE_PATH, 'collected_data_20.npy'), allow_pickle=True)
#     train_loader, val_loader = process_data_single_step(collected_data, batch_size=batch_size)
#     # Else if it is --inference do this:
#     model = NeuralODEModel(4+1,4+1)
#     train_ode(model,train_loader,val_loader,lr=1e-3,num_epochs=25,horizon=4,solver='dopri15')

def plot_losses(loss_train, loss_val):
  plt.plot(loss_train, label='Training Loss', color='blue')
  plt.plot(loss_val, label='Validation Loss', color='orange')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

