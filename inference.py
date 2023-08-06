from NeuralODE import *

T = 10
t = torch.arange(0, T, 1)

# # SINE-WAVE
# # Define the parameters for the sine wave
frequency = 1.0
amplitude = 20.0

# # Generate the sine wave
# t = torch.arange(0, T, 1)
sine_wave = amplitude * torch.sin(2 * np.pi * frequency * t / T)

# # Scale the output values to range between -20 and +20
control_sequence = (sine_wave.clamp(min=-20.0, max=20.0)).unsqueeze(1)
# control_sequence = action_list_dataset[0][:T]
plt.plot(np.arange(T), control_sequence.numpy())
plt.xlabel('Time')
plt.ylabel('Control sequence')
plt.show()

model_path = "models/NeuralODE_hor-7_lr_1e-3_epochs-50_dopri5.pt"
model_type = "NeuralODE"
solver = "dopri5"
states_pybullet,model_pred = get_inference(model_path,model_type,control_sequence,T=T,solver=solver) 

x_error = np.mean(np.square(states_pybullet[:,0] - model_pred[:,0]))*100
x_dot_error = np.mean(np.square(states_pybullet[:,2] - model_pred[:,2]))*100
theta_error = np.mean(np.square(states_pybullet[:,1] - model_pred[:,1]))
theta_dot_error = np.mean(np.square(states_pybullet[:,3] - model_pred[:,3]))
print("Total loss: ", x_error+x_dot_error+theta_error+theta_dot_error)
print("Mean Squared Error (x): ", x_error)
print("Mean Squared Error (x_dot): ", x_dot_error)
print("Mean Squared Error (theta): ", theta_error)
print("Mean Squared Error (theta_dot): ", theta_dot_error)
