from NeuralODE import *

batch_size = 500
collected_data = np.load('data/collected_data_20.npy', allow_pickle=True)
train_loader, val_loader= process_data_single_step(collected_data, batch_size=batch_size)

lr=1e-3
horizon=7
solver='dopri5'
num_epochs = 50
time_interval = 0.1
model_type="NeuralODE"

## TRAIN MODEL ##
trained_model, train_loss, val_loss = train(train_loader, val_loader, num_epochs=num_epochs, lr=lr, horizon=horizon, optim="Adam", time_interval=time_interval, solver=solver, model_type=model_type)

## SAVE MODEL ##
name = str(model_type)+"_" +solver+"_hor-"+str(horizon)+"_"+str(num_epochs)+".pt"
model_save_path = os.path.join("trained_models/", name)
print("Model saved as "+ name)
torch.save(trained_model.state_dict(), model_save_path)

## PLOT INFERENCE ##

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
