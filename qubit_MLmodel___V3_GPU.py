"""
author: wenzheng dong
----------------------
This module implements a torch-based D-NN to learn the ctrl_env convolution 

"""

# Preamble
import math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import zipfile    
import os
from os.path import abspath, dirname, join
import pickle
from math import prod
import scipy.optimize as opt
from scipy.linalg import hadamard
from scipy.linalg import expm
import scipy.integrate as integrate
#import scipy.special.erf as erf
from joblib import Parallel, delayed
# from numba import njit
import time

from torch.utils.data import TensorDataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter

from qubit_simulator__GPU import pauli_operators


n_cores = 6
my_device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {my_device}")
torch.cuda.empty_cache()

# Initialize TensorBoard writer
#writer = SummaryWriter('runs/qubit_ML_experiment')

############################################################################################################
#########        build NN model 
############################################################################################################


class MyNN(nn.Module):
    """
    define simple NN model: there are '4' hidden layers
    """
    def __init__(self, nInput=12, nHidden=24, nOutput=18) -> None:
        super().__init__()
        self.layers   = nn.Sequential(
            nn.Linear(nInput, nHidden),         # 1st hidden layer
            nn.ReLU(),
            # nn.Dropout(p=0.3),                  #💥 Add dropout for regularization
            nn.Linear(nHidden,nHidden),         # 2nd hidden layer
            # nn.BatchNorm1d(nHidden),            #💥 Add batch normalization
            nn.ReLU(), 
            nn.Linear(nHidden,nHidden),         # 3rd hidden layer    
            nn.ReLU(),
            nn.Linear(nHidden, nHidden),        # 4th hidden layer    
            nn.ReLU(),
            nn.Linear(nHidden, nHidden),        # 5th hidden layer    
            nn.ReLU(),
            nn.Linear(nHidden, nOutput),        # Output layer
            #nn.Tanh(),                          # Output layer : in range (-1, 1)  
            )
    def my_trig_layer(self,x):
        """
        apply the trignometric function to the input ctrl_params
        """
        return torch.cat((x, torch.sin(x), torch.cos(x), torch.sin(2*x), torch.cos(2*x)), dim=1 )    
    def forward(self,x):
        return (self.layers((x)))


 
class QubitNeuralNetwork(MyNN):
    """
    A class that train the NN model:
    Input: the 'inputs' & targets (it will be 'tensor'-preprocessed)
    Output:
    """
    def __init__(self, inputs, targets, nHidden, dataType = 'standard') -> None:
        super().__init__()
        self.dataType = dataType     
        self.inputs     = self._input_normalize(inputs)                                          # Input data
        self.targets    = targets.to(my_device)                                                  # Output data
        self.num_sample = inputs.shape[0]                                                        # numer of data
        self.nInputs    = math.prod(inputs.shape[1:])                                            # nInput based on input data
        self.nOutputs   = math.prod(targets.shape[1:])                                           # nOutput based on outout data
        self.model      = MyNN(nInput  = self.nInputs, 
                               nHidden = nHidden, 
                               nOutput = self.nOutputs).to(my_device)                            # Construct a NN model
        self.opt        = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)  # set optimizer of the NN model
        self.criterion  = nn.MSELoss()                                                           # set loss_function of the NN model
        self.losses_training   = []
        self.losses_testing    = []                                                                      
        self.Epochs     = 1000                                                                   #   
        self.batch_size = 64 
        self.split      = int(0.75 * self.num_sample)
        (self.train_dl , self.testn_dl)   = self._load_data_() 
                                                               
    
    def _input_normalize(self, inputs):
        """
        input data norm: X-pulses, only 0-th slice is non-0 ! 
        """
        if self.dataType == 'standard':
            [C,L,u] =  inputs.shape
            if not(torch.all(inputs[:, :, 1:] == 0)):
                raise Exception('Ctrl are not X-pulses')
            mean = inputs[:, :, 0].mean(dim=0, keepdim=True)  # Compute the mean along the sample axis
            std =  inputs[:, :, 0].std(dim=0, keepdim=True)    # Compute the std along the sample axis
            # Normalize only C[:, 0, :] using broadcasting
            inputs[:, :, 0] = (inputs[:, :, 0] - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero
        elif self.dataType == 'Haar':
            mean = inputs.mean(dim=0, keepdim=True)  # Compute the mean along the sample axis
            std =  inputs.std( dim=0, keepdim=True)  # Compute the std along the sample axis
            inputs = (inputs-mean)/ (std + 1e-8) # Add small epsilon to avoid division by zero
        else:
            raise Exception('no beyond standard or Haar')
        return inputs.to(my_device) 
    
    def _load_data_(self):
        """
        load the data for training and testing
        """
        # training data:
        __train_ds      = TensorDataset(torch.reshape(self.inputs[0:self.split] , (-1,self.nInputs)), \
                                        torch.reshape(self.targets[0:self.split], (-1,self.nOutputs)))
        self.train_dl   = DataLoader(__train_ds, self.batch_size, shuffle=False)
        # testing data:                                              
        __testn_ds      = TensorDataset(torch.reshape(self.inputs[self.split:] ,  (-1,self.nInputs)), \
                                        torch.reshape(self.targets[self.split:],  (-1,self.nOutputs)))             
        self.testn_dl   = DataLoader(__testn_ds, self.batch_size, shuffle=False) 
        self.testn_dlx   = torch.reshape(self.inputs[self.split:] ,  (-1,self.nInputs))               # <--------------
        self.testn_dly   = torch.reshape(self.targets[self.split:],  (-1,self.nOutputs))              # <--------------

        return (self.train_dl, self.testn_dl)

    def get_data_shape(self):
        """
        get the shape of data
        """
        print(f"Input data has #_batch = {self.num_sample}; #_input_feature = {self.nInputs}. Output data has #_output_feature = {self.nOutputs}.") 

    def train(self):
        """
        training the neural network
        """
        self.model = self.model.to(my_device)
        for epoch in range(self.Epochs):
            self.model.train()  # 设为训练模式   # <--------------
            
            running_loss = 0.0                  # <--------------
            for x, y in self.train_dl:
                #print(x.size, y.size)
                #print(x,y)
                x , y  = x.to(my_device) , y.to(my_device)
                self.opt.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.opt.step()
                running_loss += loss.item()  # Accumulate the loss             # <--------------
            if (epoch + 1) % 200 == 0:
                print('Loss after epoch %5d: %.3f' %(epoch + 1, loss))    
            self.losses_training.append(running_loss / len(self.train_dl))     # <--------------  

            self.model.eval()   # 设为评估模式                                  # <--------------
            with torch.inference_mode():
                test_pred = self.model(self.testn_dlx)
                test_loss = self.criterion(test_pred, self.testn_dly)
                self.losses_testing.append(test_loss.item())   
      
            # Log to TensorBoard
        # writer.add_scalar('Training Loss', loss.item(), epoch)             
        return None

    def save_train_model(self, save_directory, filename='optimizer_G_L'):
        """
        Function to save the trained model
        """
        torch.save(self.opt.state_dict(), join(save_directory,f"{filename}={self.inputs.shape[1]}.pt"))
        return None  
    
    def save_losses(self, save_directory, filename='losses_result_G_L'):
        # Create a dictionary containing both losses
        losses_dict = {
            'losses_testing': torch.tensor(self.losses_testing),
            'losses_training': torch.tensor(self.losses_training) }
        torch.save(losses_dict, join(save_directory, f"{filename}={self.inputs.shape[1]}.pt"))
        return None

    def test(self,save_directory, load_opt_name='optimizer_G_L'):
        """
        load the saved optimizer from training for testing
        """
        try: # load the optimizer pt file
            self.opt.load_state_dict(torch.load(join(save_directory,f"{load_opt_name}={self.inputs.shape[1]}.pt"), weights_only=True))                       
        except FileNotFoundError:
            print(f"Optimizer state file '{load_opt_name}' not found.")
            return  

        # for x,y in self.testn_dl:
        #     x , y  = x.to(my_device) , y.to(my_device)
        #     pred = self.model(x)
        #     loss = self.criterion(pred, y)
        #     self.losses_testing.append(loss.item())        # log the loss on testing set
        
        # writer.close() # Close the TensorBoard writer

        return self.losses_testing 
    
    def _load_trained_model(self):
        """
        """
        pass 


    def optimize_ctrl(self, load_opt_name='optimizer_G_L'):
        """
        Fix the NN's weights; optimize the input ctrl_params to 
        obtain the P* that gives best fidelity
        _________________________________________
        """
        try: # load the optimizer pt file
            self.opt.load_state_dict(torch.load(join(dirname(abspath(__file__)),\
                                f"filename={self.inputs.shape[1]}.pt")))                       
        except FileNotFoundError:
            print(f"Optimizer state file '{load_opt_name}' not found.")
            return  

        def infidelity(ctl_params):
            """
            ctrl_params are aray with dim=(L,3)
            """
            # sum(Parallel(n_jobs=n_cores, verbose=0)(delayed(devi_sigma_O_T)(o,o,qubit_params,bath_params)  for  o in  _O_ ))
            return self.model(np.array(ctl_params ,dtype='float32').flatten())

        _opt_ = opt.minimize(fun = infidelity,
        					 x0= [np.pi, 0,0] * self.inputs.shape[1] ,
        					 method ='Nelder-Mead',
        					 options={'maxiter': 1000}
        					 )
        return  _opt_.x


    def optimize_ctrl_GPU(self, target_U = torch.eye(2) ,load_opt_name='optimizer_G_L', num_steps=1000):
        """
        Fix the NN's weights; optimize the input ctrl_params to 
        obtain the P* that gives best fidelity
        _________________________________________
        Method:
            use the NN instead of Nelder-Mead to optimized
        """
        try: # load the optimizer pt file
            self.opt.load_state_dict(torch.load(join(dirname(abspath(__file__)),\
                                f"filename={self.inputs.shape[1]}.pt")))                       
        except FileNotFoundError:
            print(f"Optimizer state file '{load_opt_name}' not found.")
            return  
        def _tomo_target(target_U):  #obtain the 3*6-18 tomo from target_U
            msmt_O =  [target_U @ O @ (torch.conj(target_U).transpose(-2,-1)) for O in  pauli_operators[1:]]            # All R-frame observables to make T-frame tilde{O} = pauli
            msmt_S = [(pauli_operators[0]+pauli_operators[1])/2, (pauli_operators[0]-pauli_operators[1])/2, \
                      (pauli_operators[0]+pauli_operators[2])/2, (pauli_operators[0]-pauli_operators[2])/2,\
                      (pauli_operators[0]+pauli_operators[3])/2, (pauli_operators[0]-pauli_operators[3])/2 ]
            tomo_target = torch.zeros((len(msmt_O), len(msmt_S)))
            for idx_O, O in enumerate(msmt_O):            
                for idx_S, S in enumerate(msmt_S):
                    # below: 
                    tomo_target[idx_O,idx_S] = torch.mean(torch.einsum('ij,jl,lm,mi-', target_U, S, target_U, O).real, dim=0 )  # calculate E[O(T)]_rhoS in Rotating frame
            # the results are rotating frame simultion , yet it corresponds to toggling frame results with stantard \tidle{O} = pauli
            return tomo_target
        tomo_target = _tomo_target(target_U)

        # Create control parameters that need to be optimized
        ctrl_params = torch.randn((1, self.nInputs), requires_grad=True, device=my_device)  

        # Define the optimizer for the control parameters (weights are fixed)
        optimizer = torch.optim.Adam([ctrl_params], lr=1e-4, weight_decay=1e-5)

        # Define the loss function (e.g., negative fidelity or mean squared error)
        # Placeholder fidelity loss function; should be replaced with the actual fidelity calculation
        def infidelity_loss(predictions, targets):
            return torch.mean((predictions - targets) ** 2)  # tomo distance

        # Target data (from target_U)
        target_data = torch.randn((1, 18), device=my_device)  

        # Optimization loop
        for step in range(num_steps):
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass: get model predictions using the current ctrl_params
            predictions = self.model(ctrl_params)

            # Compute the loss (negative fidelity)
            loss = infidelity_loss(predictions, tomo_target)

            # Backward pass: compute gradients and optimize ctrl_params
            loss.backward()
            optimizer.step()

            if step % 100 == 0:  # Print every 100 steps for visibility
                print(f"Step {step}/{num_steps}, Loss: {loss.item()}")

        # Return the optimized control parameters
        return ctrl_params



############################################################################################################
#########       🔴 🔴 🔴 🔴  training the NN  🔴 🔴 🔴 🔴
############################################################################################################


if __name__ == '__main__':

    #################################################################
    ## Haar input data
    #################################################################


    # for i in range(1,18):
    #     tic = time.time()
    #     print(f"Now work on data set {i}:")

    #     C_Haar_All = torch.load(f"simulation_data/set_{i}/ctrl_to_Haar.pt",weights_only=True)
    #     inputs = C_Haar_All.to(torch.float32)
    #     targets= torch.load(f"simulation_data/set_{i}/results.pt",weights_only=True)
    #     save_directory = os.path.dirname(f"simulation_data/set_{i}/ctrl_to_Haar.pt")

    #     nHaar_Hid = 240
    # ################ [1] Gaussian control ####################
    #     print('start training Gaussian ctrl:')
    #     qubit_instance = QubitNeuralNetwork(inputs, targets, nHidden= nHaar_Hid, dataType = 'Haar')      # Instantiate the class
    #     qubit_instance.get_data_shape()
    #     qubit_instance.train()                                                 # training
    #     qubit_instance.save_train_model(save_directory=save_directory, 
    #                                    filename="optimizer_G_Haar")              # save trained model
    #     qubit_instance.test(save_directory=save_directory, \
    #                        load_opt_name="optimizer_G_Haar")                     # test
    #     qubit_instance.save_losses(save_directory=save_directory,
    #                               filename='losses_result_G_Haar')               # save train&test loss
        
    #     print(f"Train_Loss= {np.average(np.array(qubit_instance.losses_training))}.\n  \
    #             Test_Loss= {np.average(np.array(qubit_instance.losses_testing))}")
        
    #     toc = time.time()
    #     print(f"Elapsed time:{toc-tic} seconds \n \n")




    #################################################################
    ## Conventional input data
    #################################################################
    # for i in range(1,18):
    for i in range():
        tic = time.time()
        print(f"Now work on data set {i}:")

        C_params_ALL = torch.load(f"simulation_data/set_{i}/C_params_ALL.pt",weights_only=True)
        inputs = C_params_ALL
        targets= torch.load(f"simulation_data/set_{i}/results.pt",weights_only=True)
        save_directory = os.path.dirname(f"simulation_data/set_{i}/C_params_ALL.pt")

        nParm_Hid = 30

    ################ [1] Gaussian control ####################
        print('start training Gaussian ctrl:')
        qubit_instance = QubitNeuralNetwork(inputs, targets, nHidden=nParm_Hid,dataType = 'standard')                   # Instantiate the class
        qubit_instance.get_data_shape()
        qubit_instance.train()                                                 # training
        qubit_instance.save_train_model(save_directory=save_directory, 
                                       filename="optimizer_G_Params")              # save trained model
        qubit_instance.test(save_directory=save_directory, \
                           load_opt_name="optimizer_G_Params")                     # test
        qubit_instance.save_losses(save_directory=save_directory,
                                  filename='losses_result_G_Params')               # save train&test loss
        
        print(f"Train_Loss= {np.average(np.array(qubit_instance.losses_training))}.\n  \
                Test_Loss= {np.average(np.array(qubit_instance.losses_testing))}")
        
        toc = time.time()
        print(f"Elapsed time:{toc-tic} seconds \n \n")


    