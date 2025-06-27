"""
This module implements a simulator for a noisy qubit in the rotating frame
the inputs are 
1. noise trajectories
2. control_pulses
3. Evoltion time T and window L  ...

Outputs are
E[O] for all O and for all rho_S
"""

# preample
import numpy as np
from functools import reduce
from itertools import product
from scipy.linalg import expm
from datetime import datetime
import torch

###############################################################################
torch.cuda.empty_cache()
# Convert Pauli operators to PyTorch tensors
pauli_operators = [torch.tensor([[1, 0],   [0, 1]], dtype=torch.complex64, device= 'cuda' ),
                   torch.tensor([[0, 1],   [1, 0]], dtype=torch.complex64, device= 'cuda' ),
                   torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device= 'cuda' ),
                   torch.tensor([[1, 0],   [0, -1]], dtype=torch.complex64, device= 'cuda' )]
np.random.seed(seed = 42)

def RTN_generator(T, gamma, g, MM=1000, K=1000):
    """
    RTN sampling: a zero mean  zero-frequency-center telegraph noise
    Return:  A numpy shape = (K * MM) noise sample, where row->trajec col->time 
    ---------------------------------------------------------------------------------------
    If we know that state is s at time t: z_s(t), then at t+dt, the flip to s' has probablity
	P_flip(t, t+dt) = 1- e^(-gamma dt)
    ---------------------------------------------------------------------------------------
    T         : The total evolution time 
    gamma     : flipping rate of RTN
    g         : coupling
    MM        : Time discretization 
    K         : total noise realization
    """
    trajectory_table = np.zeros((K,MM))     #shape = num_sample * num_time_chop
    for i in range(K):
        trajectory_table[i][0] = 1 if (np.random.uniform(0,1)>0.5) else -1 #  \pm 1 full-random zero-mean
        j=1
        while j<MM:
            trajectory_table[i][j] = 1 * trajectory_table[i][j-1] if ( np.exp(-gamma* T/MM)  > np.random.uniform(0, 1)) \
                else -1* trajectory_table[i][j-1]
            j+=1
    # now add cos modulation 
	#for i in range(K):
	#	phi =  np.random.uniform(0, 1)*2*np.pi
	#	for j in range(MM):
	#		trajectory_table[i][j] = trajectory_table[i][j] * np.cos(Omega * j * dt + phi)
    return g * trajectory_table



class NoisyQubitSimulator():
    """
    Class for simulating a "dephasing" - ONLY noisy  qubit  in rotating frame, but OUTPUT in toggling frame !!!!!!!
    ----------------------------------------------------------
    it used the RTN_generator generated noise trajectories

    Tensors:
    C_params (C, L, 3)      
    RTN      (K, MM) 
    H_tensor (C,K, MM, 2,2)   #ctrl  #noise #slice #sys_dim

    input includes T,L and control pulses
    output includes E[O] for all O and for all rho_S
    """
    def __init__(self, T, C_params, C_shape="Gaussian", MM=1000, K=1000, gamma=10**4, g=2* 10* 10**5, MultiTimeMeas =False):
        """
        T               : Evolution time
        L               : Total num of windows = pulse_number
        tau             : One window time duration
        C_params        : tensor: shape=(C, L, 3).   with 3 = [theta, (alpha, beta)] gives the pulse amplitude & direction
        C_shape         : Waveforms: Gaussian or Triangle
        C               : Total number of controls 
        MM              : Total time pieces in [0,T]
        K               : total noise trajec number
        """
        self.T          = T                 # Evolution time
        self.C_params   = C_params          # A list of L sublists, each=[theta, alpha, beta] gives rot_amplitude, n_x= cos(a)cos(b), n_y = cos(a)sin(b), n_z= sin(a)
        self.C          = C_params.shape[0] # Total num of controls = number of circuit
        self.L          = C_params.shape[1] # Total num of windows = pulse_number
        self.C_shape    = C_shape           # Waveforms: Gaussian or Triangle (a single string)
        self.MM         = MM                # Total time pieces in [0,T]
        self.K          = K                 # total noise trajec number
        self.tau        = T/self.L          # One window time duration
        self.dt         = T/MM              # duration of each small time_piece
        self.gamma      = gamma             # fluctuation rate 
        self.g          = g                 # coupling  strength
        self.trajectory = self._set_noise()
        self.MultiTimeMeas = MultiTimeMeas  # Varing msmt time ?	

    def _set_noise(self):
        """
        Return: tensor (K, MM) of noise_trajectory
        """
        return torch.tensor(RTN_generator(T=self.T, gamma=self.gamma, g= self.g, MM=self.MM, K=self.K) , dtype=torch.complex64, device= 'cuda')
    

    def _set_ctrl_shape(self): 
        """
        set a tensor [MM/L] , only for 1 window,  NORMALIZED waveforms based on C_shape
        """
        if self.C_shape == "Gaussian":
            h_t = lambda t: (5.64189583548624/self.tau)* np.exp( -t**2/(0.1*self.tau)**2)  *(t<0.5*self.tau)*(t>-0.5*self.tau)      # Gaussian symmetric & normalized  waveform : the bandwith = T/L/10
        elif  self.C_shape == "Triangle":
            h_t = lambda t: (4/self.tau**2) *((t<=0)*t + (t>0)*(-t) + self.tau/2)  *(t<0.5*self.tau)*(t>-0.5*self.tau)          # triangle symmetric & normalized  waveform    
        ts = np.linspace(-0.5*self.tau,0.5*self.tau, int(self.MM/self.L), endpoint = False)
        return torch.tensor(h_t(ts+0.5*self.dt), dtype=torch.complex64, device= 'cuda')     # Discrete time_step sampling in ONE window

    def _angle(self):
        """
        Return ctrl's rot_angle [C,L]
        """ 
        return self.C_params[:, :, 0]       
    def _nx(self):
        """
        Return ctrl's nx tesnor [C,L]
        """ 
        return torch.cos(self.C_params[:, :, 1]) * torch.cos(self.C_params[:, :, 2]) #cos(a)cos(b)
    def _ny(self):
        """
        Return ctrl's nx tesnor [C,L]
        """ 
        return torch.cos(self.C_params[:, :, 1]) * torch.sin(self.C_params[:, :, 2]) #cos(a)sin(b)
    def _nz(self):
        """
        Return ctrl's nx tesnor [C,L]
        """ 
        return torch.sin(self.C_params[:, :, 1])                        #  sin(a) 

    def _U_ctrl_n(self, ):  # 
        """
        return U_ctrl_n tensor  [C,L ,2,2]  //   U_ctrl after window-n FINISHED, (L-1) gives U_ctrl_T
        """
        # All has shape [C,L]
        theta = self._angle() 
        nx = self._nx()
        ny = self._ny()
        nz = self._nz()
        _U_piecewise =  torch.cos(theta).unsqueeze(-1).unsqueeze(-1) * pauli_operators[0]- \
            1j * torch.sin(theta).unsqueeze(-1).unsqueeze(-1) * \
          ( nx.unsqueeze(-1).unsqueeze(-1) * pauli_operators[1]+
            ny.unsqueeze(-1).unsqueeze(-1) * pauli_operators[2]+
            nz.unsqueeze(-1).unsqueeze(-1) * pauli_operators[3] )         # shape [C,L,2,2]
        _U_cumu = _U_piecewise
        for n in range(1,self.L):
            _U_cumu[:, n, :, :] = torch.einsum('cij,cjk->cik',_U_cumu[:, n, :, :],_U_cumu[:, n-1, :, :])
        return  _U_cumu

    def _set_control_hamiltonian(self):
        """
        set control_hamiltonian for circuit
        Return H-ctrl tensor [C, MM, 2, 2] 
        """
        # All has shape [C,L]
        theta = self._angle()
        nx = self._nx()
        ny = self._ny()
        nz = self._nz()
        h_Win_global = theta.unsqueeze(-1).unsqueeze(-1)* \
             ( nx.unsqueeze(-1).unsqueeze(-1) * pauli_operators[1]+
               ny.unsqueeze(-1).unsqueeze(-1) * pauli_operators[2]+
               nz.unsqueeze(-1).unsqueeze(-1) * pauli_operators[3] )  # shape [C,L,2,2]
        h_win_local = self._set_ctrl_shape()                          # shape [MM/L]
        # Next have full control Hamiltonian:
        # H_Win_global.size=[C,L]     H_win_local.shape=[MM/L]  
        h_control = torch.einsum('clij,m->clmij', h_Win_global,h_win_local).reshape(h_Win_global.shape[0],h_Win_global.shape[1]*h_win_local.shape[0], 2,2)  
        return h_control       
    
    def _set_noise_hamtiltonian(self):
        """
        set the H_sb hamiltonian
        Return H_sb tensor [K,MM, 2,2 ]
        """
        return self.trajectory.unsqueeze(-1).unsqueeze(-1)*pauli_operators[3]

    def _set_total_Hamiltonian(self):
        """
        set the H_total of the SB total system
        Return: H_toal tensor [C, K, MM, 2,2 ]
        """
        h_ctrl = self._set_control_hamiltonian()               # size =[C,MM,2,2]
        h_ctrl = h_ctrl.unsqueeze(1).repeat(1,self.K, 1,1,1)   # size =[C,K,MM,2,2]
        h_sb  = self._set_noise_hamtiltonian()                 # size = [K,MM, 2,2]
        h_sb = h_sb.unsqueeze(0).repeat(self.C, 1, 1,1,1)       # size = [C,K,MM, 2,2]
        return h_ctrl+h_sb 

    def _set_evolution(self): # 👌
        """
        evolute the total_Hamiltonian (on MM) and get keep over ALL noise (on K)
        Return: tensor size [K,C,2,2]  
        """
        total_hamiltonians = self._set_total_Hamiltonian()                  #szie =  [C,K,MM, 2,2]
        total_hamiltonians = torch.permute(total_hamiltonians,(2,1,0,3,4))  #size = [MM,K,C,2,2]
        #print(datetime.now().strftime("%H:%M:%S"))
        if self.MultiTimeMeas == False:
            """
            only measure at t= T:
            """
            #Step 1: Calculate the matrix exponential for each (2, 2) matrix in H
            total_hamiltonians = total_hamiltonians.contiguous().to(torch.complex64).to(device='cuda')
            H_exp = torch.matrix_exp(-1j*self.dt*total_hamiltonians)        # size = [MM,K,C,2,2]

            #Step 2: Compute the cumulative matrix product along the first axis (M)
            cmulative_product = H_exp[0, :, :, :, :] 
            for m in range(1,self.MM):
                cmulative_product = torch.einsum('kcij,kcjl->kcil' , H_exp[m, :, :, :, :], cmulative_product)
                #cumulative_product[: , : , :, : ] = torch.einsum(,H_exp[m] )

            U_total = cmulative_product.view(self.K, self.C, 2, 2)
            
        else:
            """
            need construction
            """
            U_total = None
        return U_total   # torch.Size([K, C, 2, 2])


    def _set_evolution__old(self): # 🔴 Buggy & Useless
        """
        evolute the total_Hamiltonian (on MM) and get keep over ALL noise (on K)
        Return: tensor size [K,C,2,2]  
        """
        total_hamiltonians = self._set_total_Hamiltonian()                  #szie =  [C,K,MM, 2,2]
        total_hamiltonians = torch.permute(total_hamiltonians,(2,1,0,3,4))  #size = [MM,K,C,2,2]
        #print(datetime.now().strftime("%H:%M:%S"))
        if self.MultiTimeMeas == False:
            """
            only measure at t= T:
            """
            #Step 1: Calculate the matrix exponential for each (2, 2) matrix in H
            total_hamiltonians = total_hamiltonians.contiguous().to(torch.complex64).to(device='cuda')
            H_exp = torch.matrix_exp(-1j*self.dt*total_hamiltonians)        # size = [MM,K,C,2,2]

            #Step 2: Compute the cumulative matrix product along the first axis (M)
            #We'll use torch.cumprod on a reshaped tensor and then batch matrix multiplication
            #Reshape the tensor so that we can perform the multiplication along a new batch dimension
            H_exp_reshaped = H_exp.permute(1, 2, 0, 3, 4).reshape(self.K * self.C, self.MM, 2, 2)
            #Initialize the output tensor with the first matrix in the sequence
            cumulative_product = H_exp_reshaped[:, 0]
            #Perform the cumulative matrix multiplication
            for i in range(1, self.MM):
                cumulative_product = cumulative_product @ H_exp_reshaped[:, i]
            #Reshape the result to match the desired output shape
            U_total = cumulative_product.view(self.K, self.C, 2, 2)
            
            #evolve = lambda U,U_j: U_j @ U   # define a lambda function for calculating the propagator
            #U_total = reduce(evolve, torch.matrix_exp(-1j*self.dt* total_hamiltonians))
            #for mm in range(MM):    # propagate along time
        else:
            """
            need construction
            """
            U_total = None
        return U_total   # torch.Size([K, C, 2, 2])

    def circuit_implement(self):
        """
        Return tensor --[C, 3, 6]
        Output:  <O>|_S for all O and all rho_S in "Toggling" frame 
        ---------------------------------
        Tensors: 
                U_all [K, C, 2, 2]
                msmt_O [3, C,2,2]
                msmt_S [6, 2,2 ]
        ---------------------------------
        Notice: since the evolution is in rotating frame and the ML_module will be working fror toggling frame dynamics [perturbation theory],
        the Output/readout here should be in T-frame. 
        we want tilde{O} = U_0^+ @ O @ U_0  = pauli
        thus the R-frame O = U_0 @ pauli @ U_0^+ to make sure T-frame's O properly cycle over pauli         
        """
        if self.MultiTimeMeas == False:
            """
            single time
            """
            U_ctrl_T = self._U_ctrl_n()[: , -1, :, :].view(self.C, 2,2) 
            msmt_O =  [U_ctrl_T @ O @ (torch.conj(U_ctrl_T).transpose(-2,-1)) for O in  pauli_operators[1:]]            # All R-frame observables to make T-frame tilde{O} = pauli
            msmt_S = [(pauli_operators[0]+pauli_operators[1])/2, (pauli_operators[0]-pauli_operators[1])/2, \
                      (pauli_operators[0]+pauli_operators[2])/2, (pauli_operators[0]-pauli_operators[2])/2,\
                      (pauli_operators[0]+pauli_operators[3])/2, (pauli_operators[0]-pauli_operators[3])/2 ]            # All initial states
            U_all = self._set_evolution()                               # size= [K, C,2,2]
            U_all_dagger = torch.conj(U_all).transpose(-2,-1)           # size =[K, C,2,2]
            results = torch.zeros((self.C, len(msmt_O), len(msmt_S)))
            for idx_O, O in enumerate(msmt_O):            
                for idx_S, S in enumerate(msmt_S):
                    # below: 
                    #print('debug',U_all.shape, S.shape, U_all_dagger.shape, O.shape)
                    results[:, idx_O,idx_S] = torch.mean(torch.einsum('kcij,jl,kclm,cmi->kc', U_all, S, U_all_dagger, O).real, dim=0 )  # calculate E[O(T)]_rhoS in Rotating frame
                    #results[:, idx_O,idx_S] = torch.mean(torch.einsum('kcij,ij,klmn,cmn->kc', U_all, S, U_all_dagger, O).\
                    #                                     diagonal(offset=0, dim1=-2, dim2=-1).sum(-1).real,   dim=0 )  # calculate E[O(T)]_rhoS in Rotating frame
            # the results are rotating frame simultion , yet it corresponds to toggling frame results with stantard \tidle{O} = pauli
            # print(np.allclose(results, results1))
        else:
            """
            need construction
            """
            results =None
        return results



if __name__ == '__main__':
    import time
    tic = time.time()

    torch.cuda.empty_cache()

    #################################################
    # generate noise trajectories
    ###################################################
    input_parameters = {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**4,  "g": 2* 10* 10**5}
    T =10**-6
    MM = 1000
    K  = 1024

    gamma=10**4
    g=2* 10* 10**5 

    #C_params = torch.from_numpy(np.array([[[0.6*np.pi, 0.8*np.pi, -0.5*np.pi],\
    #                                       [1.4*np.pi, 2.30*np.pi, 0.7*np.pi],\
    #                                       [0.6*np.pi, -0.8*np.pi, -2.5*np.pi],\
    #                                       [1.14*np.pi, 1.130*np.pi, 0.17*np.pi] ]])\
    #                                        ).to(torch.complex64).to(device='cuda')
    C_params = torch.rand(40,4,3, device='cuda')   # C=40 GPU memory = 16G, time =13s // more C out of memory

   
    tic = time.time()

    #################################################
    # generate simulation data of noisy qubits
    ###################################################

    noisy_qubit = NoisyQubitSimulator(T=input_parameters["T"], 
                                      C_params=C_params,
                                      C_shape="Gaussian", 
                                      MM= input_parameters["MM"], 
                                      K= input_parameters["K"],
                                      gamma= input_parameters["gamma"], 
                                      g= input_parameters["g"], 
                                      MultiTimeMeas= False)
    
    #print(noisy_qubit._angle().shape, noisy_qubit._nx().shape, noisy_qubit._ny().shape, noisy_qubit._nz().shape)
    #print(noisy_qubit._U_ctrl_n().shape)
    #print(noisy_qubit._U_ctrl_n())
    #print(noisy_qubit._set_ctrl_shape().shape)
    #print(noisy_qubit._set_control_hamiltonian())
    #print(noisy_qubit._set_control_hamiltonian().shape,'\n')
    #print(noisy_qubit._set_noise_hamtiltonian().shape,'\n')
    #print(noisy_qubit._set_noise_hamtiltonian())
    #print('total_H', noisy_qubit._set_total_Hamiltonian().shape,'\n')
    #print('evolution_', noisy_qubit._set_evolution().shape)
    #print(noisy_qubit._set_evolution(), noisy_qubit._set_evolution().shape,'\n')
    #print(noisy_qubit.circuit_implement(), noisy_qubit.circuit_implement().shape)

    result = noisy_qubit.circuit_implement()

    print('result size',result.shape,'\n result =', result)

    print(f"Max memo usage: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    toc = time.time()
    print(f"Elapsed time:{toc-tic} seconds")
