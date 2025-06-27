import torch
#from scipy.linalg import expm
import numpy as np



############################################################################################################
#########        Paley ordering Walsh digital bases 
############################################################################################################

def standard_Walsh(N):
    """
    Return standard (N*N) Walsh matrix W_N:
    """
    if N == 1:
        return np.array([[1]])
    else:                                   

        H = standard_Walsh(N // 2)
        return np.block([[H, H],\
                        [H, -H] ])

def paley_Walsh(N):
    """
    Return paley order Walsh matrix W_N_paley [(N*N)]
    """


    W = standard_Walsh(N)
    n = W.shape[0]
    indices = list(range(n))

    # Function to count sign changes in a row
    def sign_changes(row):
        return np.sum(row[:-1] != row[1:])
    
    # Sort indices based on sign changes in the corresponding rows of H
    indices.sort(key=lambda i: sign_changes(W[i]))
    
    return W[indices]

W_8_paley = paley_Walsh(8)  # A proper paley_order Walsh-8   
# print("Walsh-8", W_8_paley) #test pass
# print("Walsh-16",paley_Walsh(16))  #test pass

############################################################################################################
#########        Walsh tranformation on the ctrl params: should be feed to NNs 
############################################################################################################

# Convert Pauli operators to PyTorch tensors
pauli_operators = [torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64, device= 'cuda' ),
                   torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device= 'cuda' ),
                   torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device= 'cuda' ),
                   torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device= 'cuda' )]

class ControlF1Haar_Tensor():
    """
    A class that handles ctrl_to_F1_Haar tranformation
    in Tensor form !!!! 
    -----------------------------------------------
    The C_Tensor consists multi-control sets  
    -----------------------------------------------
    Tensors:
    C_Tensor     [C, L, 3]              input
    get_F1_haar_Layer  [3, C,L, N_]     output

    """
    def __init__(self, T, C_Tensor, C_shape, N_=8, MM=100, MultiTimeMeas=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T = T
        self.C_Tensor = torch.tensor(C_Tensor, device=self.device)       
        self.C = C_Tensor.shape[0]    # 
        self.L = C_Tensor.shape[1]
        self.C_shape = C_shape
        self.tau = self.T / self.L
        self.N_ = N_ 
        self.MM = MM    # time chopping in each window

    def _set_haar_frames(self):
        """
        Return : tensor [N_, MM]
        -------------------------------------
        The **simple**-Haar frames phi_(n)(t_l)    
        """ 
        result = torch.zeros((self.N_, self.MM),  device='cuda')
        for n in range(self.N_):
            walsh_mat = W_8_paley
            result[n, :] = torch.tensor([walsh_mat[n, min(max(int(self.N_ * tdx // self.MM), 0), self.N_ - 1)] \
                                         for tdx in range(self.MM) ])
        return result

    def _angle(self):
        """
        Return ctrl's rot_angle [C,L]
        """ 
        return self.C_Tensor[:, :, 0]       
    def _nx(self):
        """
        Return ctrl's nx tesnor [C,L]
        """ 
        return torch.cos(self.C_Tensor[:, :, 1]) * torch.cos(self.C_Tensor[:, :, 2]) #cos(a)cos(b)
    def _ny(self):
        """
        Return ctrl's nx tesnor [C,L]
        """ 
        return torch.cos(self.C_Tensor[:, :, 1]) * torch.sin(self.C_Tensor[:, :, 2]) #cos(a)sin(b)
    def _nz(self):
        """
        Return ctrl's nx tesnor [C,L]
        """ 
        return torch.sin(self.C_Tensor[:, :, 1])                        #  sin(a) 
    def _U_ctrl_n(self, ):  # 
        """
        return U_ctrl_n tensor  [C,L, 2,2]  //   U_ctrl after window-n FINISHED, (L-1) gives U_ctrl_T
        """
        # All has shape [C,L]
        theta = self._angle()   # shape = [C,L]
        nx = self._nx()         # shape = [C,L]
        ny = self._ny()         # shape = [C,L]
        nz = self._nz()         # shape = [C,L]
        _U_piecewise =  torch.cos(theta).unsqueeze(-1).unsqueeze(-1) * pauli_operators[0]- \
            1j * torch.sin(theta).unsqueeze(-1).unsqueeze(-1) * \
          ( nx.unsqueeze(-1).unsqueeze(-1) * pauli_operators[1]+
            ny.unsqueeze(-1).unsqueeze(-1) * pauli_operators[2]+
            nz.unsqueeze(-1).unsqueeze(-1) * pauli_operators[3] )         # shape [C,L,2,2]
        _U_cumu = _U_piecewise
        for n in range(1,self.L):
            _U_cumu[:, n, :, :] = torch.einsum('cij,cjk->cik',_U_cumu[:, n, :, :],_U_cumu[:, n-1, :, :])
        return _U_cumu #torch.tensor(_U_cumu, device='cuda')   
    
    def _h_t(self):
        """
        Return the h_t tensor [MM]
        h_t is the profile integral of C_shape
        """
        if self.C_shape == "Gaussian":
            profile = lambda t: 0.5 - 0.5 * np.cos(5 - (10 * t) / self.tau)
        elif self.C_shape == "Triangle":
            profile = lambda t: (2 * t**2 / self.tau**2) if t < 0.5 * self.tau \
                                else ((4 * self.tau - 2 * t) * t / self.tau**2)        
        return  torch.tensor([profile(tdx) for tdx in range(self.MM) ], device='cuda')        
    
    def _U_ctrl_continous(self):
        """
        Return: U_ctrl_continous tensor [C, L, MM ,2,2]
        -----------------------------------
        Control propergator: It is indirect way to calculate U_ctrl at NEAR CONTINUOIUS time in [0,T]: discreteize to MM>>1 pieces
        """
        _the_U_ctrl = self._U_ctrl_n()[:, :-1, :,:]   # window from 1st to (L-1)-th 
        _U_finished =torch.cat(( torch.eye(2).repeat(self.C, 1, 1,1).to(device='cuda'), \
                                 _the_U_ctrl), dim=1 ) # catch the finished prev parts// size= [C,L,2,2] \
            # in 0-th window the finished ins null as idenity
        _U_finished = _U_finished.unsqueeze(2).repeat(1,1, self.MM, 1,1)  # size =  [C,L, MM, 2,2]
        _U_finished = torch.tensor(_U_finished, dtype=torch.complex64, device='cuda') 
        
        _h_t_ = self._h_t().unsqueeze(0).unsqueeze(0).repeat(self.C,self.L,1)      # size = [C,L,MM]
        _angle_ = self._angle().unsqueeze(-1).repeat(1,1,self.MM)                  # size = [C,L,MM]
        _nx_ = self._nx().unsqueeze(-1).repeat(1,1,self.MM)                        # size = [C,L,MM]
        _ny_ = self._ny().unsqueeze(-1).repeat(1,1,self.MM)                    
        _nz_ = self._nz().unsqueeze(-1).repeat(1,1,self.MM)                  
        _cos_part = torch.cos(_angle_*_h_t_)                                          # size= [C,L,MM]
        _sin_part = torch.sin(_angle_*_h_t_) 
        _U_nogoing = _cos_part.unsqueeze(-1).unsqueeze(-1) * pauli_operators[0] +\
               -1j *  _sin_part.unsqueeze(-1).unsqueeze(-1) * (\
                _nx_.unsqueeze(-1).unsqueeze(-1) * pauli_operators[1]\
              + _ny_.unsqueeze(-1).unsqueeze(-1) * pauli_operators[2]\
              + _nz_.unsqueeze(-1).unsqueeze(-1) * pauli_operators[3]  )             # size = [C,L,MM,2,2]
        _U_nogoing = torch.tensor(_U_nogoing, dtype=torch.complex64, device='cuda') 
        return torch.einsum('clmij,clmjk->clmik',_U_nogoing, _U_finished)            # U_on @ U_fin size =[C,L,M,2,2]

    def _set_y_u(self):
        """
        Return the y_u tensor [3, C, L, MM]
        ---------------------------------------------------------------
        u:  (int)  1<=u<=3  pauli x,y,z
        y_u(t) = Tr[U0d @ sig_z @ U0 @ sig_u]  ///  As U0 = `U_ctrl_continuous()`,  0<t_l<tau 
        """
        _U0  = self._U_ctrl_continous()             #[C,L,MM,2,2]
        _U0d = torch.conj(_U0).transpose(-2,-1)
        _sig_x = pauli_operators[1].unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(self.C, self.L, self.MM,1,1)
        _sig_y = pauli_operators[2].unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(self.C, self.L, self.MM,1,1)
        _sig_z = pauli_operators[3].unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(self.C, self.L, self.MM,1,1)
        print('check', _U0.shape, _sig_x.shape) 
        return torch.stack(( 
                 torch.einsum('clmij,clmjk,clmks,clmsi->clm',_U0d, _sig_z, _U0, _sig_x)/2, \
                 torch.einsum('clmij,clmjk,clmks,clmsi->clm',_U0d, _sig_z, _U0, _sig_y)/2, \
                 torch.einsum('clmij,clmjk,clmks,clmsi->clm',_U0d, _sig_z, _U0, _sig_z)/2)  ,\
                dim=0)   # size =  [3, C, L, MM]
    
    def get_F1_haar_Layer(self):
        """
        Return the F1 tensor [3, C, L, N_]
        --------------------------------------------------------
        F1_u(l,n)= int^T_0 [y_u(t) phi_{l,n}(t)]dt 
        = int^{l*tau}_{(l-1)tau} [y_u(t) W_n(t)]dt  for u ={x,y,z}
        """
        _y_u = self._set_y_u()                                 # [3, C, L, MM]
        _y_u = _y_u.unsqueeze(-2).repeat(1,1,1,self.N_,1)      # [3, C, L, N_,  MM]         
        _haar= self._set_haar_frames()                         # [N_, MM]
        _haar=_haar.unsqueeze(0).unsqueeze(1).unsqueeze(1).repeat(3,self.C,self.L,1,1) #size= [3, C, L, N_, MM] 
        return self.tau/self.MM * torch.einsum('uclnm,uclnm->ucln', _y_u, _haar)    






if __name__ == '__main__':
    import time
    C =10000
    L =4
    N_ = 8
    T= 1.0* 10**-6
    rnd_C_Tensor = torch.rand(C, L ,3)
    tic = time.time()
    
    test = ControlF1Haar_Tensor(T=T, C_Tensor=rnd_C_Tensor, C_shape="Gaussian", N_=N_)
    #print(test._set_haar_frames(), test._set_haar_frames().shape)
    #print(test._U_ctrl_continous(), test._U_ctrl_continous().shape)
    #print(test._set_y_u(), test._set_y_u().shape)
   # print(test.get_F1_haar_Layer(),test.get_F1_haar_Layer().shape)
    print(test.get_F1_haar_Layer())
    #test.output_Layer()
    toc = time.time()
    print(f"Elapsed time: {toc - tic} seconds")