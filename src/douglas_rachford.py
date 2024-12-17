import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
This file is a Python adaptation of O'Connor's matlab implementation in 
https://github.com/danielvoconnor/TV_deblur_spaceVariantKernel/blob/master/tvDeblur_varBlur_freeBCs_DR.m
All rights reserved
"""

class DRSolver:
    """
    This class implements non-blind Deconvolution on a grayscale blurry and noisy 
    image with a Total Variation Regularizer.

    We use the Douglas-Rachford algorithm to iteratively minimize the reconstruction
    loss while enforcing sparse gradients.
    """
    def __init__(self, b, kernels, weights, lmda, beta, t, over_relax, log=False, img_og=None):
        self.b = b
        self.img_og = img_og
        self.weights = weights
        self.lmda = lmda
        self.beta = beta
        self.t = t
        self.over_relax = over_relax
        self.setup(b, kernels)
        self.log = log

    def setup(self, b, kernels):
        self.num_rows, self.num_cols, self.num_kernels = self.weights.shape
        self.k_size = (kernels.shape[0] - 1) // 2
        
        self.kernels = self.beta * kernels
        self.b = self.beta * b
        self.lmda = self.beta * self.lmda
        
        """Setup gradient operators and their eigenvalues"""
        # Create derivative kernels
        Dy = self.beta * np.array([[-1], [1]])
        Dx = self.beta * np.array([[-1, 1]])
        
        # Get eigenvalues for FFT-based operations
        self.eig_val_Dy = self.get_eig_val_arr_for_cyclic_conv(Dy, self.num_rows + 2*self.k_size, self.num_cols + 2*self.k_size)
        self.eig_val_Dx = self.get_eig_val_arr_for_cyclic_conv(Dx, self.num_rows + 2*self.k_size, self.num_cols + 2*self.k_size)
        
        self.eig_val_Dy_trans = np.conj(self.eig_val_Dy)
        self.eig_val_Dx_trans = np.conj(self.eig_val_Dx)


        """Setup convolution operators for each kernel"""
        shape = (self.num_rows + 2*self.k_size, self.num_cols + 2*self.k_size, self.num_kernels)
        self.eig_val_arrs_Kp = np.zeros(shape, dtype=complex)
        self.eig_val_arrs_Kp_trans = np.zeros(shape, dtype=complex)
        
        for k in range(self.num_kernels):
            eig_val = self.get_eig_val_arr_for_cyclic_conv(
                self.kernels[:,:,k],
                self.num_rows + 2*self.k_size,
                self.num_cols + 2*self.k_size
            )
            self.eig_val_arrs_Kp[:,:,k] = eig_val
            self.eig_val_arrs_Kp_trans[:,:,k] = np.conj(eig_val)

        """Setup weight matrices"""
        self.Lambda = 1.0 / np.sum(self.weights**2, axis=2)
        self.v = np.zeros_like(self.weights)
        for k in range(self.num_kernels):
            self.v[:,:,k] = self.weights[:,:,k] * self.Lambda
        self.v_norm_squared = np.sum(self.v**2, axis=2)

        """Setup matrix for linear system solution"""
        t = self.t
        
        # I + t^2 A^T A
        self.eig_vals_mtrx = np.zeros((self.num_rows + 2*self.k_size, self.num_cols + 2*self.k_size), dtype=complex)
        
        # kernel terms
        for k in range(self.num_kernels):
            self.eig_vals_mtrx += (self.eig_val_arrs_Kp_trans[:,:,k] * self.eig_val_arrs_Kp[:,:,k])
        
        # gradient terms
        self.eig_vals_mtrx += (self.eig_val_Dy_trans * self.eig_val_Dy + self.eig_val_Dx_trans * self.eig_val_Dx)
        self.eig_vals_mtrx = (t**2) * self.eig_vals_mtrx + 1
        
        # Initialize primal and dual variables
        shape_x = (self.num_rows + 2*self.k_size, self.num_cols + 2*self.k_size)
        shape_z1 = (self.num_rows + 2*self.k_size, self.num_cols + 2*self.k_size, self.num_kernels)
        shape_z2 = (self.num_rows + 2*self.k_size, self.num_cols + 2*self.k_size, 2)  # For D1, D2
        
        self.z_x = np.zeros(shape_x)
        self.z_x[self.k_size:self.k_size + self.num_rows, 
                self.k_size:self.k_size + self.num_cols] = self.b / self.beta
        
        self.z_z1 = np.zeros(shape_z1)
        self.z_z2 = np.zeros(shape_z2)

    @staticmethod
    def get_eig_val_arr_for_cyclic_conv(kernel, num_rows, num_cols):
        padded_kernel = np.zeros((num_rows, num_cols))
        kh, kw = kernel.shape
        padded_kernel[:kh, :kw] = kernel
        return fft2(np.roll(np.roll(padded_kernel, -kh//2, axis=0), -kw//2, axis=1))

    def apply_cyclic_conv_2d(self, x, eig_val_arr):
        return np.real(ifft2(fft2(x) * eig_val_arr))

    def grad(self, x):
        # gradient with FFT
        Dy = self.apply_cyclic_conv_2d(x, self.eig_val_Dy)
        Dx = self.apply_cyclic_conv_2d(x, self.eig_val_Dx)
        return np.stack([Dy, Dx], axis=2)

    def grad_T(self, y):
        # transpose gradient with FFT
        return (self.apply_cyclic_conv_2d(y[:,:,0], self.eig_val_Dy_trans) + 
                self.apply_cyclic_conv_2d(y[:,:,1], self.eig_val_Dx_trans))

    def prox_g1_star(self, z_z1):
        t = self.t
        y_hat = z_z1 / t
        
        U_y_hat = np.zeros((self.num_rows, self.num_cols))
        for k in range(self.num_kernels):
            U_y_hat += (self.weights[:,:,k] * y_hat[self.k_size:self.k_size+self.num_rows, self.k_size:self.k_size+self.num_cols,k])
        
        Lambda_U_y_hat = self.Lambda * U_y_hat
        w = np.zeros_like(self.weights)
        for k in range(self.num_kernels):
            w[:,:,k] = self.weights[:,:,k] * Lambda_U_y_hat
        
        term = np.sum(self.v * w, axis=2) / self.v_norm_squared
        
        tau = 1.0 / (t * self.v_norm_squared)
        tmp = (tau * self.b + term) / (tau + 1)
        vec = self.Lambda * (tmp - U_y_hat)
        
        prox_g1 = np.zeros_like(z_z1)
        for k in range(self.num_kernels):
            prox_g1[self.k_size:self.k_size+self.num_rows, self.k_size:self.k_size+self.num_cols,k] = self.weights[:,:,k] * vec
        
        prox_g1 = y_hat + prox_g1
        return z_z1 - t * prox_g1

    def prox_g2_star(self, z_z2):
        # TODO: Switch to L1
        """Proximal operator for TV term"""
        norm = np.sqrt(np.sum(z_z2**2, axis=2))
        scale = np.minimum(1, 1/norm)
        return self.lmda * (z_z2 * scale[:,:,np.newaxis])

    def solve(self, n_iters):
        """
        Main Deconvolution loop adapted directly from
        
        https://github.com/danielvoconnor/TV_deblur_spaceVariantKernel/blob/master/tvDeblur_varBlur_freeBCs_DR.m
        """

        if self.log:
            self.losses = {'fidelity' : [], 
                              'TV' :    [], 
                              'cost' : []}
        
        for iter in tqdm(range(n_iters)):

            x_plus_x = self.z_x.copy()
            x_plus_z1 = self.prox_g1_star(self.z_z1)
            x_plus_z2 = self.prox_g2_star(self.z_z2)
            
            term_x  = 2*x_plus_x  - self.z_x
            term_z1 = 2*x_plus_z1 - self.z_z1
            term_z2 = 2*x_plus_z2 - self.z_z2
            
            self.update_y_variables(term_x, term_z1, term_z2)
            self.update_z_variables(x_plus_x, x_plus_z1, x_plus_z2)

            if self.log:
                self.compute_metrics(x_plus_x)
        
        return x_plus_x[self.k_size:self.k_size+self.num_rows, self.k_size:self.k_size+self.num_cols]

    def update_y_variables(self, term_x, term_z1, term_z2):
        t = self.t
        A_trans_term_z = np.zeros_like(term_x)
        for k in range(self.num_kernels):
            A_trans_term_z += self.apply_cyclic_conv_2d(term_z1[:,:,k],self.eig_val_arrs_Kp_trans[:,:,k])
        A_trans_term_z += self.grad_T(term_z2)
        rhs = term_x - t * A_trans_term_z
        self.y_plus_x = np.real(ifft2(fft2(rhs) / self.eig_vals_mtrx))
        
        self.y_plus_z1 = np.zeros_like(self.z_z1)
        for k in range(self.num_kernels):
            self.y_plus_z1[:,:,k] = self.apply_cyclic_conv_2d(self.y_plus_x,self.eig_val_arrs_Kp[:,:,k])
        
        self.y_plus_z1 = term_z1 + t * self.y_plus_z1
        self.y_plus_z2 = term_z2 + t * self.grad(self.y_plus_x)

    def update_z_variables(self, x_plus_x, x_plus_z1, x_plus_z2):
        self.z_x  = self.z_x  + self.over_relax * (self.y_plus_x  - x_plus_x)
        self.z_z1 = self.z_z1 + self.over_relax * (self.y_plus_z1 - x_plus_z1)
        self.z_z2 = self.z_z2 + self.over_relax * (self.y_plus_z2 - x_plus_z2)

    def compute_metrics(self, x_plus_x):
        Kx = np.zeros((self.num_rows, self.num_cols))
        for k in range(self.num_kernels):
            Kpx = self.apply_cyclic_conv_2d(x_plus_x, self.eig_val_arrs_Kp[:,:,k])
            Kx += (self.weights[:,:,k] * Kpx[self.k_size:self.k_size+self.num_rows, self.k_size:self.k_size+self.num_cols])
        
        grad_x = self.grad(x_plus_x)
        
        recon_loss = 0.5 * np.sum((Kx - self.b)**2)
        TV_loss = self.lmda * np.sum(np.abs(grad_x))
        cost = recon_loss + TV_loss
        cost = cost / self.beta**2
        x_plus_x = x_plus_x[self.k_size:self.k_size+self.num_rows, self.k_size:self.k_size+self.num_cols]
        error = np.linalg.norm((x_plus_x - self.img_og).flatten())
        
        self.losses['cost'].append(cost)
        self.losses['fidelity'].append(error)
        self.losses['TV'].append(TV_loss)
