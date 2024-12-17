import numpy as np
import scipy
from scipy import fft
import tqdm

class ADMMTVSolver():
    """
    This class implements non-blind Deconvolution on a grayscale blurry and noisy
    image with a Total Variation (TV) regularizer.

    We use the Alternating Deconvolution Method of Multupliers (ADMM) algorithm to iteratively
    minimize the reconstruction loss while enforcing sparse gradients.
    """

    def __init__(self, img, psf, lmda, rho, log=False):
        self.img = img
        self.psf = psf
        self.lmda = lmda
        self.rho = rho
        self.threshold = lmda/rho
        self.log = log
        self.H, self.W = img.shape
        self.precompute_x_updates()

    def precompute_x_updates(self):
        """
        The proximal operator for computing x in the main ADMM loop is simply an inverse filtering step. 
        Thankfully, most of the quantities used in it are independent of the current 
        """
        psf_ft = fft.fft2(self.psf, s=(self.H, self.W))
        self.psf_ft = psf_ft
        psf_ft_conj = np.conj(psf_ft)
        b_ft = fft.fft2(self.img)

        grad_x = np.array([[-1,1]])
        grad_y = np.array([[-1],[1]])

        grad_x_ft = fft.fft2(grad_x, s=(self.H, self.W))
        grad_x_ft_conj = np.conj(grad_x_ft)
        grad_y_ft = fft.fft2(grad_y, s=(self.H, self.W))
        grad_y_ft_conj = np.conj(grad_y_ft)

        self.grad_x_ft = grad_x_ft
        self.grad_y_ft = grad_y_ft
        self.proximal_x_num1 = psf_ft_conj * b_ft
        self.proximal_x_denom = (psf_ft_conj * psf_ft) + self.rho * (grad_x_ft_conj * grad_x_ft + grad_y_ft_conj * grad_y_ft)

    def proximal_x(self, v1, v2):
        v1_ft = fft.fft2(v1)
        v2_ft = fft.fft2(v2)
        numerator = self.proximal_x_num1 + self.rho * (np.conj(self.grad_x_ft) * v1_ft + np.conj(self.grad_y_ft) * v2_ft)
        return np.real(fft.ifft2(numerator / self.proximal_x_denom))
    
    def proximal_z(self, v):
        return np.maximum(0, v - self.threshold) - np.maximum(0, -v - self.threshold)

    def grad(self, x):
        dx = np.real(fft.ifft2(self.grad_x_ft * fft.fft2(x)))
        dy = np.real(fft.ifft2(self.grad_y_ft * fft.fft2(x)))
        return dx, dy
    
    def solve(self, n_iters):
        """
        Implement Deconvolution with Total Variance regularizer through ADMM:

            minimize f(x) + g(z)
            sub. to  Ax + Bz = c

        where

            f(x) = ‖Kx-b‖₂²                 
            g(z) = λ‖z‖₁                     
            A = ∇, B = -I, c = 0             

        """

        if self.log:
            self.losses = {'fidelity' : [], 
                              'TV' :    [], 
                              'total' : []}
            self.xs = []

        x = np.zeros((self.H, self.W))
        z1 = np.zeros((self.H, self.W))
        z2 = np.zeros((self.H, self.W))
        u1 = np.zeros((self.H, self.W))
        u2 = np.zeros((self.H, self.W))

        for _ in tqdm.tqdm(range(n_iters)):
            v1 = z1 - u1
            v2 = z2 - u2
            x = self.proximal_x(v1, v2)
            dx, dy = self.grad(x)
            v1 = dx + u1
            v2 = dy + u2
            z1 = self.proximal_z(v1)
            z2 = self.proximal_z(v2)
            u1 = u1 + dx - z1
            u2 = u2 + dy - z2

            x = np.clip(x, 0, 1)

            # record reconstruction loss:
            if self.log:
                self.compute_loss(x, z1, z2, self.psf)
                self.xs.append(x)

        return x

    def compute_loss(self, x, z1, z2, psf):
        x_fft = fft.fft2(x)
        convolved = np.abs(fft.ifft2(x_fft * self.psf_ft))
        fidelity = 0.5 * np.linalg.norm(convolved - self.img)**2 # L22 reconstruction loss
        dx, dy = self.grad(x)
        TV_regularizer = self.lmda * (np.linalg.norm(dx, ord=1) + np.linalg.norm(dy, ord=1)) # L1 TV loss
        self.losses['fidelity'].append(fidelity)
        self.losses['TV'].append(TV_regularizer)
        self.losses['total'].append(fidelity + TV_regularizer)