import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
import scipy

def get_gaussian_psf(k_size, std):
    gaussian_1d = scipy.signal.windows.gaussian(k_size, std).reshape(k_size, 1)
    gaussian_kernel = gaussian_1d @ gaussian_1d.T
    gaussian_kernel /= np.sum(gaussian_kernel)

    return gaussian_kernel

def create_distorted_psf_grid(base_psf, image_size, grid_size):
    """
    Rotate PSFs radially according to their position relative to center of image
    """
    rows, cols = grid_size
    h, w = image_size
    psf_h, psf_w = base_psf.shape
    
    x_centers = np.linspace(0, w, cols+2)[1:-1]  
    y_centers = np.linspace(0, h, rows+2)[1:-1]  
    
    center_x = w / 2
    center_y = h / 2
    
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    psf_grid = np.zeros((rows, cols, psf_h, psf_w))
    
    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):

            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx**2 + dy**2)
            angle = -np.degrees(np.arctan2(dy, dx))
            stretch_factor = 1 + (distance / max_dist) * 1.5
            
            stretched_psf = np.copy(base_psf)
            new_width = int(psf_w * stretch_factor)
            temp_psf = cv2.resize(stretched_psf, (new_width, psf_h), 
                                interpolation=cv2.INTER_LINEAR)
            
            if new_width > psf_w:
                start = (new_width - psf_w) // 2
                temp_psf = temp_psf[:, start:start+psf_w]
            else:
                pad_width = (psf_w - new_width) // 2
                temp_psf = np.pad(temp_psf, ((0,0), (pad_width, pad_width)), 
                                mode='constant')
            
            rotated_psf = rotate(temp_psf, angle, reshape=False)
            rotated_psf = rotated_psf / rotated_psf.sum()
            psf_grid[i, j] = rotated_psf
            
    return psf_grid


def radial_gaussian_grid(k_size, grid_size, image_size, std):
    """
    Create spatially varying gaussian PSFs on a grid using radial distortion
    
    Args:
        image_size (tuple): Size of the image (height, width)
        grid_size (tuple): Size of PSF grid (rows, cols)
        psf_size (int): Size of each PSF kernel
        
    Returns:
        kernels: Array of PSFs with shape (psf_size, psf_size, grid_size[0]*grid_size[1])
        U: Array of weights with shape (image_size[0], image_size[1], grid_size[0]*grid_size[1])
    """
    grid_size_H, grid_size_W = grid_size
    x = np.linspace(-2, 2, k_size)
    y = x[:, np.newaxis]
    base_psf = get_gaussian_psf(k_size, std)
    
    psf_grid = create_distorted_psf_grid(base_psf, image_size, (grid_size_H, grid_size_W))
    num_psfs = grid_size_H * grid_size_W
    kernels = np.zeros((k_size, k_size, num_psfs))
    U = np.zeros((*image_size, num_psfs))
    
    x = np.linspace(-1, 1, grid_size_W)
    y = np.linspace(-1, 1, grid_size_H)
    xx, yy = np.meshgrid(x, y)
    
    # paste PSFs into kernels obj
    for i in range(grid_size_H):
        for j in range(grid_size_W):
            idx = i * grid_size_W + j
            kernels[:,:,idx] = psf_grid[i,j]
    
    # weights (U) fall off with distance 
    img_y = np.linspace(-1, 1, image_size[0])
    img_x = np.linspace(-1, 1, image_size[1])
    img_xx, img_yy = np.meshgrid(img_x, img_y)
    
    for i in range(grid_size_H):
        for j in range(grid_size_W):
            idx = i * grid_size_W + j
            dist = np.sqrt((img_xx - xx[i,j])**2 + (img_yy - yy[i,j])**2)
            U[:,:,idx] = np.exp(-3 * dist**2)  
    
    U_sum = U.sum(axis=2, keepdims=True)
    U = U / U_sum
    
    return kernels, U


def random_gaussian_grid(k_size, grid_size, image_size, std_min=0.01, std_max=4):
    """
    Create a grid of gaussian PSFs with random standard deviations
    
    Args:
        k_size (int): Size of each PSF kernel
        grid_H (int): Number of grid rows
        grid_W (int): Number of grid cols
        std_min (float): Minimum standard deviation
        std_max (float): Maximum standard deviation
        
    Returns:
        kernels: Array of PSFs with shape (k_size, k_size, grid_H*grid_W)
        U: Array of weights with shape (grid_H*k_size, grid_W*k_size, grid_H*grid_W)
    """
    grid_H, grid_W = grid_size
    num_psfs = grid_H * grid_W
    kernels = np.zeros((k_size, k_size, num_psfs))
    U = np.zeros((*image_size, num_psfs))
    
    x = np.linspace(-1, 1, grid_W)
    y = np.linspace(-1, 1, grid_H)
    xx, yy = np.meshgrid(x, y)
    x = np.linspace(-2, 2, k_size)
    y = x[:, np.newaxis]
    
    for i in range(grid_H):
        for j in range(grid_W):
            idx = i * grid_W + j
            std = np.random.uniform(std_min, std_max)
            kernel = get_gaussian_psf(k_size, std)
            kernels[:,:,idx] = kernel
            
    img_y = np.linspace(-1, 1, image_size[0])
    img_x = np.linspace(-1, 1, image_size[1])
    img_xx, img_yy = np.meshgrid(img_x, img_y)
    
    for i in range(grid_H):
        for j in range(grid_W):
            idx = i * grid_W + j
            dist = np.sqrt((img_xx - xx[i,j])**2 + (img_yy - yy[i,j])**2)
            U[:,:,idx] = np.exp(-7 * dist**2)
            
    U_sum = U.sum(axis=2, keepdims=True)
    U = U / U_sum
    
    return kernels, U


def visualize_psf_grid(psfs, weights):
    H, W, num_psfs = weights.shape
    grid_H = int(np.sqrt(num_psfs))
    grid_W = num_psfs // grid_H
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    k_size = psfs.shape[0]
    psf_grid = np.zeros((grid_H * k_size, grid_W * k_size))
    
    for i in range(grid_H):
        for j in range(grid_W):
            idx = i * grid_W + j
            psf_grid[i*k_size:(i+1)*k_size, j*k_size:(j+1)*k_size] = psfs[:, :, idx]
    
    ax1.imshow(psf_grid, cmap='gray')
    ax1.set_title('Spatially-Varying PSF')
    ax1.axis('off')
    
    im = ax2.imshow(weights[:, :, 0], cmap='gray')
    ax2.set_title('Weight for First PSF')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def apply_spatial_psf(img, psfs, weights):
    """
    Apply blur based on Nagy-O'Leary model for spatially-varying PSF
    """
    blurred = np.zeros_like(img)
    for i in range(psfs.shape[2]):
        psf_result = scipy.signal.convolve2d(img, psfs[:,:,i], mode='same', boundary='wrap')
        blurred += psf_result * weights[:,:,i]
        
    return blurred
