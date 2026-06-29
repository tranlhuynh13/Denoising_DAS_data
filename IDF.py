from scipy.fft import rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
from SOMF_slope import *
from SOMF_flatten import *
import numpy as np

def bandpass_filter(din, fh=200, n=6, fs=2000):
    assert len(din.shape) == 2, "Only 2D inputs are accepted"
    ns, nt = din.shape
    assert ns == fs, "Sample size doesn't match sampling rate"

    # Initialize Butterworth filter.
    nyq = 0.5 * fs
    fh_n = fh / nyq
    b, a = butter(N=n, Wn=fh_n, btype='low', analog=False, output='ba')

    do = np.zeros_like(din)
    for i in range(0, nt):
        # Filter the signal trace by trace
        do[:, i] = filtfilt(b, a, din[:, i]) 

    return do 

#def SOMF(data, input, n_iter=5, l_iter=20, dip_order=3, eps_dv=1e-2, eps_cg=1, tol_cg=1e-6, rect=np.array([20,20]), radius=20, flat_order=2, eps=1e-4, eps2=1e-4, slope_mask_coeff=0.005):
def SOMF(data, input, n_iter=5, l_iter=20, dip_order=3, eps_dv=1e-2, eps_cg=1, tol_cg=1e-6, rect=np.array([20,20]), gaussian_sigma=1, radius=20, slope_mask_coeff=0.005):

    dips = local_slope_estimation(data, n_iter, l_iter, dip_order, eps_dv, eps_cg, tol_cg, rect, slope_mask_coeff, gaussian_sigma)

    #dips[[1, -1], :] = 0
    #dips[:, [1, - 1]] = 0

    # --- Sanitize dips before flattening ---
    #print("Raw dips stats:", np.nanmin(dips), np.nanmax(dips))

    # Replace NaN/Inf with zeros
    #dips = np.nan_to_num(dips, nan=0.0, posinf=0.0, neginf=0.0)

    # Clamp slopes to physically reasonable range
    # For normalized data, slopes should usually be within [-1, +1]
    #dips /= np.max(np.abs(dips)) + 1e-12

    flat_din = struct_flatten(input, dips, radius)

    #flat_din = struct_flatten_old(input, dips, radius, flat_order, eps, eps2)
    #flat_din[np.abs(flat_din) > 100] = np.sign(flat_din[np.abs(flat_din) > 100]) * 100
    #flat_din = np.nan_to_num(flat_din)
    #filt_flat_din = np.zeros_like(flat_din)

    do = np.median(flat_din, axis=2, keepdims=False)

    #filt_flat_din = median_filter(flat_din[:, :, :], size=nw, axes=2)
    
    #do = filt_flat_din[:, :, filt_flat_din.shape[2]//2]
    print(f"output shape: {do.shape}")
    return do


def dip_filter(din, w):
    """
    din : samples x traces
    w   : slope threshold 
    """
    din_T = din.T 
    nt, nx = din_T.shape
    
    freq_din = rfft2(din_T, norm='ortho')
    
    f = fftfreq(nt)
    k = rfftfreq(nx)

    F, K = np.meshgrid(f, k, indexing='ij')
    
    # Avoid division by zero at F=0 by setting those slopes very large
    slope = np.zeros_like(F)
    nonzero_f_mask = F != 0
    slope[nonzero_f_mask] = K[nonzero_f_mask] / F[nonzero_f_mask]

    mask = np.ones_like(freq_din)
    mask[np.abs(slope) < w] = 0  # suppress small slopes (horizontal events)

    filtered_freq_din = freq_din * mask
    
    dout_T = irfft2(filtered_freq_din, s=din_T.shape, norm='ortho')
    
    return dout_T.T    

'''
NOT USED
def dip_filter(din, w):
    """
    din : sample x trace
    w : slope threshold 
    """
    din = din.T
    nt = din.shape[0]
    
    # freq_din has shape (trace x (sample // 2 + 1))
    freq_din = rfft2(din)

    nw = w * nt
    ns2 = freq_din.shape[1]
    mask = np.zeros_like(freq_din)

    for i1 in range(0, nt):
        for i2 in range(0, ns2):
            if i2 < np.abs((ns2 / nw) * (i2 - (nt / 2))):
                mask[i1, i2] = 1
    
    freq_din = mask * freq_din

    dout = irfft2(freq_din, s=din.shape)
    dout = dout.T
    return dout
'''