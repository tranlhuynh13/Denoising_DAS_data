import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pyproj import CRS, Transformer
import pandas as pd
import numpy as np
import h5py
import data

# 2. Define your neural network architecture
class SMModel(nn.Module):
    def __init__(self, in_shape, out_shape, enc_layers=5, dec_layers=5):
        super(SMModel, self).__init__()
        dims_enc = calc_dimension_SM(
            in_shape, enc_layers, mode='enc')
        self.enc = nn.ModuleList(
            [EncBlock(dims_enc[i], dims_enc[i + 1], mask_ratio=0.5) for i in range(enc_layers)])

        dims_dec = calc_dimension_SM(
            dims_enc[-1], dec_layers, mode='dec')

        self.dec = nn.ModuleList()
        for i in range(0, dec_layers):
            if i == 0:
                # Modify input channels for skip connection
                in_channel0 = dims_dec[i][1] + dims_enc[-2][1]
                self.dec.append(
                    DecBlock(in_channel0, dims_dec[i+1][1], mask_ratio=0.5))
            elif i == 1:
                # Modify input channels for skip connection
                in_channel1 = dims_dec[i][1] + dims_enc[-3][1]
                self.dec.append(
                    DecBlock(in_channel1, dims_dec[i+1][1], mask_ratio=0.5))
            elif i == (dec_layers - 1):
                self.dec.append(
                    DecBlock(dims_dec[i][1], dims_dec[i+1][1], mask_ratio=0.5, last=True))
            else:
                self.dec.append(
                    DecBlock(dims_dec[i][1], dims_dec[i+1][1], mask_ratio=0.5))

        '''
        self.dec = nn.ModuleList(
            [DecBlock(dims_dec[i][1], dims_dec[i+1][1], mask_ratio=0.5)
             for i in range(dec_layers)]
        )
        '''

    def forward(self, x):
        input = x.clone()
        # Preprocessor
        input, m = masks(input, prev_mask=None, mask_ratio=0.4)

        # Encoding
        o1 = input
        for i in range(len(self.enc)):
            o1 = self.enc[i](o1)
            # Save summands for skip connections
            if i == len(self.enc) - 3:
                s1 = o1.clone()
            if i == len(self.enc) - 2:
                s2 = o1.clone()
        # 2nd Skip connection
        s2 = F.interpolate(s2, o1.shape[2:], mode='bilinear')
        o2 = torch.cat([o1, s2], dim=1)

        # Decoding
        for i in range(len(self.dec)):
            o2 = self.dec[i](o2)
            # 1st Skip Connection
            if i == 0:
                s1 = F.interpolate(s1, o2.shape[2:], mode='bilinear')
                o2 = torch.cat([o2, s1], dim=1)

        # Postprocessor
        o2, _ = masks(o2, prev_mask=m)

        return o2


# 2.1 Define encoding block
# Important: kernel size and stride should ensure last 2 dimensions in output are halved (Typical encoder structure in U-Net)
class EncBlock(nn.Module):
    def __init__(self, in_shape, out_shape, mask_ratio=0.5):
        super(EncBlock, self).__init__()
        _, _, iH, iW = in_shape
        N, oC, _, _ = out_shape
        shape_after_conv = (N, oC, iH, iW)
        self.block = nn.Sequential(
            PartialConv2d(in_shape, shape_after_conv, kernel_size=3,
                          stride=1, padding_mode='same', bias=True, mask_ratio=mask_ratio),
            nn.BatchNorm2d(shape_after_conv[1]),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        return self.block(x)

# 2.2 Define decoding block
class DecBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mask_ratio=0.4, last=False):
        super(DecBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', 
                        align_corners=False),
            nn.Dropout2d(p=mask_ratio),
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channel),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channel),
            nn.LeakyReLU()
        )
        if last:
            block = nn.Sequential(*list(self.block.children())[:-2])
            self.block = block

    def forward(self, x):
        return self.block(x)


# 2.* Define partial convolution layer
class PartialConv2d(nn.Module):
    def __init__(self, in_shape, out_shape, kernel_size,
                 stride=1, padding_mode='same', dilation=1, bias=True, mask_ratio=0.5):
        super(PartialConv2d, self).__init__()
        in_channels = in_shape[1]
        out_channels = out_shape[1]
        # Standard convolution weights and bias
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding_mode,
                              dilation=dilation,
                              bias=bias)
        # generate masking
        mask = (torch.rand(in_shape) > mask_ratio).float()
        self.register_buffer("mask", mask)

        # Optional: initialize weights using He initialization
        nn.init.kaiming_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """
        Forward pass of partial convolution.

        Parameters:
            x (torch.Tensor): Input tensor [B, C_in, H, W]
            mask (torch.Tensor): Binary mask [B, 1 or C_in, H, W]
                                 where 1 = valid pixel; 0 = masked pixel

        Returns:
            tuple: (output_tensor, updated_mask)
        """
        mask = self.mask

        # Ensure mask has same number of channels as input for broadcasting
        if mask.shape[1] < x.shape[1]:
            mask = mask.repeat(1, int(x.shape[1]/mask.shape[1]), 1, 1)
        else:
            mask = mask[:, :x.shape[1], :, :]

        # Apply element-wise multiplication to ignore masked regions
        masked_input = x * mask

        # Perform convolution on masked input
        output = self.conv(masked_input)

        # Compute normalization factor based on valid pixels per window
        with torch.no_grad():
            ones_kernel = torch.ones_like(self.conv.weight)
            valid_count = F.conv2d(mask,
                                   ones_kernel,
                                   bias=None,
                                   stride=self.conv.stride,
                                   padding=self.conv.padding,
                                   dilation=self.conv.dilation)

            # Avoid division by zero
            valid_count = torch.clamp(valid_count, min=1e-8)

        # Normalize output by number of valid pixels in each window
        output = output * self.conv.weight.numel() / valid_count

        # Update the binary mask for next training iteration
        new_mask = (valid_count > 1e-8).float()
        self.mask = new_mask

        return output * new_mask


def masks(x, prev_mask=None, mask_ratio=0.5):
    if prev_mask is not None:
        mask = torch.ones_like(prev_mask) - prev_mask
    else:
        mask = (torch.rand_like(x[:, :1]) > mask_ratio).float()
    return x * mask, mask


def calc_dimension_SM(in_shape, n_layer=5, mode='enc'):
    assert mode in ('enc', 'dec'), "Wrong mode!"
    lshape = []
    lshape.append(in_shape)
    N, _, iH, iW = in_shape
    if mode == 'enc':
        cC = 24
    else:
        cC = 48
    cH = iH
    cW = iW
    for i in range(0, n_layer):
        if mode == 'enc':
            cH = cH // 2
            cW = cW // 2
        else:
            if i == n_layer - 3:
                cC = 32
            if i == n_layer - 2:
                cC = 16
            if i == n_layer - 1:
                cC = 1
            cH *= 2
            cW *= 2
        lshape.append((N, cC, cH, cW))
    assert len(lshape) == n_layer + 1, "Amount of layers does not match"
    return lshape


def get_padding_2d(in_shape, out_shape, kernel_size, stride=1, padding_mode='same'):
    _, _, iH, iW = in_shape
    _, _, oH, oW = out_shape
    kH, kW = kernel_size
    padding = 0
    if padding_mode == 'valid':
        padding = 0
    else:
        pH = int(((oH - 1) * stride + kH - iH) / 2)
        pW = int(((oW - 1) * stride + kW - iW) / 2)
        padding = (pH, pW)
    return padding


def soft_threshold(X, tau):
    return torch.sign(X) * torch.clamp(torch.abs(X) - tau, min=0.0)


def update_S(Y, f_theta_Y, gamma1, gamma2):
    """
    Parameters:
        Y           : torch.Tensor
        f_theta_Y   : torch.Tensor (f_theta(Y))
        gamma1      : float
        gamma2      : float

    Returns:
        S (closed-form solution)
    """
    # Residual
    A = Y - f_theta_Y

    # Threshold
    tau = gamma2 / (2.0 * gamma1)

    # Soft-threshold solution
    S = soft_threshold(A, tau)

    return S


# Does not provide accurate results, Therefore not used
def update_M(fY, M, optimizer, gamma3=1.0, num_iters=20, verbose=False):
    """
    Update Tensor M to optimize ratio delta_x{H} / delta_y{H}.
    """
    # M = M.detach().clone().requires_grad_(True)
    # optimizer = torch.optim.Adam([M], lr=lr)

    # Fourier-domain masking
    fY_t = torch.transpose(fY, -1, -2)
    
    F_fY = torch.fft.rfftn(fY_t, dim=(-2, -1), norm='ortho')
    
    window = 10
    loss_H_values = []
    loss_H_average = []
    
    for i in range(num_iters):
        optimizer.zero_grad()
        
        steepness = min(3 + i / 10., 15.)
        
        M_soft = torch.sigmoid(steepness * (M - M.mean()))
        
        # Apply learned mask on frequency spectrum
        masked_F_fY = F_fY * M_soft

        # Fourier transform back in original domain
        X = torch.fft.irfftn(masked_F_fY, dim=(-2, -1), norm='ortho')
        
        H = fY_t - X
        
        loss = loss_M(H, gamma3)
        loss_H_values.append(loss.clone().detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            M.clamp_(0., 1.)
        if verbose:
            if (i + 1) % 5 == 0:
                print(f"Update M, Iter {i + 1}: loss = {loss}")
    
    M_bin = (M > M.mean()).float().requires_grad_(True)
    H = fY_t - torch.fft.irfftn(M_bin * F_fY, dim=(-2, -1), norm='ortho')
    
    #for i in range(0, len(loss_H_values), window):
    #    loss_H_average.append(np.mean(loss_H_values[i:i + window]))
    
    '''
    plt.plot(loss_H_values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Average")
    plt.title("Loss From Optimizing H Evolution")
    plt.grid(True)
    plt.show()
    '''
    return torch.transpose(H, -1, -2).detach(), M_bin.detach()


def loss_M(H, gamma3):
    eps = 1e-8  # small constant to avoid division by zero
    H = torch.transpose(H, -1, -2)
    
    # Spatial gradients
    grad_x = H[:, :, 1:, :] - H[:, :, :-1, :]
    grad_y = H[:, :, :, 1:] - H[:, :, :, :-1]

    # Align shapes for division (crop to same size) ((samples - 1) x (traces - 1))
    min_h = min(grad_x.shape[-2], grad_y.shape[-2])
    min_w = min(grad_x.shape[-1], grad_y.shape[-1])
    grad_x_crop = grad_x[:, :, :min_h, :min_w]
    grad_y_crop = grad_y[:, :, :min_h, :min_w]

    # l1 norm of gradient ratio
    ratio = grad_x_crop / (grad_y_crop + eps)
    loss = gamma3 * torch.norm(ratio, p=1)

    return loss


def train(Y, lr_model=0.01, lr_M=0.01, decay=0.8, g1=11, g2=10, g3=0.03, n_epoch=5, it_model=1000, it_M=10, decay_interval=100, tol=1e-6, init_w=0.2):
    model = SMModel(
        in_shape=Y.shape,
        out_shape=Y.shape,
        enc_layers=5,
        dec_layers=5,
    )

    total_loss_values = []
    loss_model_values = []
    loss_S_values = []
    #loss_H_values = []
    
    loss_model_average = []
    #loss_H_average = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    Y.requires_grad_(True)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr_model)

    Y = Y.to(device)
    S = torch.zeros_like(Y, device=device, requires_grad=True)
    H = torch.ones_like(Y, device=device)
    #M = torch.ones_like(Y, requires_grad=True, device=device)
    
    freq_Y = torch.fft.rfftn(torch.transpose(Y, -1, -2), dim=(-2, -1))
    #M = torch.distributions.Laplace(0.0, 0.03).sample(freq_Y.shape).to(dtype=Y.dtype, device=device).requires_grad_(True)
    M = init_mask_dip(torch.transpose(Y, -1, -2).shape, init_w).to(device).requires_grad_(True)
    #M = init_mask_like(freq_Y).to(device).requires_grad_(True)
    #M = torch.ones_like(freq_Y, device=device, dtype=Y.dtype, requires_grad=True)
    #opt_M = torch.optim.Adam([M], lr=lr_M)
    #model.eval()
    fY = model(Y)
    verbose = False
    
    model.train()
    for epoch in range(n_epoch):
        #verbose = True
        
        if (epoch + 1) % 1000 == 0:
            verbose = True
            
        else:
            verbose = False
        
        S = S.to(device=device)
        # Step 1
        # Optimize Theta (For clean + random noise)
        for i in range(it_model):
            opt.zero_grad()
            # Loss of model f_theta()
            fY = model(Y)
            
            loss_model = g1 * (torch.norm(Y - fY - S, p=2) ** 2)
            loss_model_values.append(loss_model.item())
            
            loss_model.backward()
            opt.step()
        
        # Update f_theta(Y) for step 2
        #model.eval()
        
        # Step 2
        # Optimize S (For erratic noise)        
        # Loss from optimizing S
        loss_S =  g2 * (torch.norm(S, p=1))
        loss_S_values.append(loss_S.item())
        
        #S = update_S(Y - fY, g1, g2, it_S, step_S, verbose)
        S = update_S(Y.detach(), fY.detach(), g1, g2)
        
        # Step 3
        # Optimize M (For horizontal noise)
        # (H is only used in inference)
        #H, M_bin = update_M(fY.detach(), M, opt_M, g3, it_M, verbose)
        
        # Loss from optimizing H
        #loss_H = loss_M(H, g3)
        #loss_H_values.append(loss_H.item())
        
        # Check framework's loss
        total_loss = loss_model + loss_S #+ loss_H
        total_loss_values.append(total_loss.item())
    
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}: total_loss={total_loss.item():.6e}")
        
        
        if total_loss.item() < tol:
            print(
                f"Early stopping at epoch {epoch}, total_loss={total_loss.item():.6e}")
            break
        
        
        if ((epoch + 1) % decay_interval == 0) and (epoch > 1):
            lr_model *= decay
            #lr_M *= 0.8
            for param_group in opt.param_groups:
                param_group['lr'] = lr_model
            #for param_group in opt_M.param_groups:
            #    param_group['lr'] = lr_M
            print(f"Learning rate is reduced to {lr_model}")

    M_bin = M
    
    window = 100
    for i in range(0, len(total_loss_values), window):
        loss_model_average.append(np.mean(loss_model_values[i:i + window]))
    
    '''
    for i in range(0, len(loss_H_values), window):
        loss_H_average.append(np.mean(loss_H_values[i:i + window]))
    '''
    
    plt.plot(total_loss_values)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss Average")
    plt.title("Total Loss Evolution")
    plt.grid(True)
    plt.show()
    
    plt.plot(loss_model_average)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Average")
    plt.title("Training Model Loss Evolution")
    plt.grid(True)
    plt.show()
    
    plt.plot(loss_S_values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Average")
    plt.title("Loss From Optimizing S Evolution")
    plt.grid(True)
    plt.show()
    
    '''
    plt.plot(loss_H_average)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Average")
    plt.title("Loss From Optimizing H Evolution")
    plt.grid(True)
    plt.show()
    '''
    return model, S.detach(), M_bin.detach(), H.detach()


# Inference strategy
def infer(model, input, mask, p=100, with_mask=True):
    model.eval()
    for i in range(len(model.dec)):
        model.dec[i].block[1].train()
    with torch.no_grad():
        outputs = model(input)
        for _ in range(0, p):
            output = model(input)
            outputs = torch.cat((outputs, output), dim=0)
        
        mean_outputs = torch.mean(outputs, dim=0)[None, :, :, :]
        if with_mask:
            F_out = torch.fft.rfftn(torch.transpose(mean_outputs, -1, -2), dim=(-2, -1), norm='ortho')
            masked_F_out = F_out * mask
            clean_sig = torch.fft.irfftn(masked_F_out, dim=(-2, -1), norm='ortho')
        else:
            clean_sig = torch.transpose(mean_outputs, -1, -2)
    return torch.transpose(clean_sig, -1, -2)


def init_mask_like(F):
    h, w = F.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    center_y = h // 2
    # emphasise horizontal components, keep low vertical freq (yy close to center)
    sigma_y = h / 8.0
    M_init = torch.ones_like(yy) - torch.exp(-((yy - center_y)**2) / (2*sigma_y**2))
    return M_init.to(F.device).unsqueeze(0).unsqueeze(0)


def init_mask_dip(shape, w):
    _, _, nt, nx = shape
        
    f = torch.fft.fftfreq(nt)
    k = torch.fft.rfftfreq(nx)

    yy, xx = torch.meshgrid(f, k, indexing='ij')
    
    # Avoid division by zero at yy=0 by calculating only over the non zero values
    slope = torch.zeros_like(yy)
    nonzero_f_mask = yy != 0
    slope[nonzero_f_mask] = xx[nonzero_f_mask] / yy[nonzero_f_mask]

    mask = torch.ones_like(yy)
    mask[torch.abs(slope) < w] = 0  # suppress small slopes (horizontal events)
    
    return mask
    


# Generate synthetic training data similar to that introduced in paper
# Written by MSc. Sebastian Konietzny
def generate_training_data(nx=1024, nt=256):
    locations = np.loadtxt(
        './data/nz2d_synthetics/receiver_utm_location.txt', skiprows=1)[:, 1:3]
    ref_locations = get_coordinates('./data/SISSLE_south_FINAL.csv')

    path = './data/south30_100Hz_UTC_20230327_085601.001.h5'
    reference_noise = h5py.File(path)['DAS'][:].T[111:111+len(ref_locations)]

    dset = data.SyntheticDASDataset('./data/nz2d_synthetics/out/', locations,
                                    reference_noise, ref_locations, patch_size=(nx, nt), log_SNR=10*np.log10(0.05))
    dloader = DataLoader(dset, batch_size=1, shuffle=False)

    return dloader


def get_coordinates(path):

    df = pd.read_csv(path)
    df = df[df['status'] == 'good']

    df[df['status'] == 'good']
    df.set_index('channel', inplace=True)
    df = df.reindex(range(df.index.min(), df.index.max() + 1),
                    fill_value=np.nan)
    df = df.drop(columns=['status'])
    df.interpolate(method='linear', inplace=True)
    df.reset_index(inplace=True)

    DAS_channel_list = df['channel']
    DAS_array_lat = df['latitude']
    DAS_array_lon = df['longitude']

    wgs84 = CRS("EPSG:4326")
    utm_crs = CRS.from_dict({
        'proj': 'utm',
        'zone': int((DAS_array_lon[0] + 180) / 6) + 1,
        'south': DAS_array_lat[0] < 0
    })
    transformer = Transformer.from_crs(wgs84, utm_crs, always_xy=True)

    x_hat = []
    for lat, lon in zip(DAS_array_lat, DAS_array_lon):
        x_hat.append(transformer.transform(lon, lat))
    return np.array(x_hat)
