import numpy as np
import torch
import torch.fft
from scipy.signal import welch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from pathlib import Path


class CubicSplineInterpolator:
    """
    Natural cubic-spline interpolator along an irregular 2D polyline.
    """
    def __init__(self, coords):
        self.coords = coords
        self.s = self._compute_s(coords)
        self.N = coords.shape[0]
        self._A = None

    @staticmethod
    def _compute_s(x):
        h = np.linalg.norm(np.diff(x, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(h)))

    def _build_A(self):
        if self._A is None:
            N = self.N
            h = np.diff(self.s)
            A = np.zeros((N, N), dtype=float)
            A[0, 0] = A[-1, -1] = 1.0
            idx = np.arange(1, N-1)
            A[idx, idx-1] = h[:-1]
            A[idx, idx] = 2 * (h[:-1] + h[1:])
            A[idx, idx+1] = h[1:]
            self._A = A
        return self._A

    def second_derivatives(self, y):
        """
        Compute second derivatives M for natural cubic spline.
        """
        A = self._build_A()
        rhs = np.zeros_like(y, dtype=float)
        h = np.diff(self.s)
        rhs[1:-1] = 6.0 * ((y[2:] - y[1:-1]) / h[1:, None] - (y[1:-1] - y[:-2]) / h[:-1, None])
        M = np.linalg.solve(A, rhs)
        return M

    def _project(self, points):
        seg = np.diff(self.coords, axis=0)
        seg_len2 = np.sum(seg**2, axis=1)
        M_pts = points.shape[0]
        s_hat = np.empty(M_pts)
        idx_lo = np.empty(M_pts, dtype=int)
        for j, p in enumerate(points):
            v = p - self.coords[:-1]
            t = np.clip(np.sum(v * seg, axis=1) / seg_len2, 0.0, 1.0)
            proj = self.coords[:-1] + t[:, None] * seg
            k = np.argmin(np.sum((proj - p)**2, axis=1))
            idx_lo[j] = k
            s_hat[j] = self.s[k] + t[k] * (self.s[k+1] - self.s[k])
        return s_hat, idx_lo

    def interpolate(self, y, new_coords):
        M_mat = self.second_derivatives(y)
        s_hat, idx_lo = self._project(new_coords)
        idx_hi = np.clip(idx_lo + 1, 0, self.N - 1)
        h_seg = self.s[idx_hi] - self.s[idx_lo]
        A = (self.s[idx_hi] - s_hat) / h_seg
        B = 1.0 - A
        C = (A**3 - A) * (h_seg**2) / 6.0
        D = (B**3 - B) * (h_seg**2) / 6.0
        y0 = y[idx_lo]
        y1 = y[idx_hi]
        M0 = M_mat[idx_lo]
        M1 = M_mat[idx_hi]
        y_hat = A[:, None] * y0 + B[:, None] * y1 + C[:, None] * M0 + D[:, None] * M1
        return y_hat

class SyntheticDASDataset(Dataset):
    def __init__(self, root, locations, ref, ref_locations, patch_size, dx=4.0, dt=0.01, log_SNR=(-2,4), transforms=None, size=1000, seed=42):
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.root = Path(root)
        self.paths = sorted(self.root.glob("*.npy"))
        assert self.paths, "No .npy files found in {self.root!r}"
        self.locations = locations
        self.ref = torch.from_numpy(ref)
        self.ref_locations = ref_locations
        self.interpolator = CubicSplineInterpolator(locations)

        self.nx, self.nt = patch_size 
        self.dx = dx
        self.dt = dt
        self.log_SNR = log_SNR
        self.transforms = transforms
        self.size = size

        self.sampler = NoiseGenerator(self.ref, self.nx, self.nt, fs=1/self.dt)
        
        self.data = [np.load(p, mmap_mode=None) for p in self.paths]
        self.data = [self.interpolator.interpolate(x, self.ref_locations) for x in self.data]
        self.arrival_times = [
            (np.abs(d) > 0.1 * np.percentile(np.abs(d), 50)).argmax(axis=1) for d in self.data
        ]


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        x = self.data[idx % len(self.data)]

        NX = len(self.ref_locations)
        ix = np.random.randint(0, max(NX - self.nx + 1, 1))

        _, NT = x.shape
        arrivals = self.arrival_times[idx % len(self.data)]
        arrival = int(np.mean(arrivals[int(ix / self.ref.shape[0] * x.shape[0]):int((ix + self.nx) / self.ref.shape[0] * x.shape[0])]))

        # we want arrivals at positions [1/6, 2*1/6]
        min_it = max(0, arrival - 2*self.nt//6)
        max_it = min(NT - self.nt, arrival - self.nt//6)
        if min_it > max_it:
            min_it = max(0, arrival - self.nt + 1)
            max_it = min(NT - self.nt, arrival)
        it = np.random.randint(min_it, max_it + 1)

        patch = x[ix:ix+self.nx,it:it+self.nt]
        #self.interpolator.interpolate(x[:,it:it+self.nt], self.ref_locations[ix:ix+self.nx])

        x = torch.from_numpy(patch)
        if self.transforms:
            x = self.transforms(x)
        
        x = x.unsqueeze(0)

        if isinstance(self.log_SNR, (int, float)):
            snr_db = self.log_SNR
        else:
            snr_db = np.random.uniform(*self.log_SNR)
        
        noise = self.sampler.sample(ix)
        noise = noise * x.std()/(noise.std()+1e-10) * 10**(-snr_db / 20)
        return x + noise, x, noise



class NoiseGenerator:
    def __init__(self, ref, nx, nt,
                 fs=100.0, erratic_frac=0.1, nperseg=512, n_components=1):
        """
        Parameters:
            ref: (channels x time) reference signal (torch.Tensor)
            nx, nt: patch size (spatial x temporal)
            fs: sampling frequency
            nperseg: Welch window size
            n_components: horizontal (low-rank) noise components
        """
        self.device = ref.device
        self.dtype = ref.dtype

        self.nx = nx
        self.nt = nt
        self.fs = fs
        self.erratic_frac = erratic_frac
        self.nperseg = nperseg
        self.n_components = n_components

        self.ref = ref

        # Low-rank decomposition (horizontal structure)
        U, S, Vh = torch.linalg.svd(self.ref, full_matrices=False)
        self.rank_k = (U[:,:n_components] * S[:n_components]) @ Vh[:n_components]
        self.residual = self.ref - self.rank_k

        # Energy fractions
        var_total = torch.var(self.ref)
        var_h = torch.var(self.rank_k)
        var_res = torch.abs(var_total - var_h)

        self.w_h = var_h / var_total
        w_res = var_res / var_total
        self.w_hf = (1 - self.erratic_frac) * w_res
        self.w_e = self.erratic_frac * w_res

        self.sigma = self.ref.std(dim=-1, keepdim=True) + 1e-10
        self.p = 0.01

    def sample_high_frequency_noise(self, ix=None, it=None, sigma=None):
        if ix is None:
            ix = torch.randint(0, max(self.ref.shape[0] - self.nx + 1, 1), (1,)).item()
        if it is None:
            it = torch.randint(0, max(self.ref.shape[1] - self.nt + 1, 1), (1,)).item()
        if sigma is None:
            sigma = self.sigma[ix:ix + self.nx]

        n_freq = self.nt // 2 + 1
        freqs = torch.linspace(0, self.fs / 2, n_freq)

        ref_np = self.ref[ix:ix + self.nx, it:it + self.nt].detach().cpu().numpy()
        f_ref, Pxx = welch(ref_np, fs=self.fs, window='hann', nperseg=self.nperseg, axis=-1)
        interp_fn = interp1d(f_ref, Pxx, kind='linear', fill_value='extrapolate', axis=-1)
        Pxx_interp = interp_fn(freqs.cpu().numpy())
        Pxx_interp = torch.tensor(Pxx_interp, dtype=self.dtype, device=self.device)
        Pxx_interp = torch.clamp(Pxx_interp, min=1e-10)

        real = torch.randn((1, self.nx, n_freq), device=self.device)
        imag = torch.randn((1, self.nx, n_freq), device=self.device)
        imag[..., 0] = 0.0
        if self.nt % 2 == 0:
            imag[..., -1] = 0.0
        shaped_fft = torch.complex(real, imag) * torch.sqrt(Pxx_interp)

        full_fft = torch.zeros((1, self.nx, self.nt), dtype=torch.complex64, device=self.device)
        full_fft[..., :n_freq] = shaped_fft
        if self.nt % 2 == 0:
            full_fft[..., n_freq:] = torch.conj(shaped_fft[..., 1:-1].flip(dims=[-1]))
        else:
            full_fft[..., n_freq:] = torch.conj(shaped_fft[..., 1:].flip(dims=[-1]))

        hf_noise = torch.fft.ifft(full_fft, dim=-1).real
        hf_noise = hf_noise / (hf_noise.std(dim=-1, keepdim=True) + 1e-10)
        return hf_noise * self.w_hf * sigma

    def sample_horizontal_noise(self, ix=None, it=None, sigma=None):
        if ix is None:
            ix = torch.randint(0, max(self.ref.shape[0] - self.nx + 1, 1), (1,)).item()
        if it is None:
            it = torch.randint(0, max(self.ref.shape[1] - self.nt + 1, 1), (1,)).item()
        if sigma is None:
            sigma = self.sigma[ix:ix + self.nx]

        freqs = torch.linspace(0, self.fs / 2, self.nt // 2 + 1)
        f_low, f_high = 0.01, 0.9 * self.fs / 2
        band = (freqs >= f_low) & (freqs <= f_high)

        basis = []
        for _ in range(self.n_components):
            white_fft = torch.randn(freqs.shape[0], dtype=torch.cfloat, device=self.device)
            white_fft[~band] = 0.0
            white_fft[0] = torch.complex(white_fft[0].real.clone(), torch.tensor(0.0, device=self.device))
            if self.nt % 2 == 0:
                white_fft[-1] = torch.complex(white_fft[-1].real.clone(), torch.tensor(0.0, device=self.device))

            full_fft = torch.zeros(self.nt, dtype=torch.cfloat, device=self.device)
            full_fft[:freqs.shape[0]] = white_fft
            if self.nt % 2 == 0:
                full_fft[freqs.shape[0]:] = torch.conj(white_fft[1:-1].flip(dims=[0]))
            else:
                full_fft[freqs.shape[0]:] = torch.conj(white_fft[1:].flip(dims=[0]))
            basis.append(torch.fft.ifft(full_fft).real)

        basis = torch.stack(basis, dim=0)
        W = torch.randn((self.nx, self.n_components), device=self.device)
        h_noise = W @ basis
        h_noise = h_noise / (h_noise.std(dim=-1, keepdim=True) + 1e-10)
        return h_noise * self.w_h * sigma

    def sample_erratic_noise(self, sigma=None):
        if sigma is None:
            sigma = self.sigma.mean()

        b = sigma / torch.sqrt(self.w_e / torch.tensor(2 * self.p, device=self.device, dtype=self.dtype))
        mask = torch.rand((self.nx, self.nt), device=self.device) < self.p
        lap = torch.distributions.laplace.Laplace(loc=0.0, scale=b).sample((self.nx, self.nt))
        return lap.to(self.device) * mask

    def sample(self, ix=None, sigma=None):

        NX, NT = self.ref.shape[-2:]
        if ix is None:
            ix = torch.randint(0, max(NX - self.nx + 1, 1), (1,)).item()
        it = torch.randint(0, max(NT - self.nt + 1, 1), (1,)).item()

        N_hf = self.sample_high_frequency_noise(ix, it, sigma)
        N_h = self.sample_horizontal_noise(ix, it, sigma)
        N_e = self.sample_erratic_noise(sigma)
        return N_hf + N_h + N_e