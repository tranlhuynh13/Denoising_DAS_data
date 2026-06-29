import numpy as np
from scipy.linalg import solve_banded
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, laplace


# Current up to date version, working
def struct_flatten(din, dips, radius=20):
    ns, nt = din.shape
    cube = np.zeros((ns, nt, 2 * radius + 1))

    # Loop over all central traces
    for i in range(nt):
        # Central trace goes in center position of local window
        cube[:, i, radius] = din[:, i]

        # Get local slope for current trace and integrate over time axis
        p_local = dips[:, i]
        cumulative_shift = np.cumsum(p_local)   # integration over samples

        t_index = np.arange(ns)

        for k in range(1, radius + 1):
            # forward prediction shift array per sample
            shift_forward = cumulative_shift * k
            # backward prediction shift array per sample
            shift_backward = -cumulative_shift * k

            idx_fwd = min(i + k, nt - 1)
            idx_bwd = max(i - k, 0)

            f_interp_fwd = interp1d(t_index,
                                    din[:, idx_fwd],
                                    kind='cubic',
                                    fill_value="extrapolate")
            f_interp_bwd = interp1d(t_index,
                                    din[:, idx_bwd],
                                    kind='cubic',
                                    fill_value="extrapolate")

            new_time_fwd = t_index + shift_forward
            new_time_bwd = t_index + shift_backward

            cube[:, i, radius + k] = f_interp_fwd(new_time_fwd)
            cube[:, i, radius - k] = f_interp_bwd(new_time_bwd)

    return cube


# ---------------------------------------------------------------------------
# NOT USED


def struct_flatten_old(din, dips, radius=20, order=2, eps=1e-2, eps2=1e-2):
    ns, nt = din.shape
    nw = radius * 2 + 1

    fl_din = np.zeros((ns, nt, nw), dtype=np.float64)
    fl_din[:, :, radius] = din[:, :]

    # predict
    for i in range(0, nt):
        f_trace = fl_din[:, i, radius]
        b_trace = fl_din[:, i, radius]
        for l in range(1, radius):
            if i + l < nt:
                _, _, _, f_trace = predict_trace(
                    f_trace, dips[:, i+l], order, eps, eps2, True, False)
                fl_din[:, i, radius + l] = f_trace

            if i - l >= 0:
                _, _, _, b_trace = predict_trace(
                    b_trace, dips[:, i-l], order, eps, eps2, False, False)
                fl_din[:, i, radius - l] = b_trace

        if i % max(1, nt // 10) == 0:
            print(
                f"Trace {i}: min={f_trace.min():.3e}, max={f_trace.max():.3e}")

    return fl_din


def predict_trace(din, dip, order, eps, eps2=None, f=False, adj=False):
    """
    din: trace (vector size n)
    dip: dips (vector size n)
    """

    n = din.size

    diag, offd = simple_regularization(n, order, eps, eps2)

    w, diag, offd = pwd_op(dip, diag, offd, order, f)

    trace = pwd_apply(din, w, adj)
    trace = np.nan_to_num(trace, copy=True, nan=eps)

    # print(f"Trace: {trace:.5e}")

    trace = solve_band_mat(trace.copy(), order * 2, diag, offd)

    # print(f"Reg. Trace: {trace:.5e}")

    return w, diag, offd, trace


# Revised version
def simple_regularization(n, order, eps=1e-4, eps2=None):
    assert order >= 2, "Order should be no less than 2"

    # Base stencil (Laplacian for 2nd derivative)
    diag = np.ones(n) * 6 * eps

    offd = np.zeros((n, 2 * order))

    offd[1:, order - 1] = - 4 * eps
    offd[:-1, order] = - 4 * eps

    offd[2:, order - 2] = 2 * eps
    offd[:-2, order + 1] = 2 * eps

    # Edge stabilization
    if eps2 is not None:
        diag[[0, -1]] = eps + eps2
        diag[[1, -2]] = eps + 5 * eps2

        offd[1, order - 1] = - 2 * eps
        offd[-2, order] = - 2 * eps

        offd[[2, 3], order - 2] = eps
        offd[[-3, -4], order + 1] = eps

    return diag, offd


def allpass_filter_flat(dip, order):
    """
    Returns a vector of coefficients for allpass pwd filter
    """
    n = order * 2
    b = np.zeros(n + 1)
    for k in range(0, n + 1):
        b[k] = 1.0
        for j in range(0, n):
            if j < n - k:
                b[k] *= (k + j + 1.0) / (2 * (2 * j + 1) * (j + 1))
            else:
                b[k] *= 1.0 / (2 * (2 * j + 1))
    for k in range(0, n + 1):
        for j in range(0, n):
            if j < n - k:
                b[k] *= (n - j - dip)
            else:
                b[k] *= (dip + j + 1)
    return b


def pwd_op(dips, diag, offd, order, f=True):
    """
    Build plane-wave destruction operator for given dip field.
    Returns filter coefficients and updated regularization diagonals.
    """
    n = dips.size
    bandw = 2 * order + 1
    a = np.zeros((n, bandw))

    for i in range(n):
        coeffs = allpass_filter_flat(dips[i], order)
        if f:
            coeffs = coeffs[::-1]
        # Normalize filter coefficients to prevent instability
        coeffs /= np.linalg.norm(coeffs) + 1e-12
        a[i, :] = coeffs

    # Ensure diag/offdiagonal arrays are not None and have correct shape
    if diag is None:
        diag = np.zeros(n)
    if offd is None:
        offd = np.zeros((n, 2*order))

   # Update diagonal terms based on local energy weighting
    diag += np.sum(a*a, axis=1)*0.25

   # Simple symmetric update for off-diagonals (stabilization only)
    for k in range(order):
        shift = k+1
        weight = 0.5/(shift+0.5)
        off_left = np.roll(
            np.sum(a[:, :bandw-shift]*a[:, shift:], axis=1), -shift)
        off_right = np.roll(
            np.sum(a[:, shift:]*a[:, :bandw-shift], axis=1), shift)
        off_left[:shift] = 0
        off_right[-shift:] = 0
        off_avg = (off_left+off_right)*weight

      # Fill left/right bands symmetrically around center columns of 'offd'
        if order-k-1 >= 0:
            offd[:, order-k-1] += off_avg/2
            offd[:, order+k] += off_avg/2

    return a, diag, offd


def pwd_apply(din, filts, adj=False):
    """
    Apply local plane-wave destruction filters or their adjoint.
    din: input vector (n,)
    filts: filter matrix (n x (2*order+1))
    adj: apply adjoint operator if True
    """
    n = len(din)
    na = filts.shape[1]
    nw = na // 2

    din_pad = np.pad(din, pad_width=nw, mode='edge')
    windows_inp = sliding_window_view(din_pad, window_shape=na)

   # Ensure number of windows matches n
    assert windows_inp.shape[0] == n, f"Expected {n} windows but got {windows_inp.shape[0]}"

    out = np.zeros_like(din, dtype=float)

    if not adj:
        tmp = np.sum(windows_inp*filts, axis=1)
        out[:] = tmp
    else:
        filts_flip = np.flip(filts, axis=1)
        tmp = np.sum(windows_inp*filts_flip, axis=1)
        out[:] = tmp

        # Explicitly return output!
    return out


def solve_band_mat(b, bandw, diag, offd):
    """
    Solve Ax = b where A is a symmetric banded matrix.
    b: right-hand side (vector)
    bandw: total bandwidth (even number)
    diag: main diagonal (vector length n)
    offd: off-diagonal bands (matrix shape n x 2*order)
    """
    assert bandw % 2 == 0, "bandwidth must be even"

    # Ensure inputs are numpy arrays
    if b is None or diag is None or offd is None:
        raise ValueError("Input vectors/matrices must not be None.")
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if not isinstance(diag, np.ndarray):
        diag = np.array(diag)
    if not isinstance(offd, np.ndarray):
        offd = np.array(offd)

    n = len(diag)
    l = u = bandw // 2

   # Construct ab array for solve_banded: shape=(l+u+1,n)
    ab = np.zeros((l + u + 1, n))

   # Fill main diagonal
    ab[u] = diag

   # Fill upper/lower diagonals safely
    for k in range(1, l+1):
        if offd.shape[1] >= l + k:
            ab[u - k, k:] = offd[:-k, l - k]
            ab[u + k, :-k] = offd[k:, l + k - 1]

    return solve_banded((l, u), ab, b)


def pwd_op_old(dips, diag, offd, order, f=True):
    """
    dips: slope vector (size n)
    diag: diagonal of regularization matrix
    offd: offdiagonal of regularization matrix
    order: PWD filter order
    f: forward flag

    return:
    a: filter coefficients for each slope (n x (2*order+1))
    diag: diagonal of resulting band matrix A
    offd: offdiagonals of resulting band matrix A
    """
    n = dips.size
    bandw = 2 * order + 1
    a = np.zeros((n, bandw))

    for i in range(0, n):
        if f:
            a[i, :] = allpass_filter_flat(dips[i], order)[::-1]
        else:
            a[i, :] = allpass_filter_flat(dips[i], order)
        # a[i, :] /= np.linalg.norm(a[i, :]) + 1e-12

    n_diag = np.pad(diag, (order, order), mode='constant', constant_values=0)
    for i in range(order, n + order):
        n_diag[i] += a[i - order, :] @ a[i - order, :]
    diag = n_diag[order:-order]

    offd = np.pad(offd, ((order, order), (0, 0)),
                  mode='constant', constant_values=0)
    a = np.pad(a, ((order, order), (0, 0)), mode='constant', constant_values=0)
    for i in range(order, n + order):
        for m in range(2 * order):
            # sum over valid k positions
            offd[i, m] = np.sum(
                [a[k, m + 1:] @ a[k, :- m - 1]
                 for k in range(i - order, i + order + 1)]
            )
    offd = offd[order:- order]
    a = a[order:- order]

    return a, diag, offd


def pwd_apply_old(din, filts, adj=False):
    """
    Calculate either W^T @ W @ input (adj is False) or W @ W^T @ input (else)
    din: input (vector size n)
    filts: filter matrix (size: n x (2*nw + 1)))
    nw: pwd filter radius, filter size would be: 2*nw+1 (deprecated)
    adj: adjoint flag
    """
    na = filts.shape[1]
    nw = int((na - 1) / 2)

    # Create sliding windows over padded input
    windows_inp = sliding_window_view(din, window_shape=na)

    window_filts = filts[nw:-nw]

    out = np.zeros_like(din, dtype=np.float64)

    if not adj:
        # Local convolution with position-dependent filters
        tmp = np.sum(window_filts * windows_inp, axis=1)
        windows_tmp = sliding_window_view(tmp, window_shape=na)
        window_filts = window_filts[nw:-nw]
        out = np.sum(window_filts * windows_tmp, axis=1)

    else:
        # reverse each filter along its second axis
        filts_flipped = np.flip(window_filts, axis=1)

        tmp = np.sum(filts_flipped * windows_inp, axis=1)
        windows_tmp = sliding_window_view(tmp, window_shape=na)
        filts_flipped = filts_flipped[nw:-nw]
        out = np.sum(filts_flipped * windows_tmp, axis=1)

    out = np.pad(out, pad_width=2*nw, mode='constant', constant_values=0)

    return out


def solve_band_mat_old(b, bandw, diag, offd):
    """
    Solve for band matrix problem
    din: vector (trace)
    bandw: upper bandwidth + lower bandwidth (scalar)
    diag: vector, main diagonal of band matrix
    offd: matrix, off diagonal of band matrix/band without main diagonal (size: (n, bandwidth))
    n: dimension of band matrix (matrix size: n x n)
    """
    assert bandw % 2 == 0, "bandw must be even"
    # Preprocess the parameters to correct input form for solve_banded
    l = int(bandw / 2)
    offd = offd.T
    A = np.insert(offd, l, diag, axis=0)

    # Plug into solver
    x = solve_banded((l, l), A, b)

    return x


# HEREAFTER ARE DRAFTS ONLY, i.e. NOT USED IN CURRENT IMPLEMENTATION
# Author used this, bug detected
def regularization(n, order, eps=1e-4, eps2=1e-4):
    """
    Create diagonal (vector, n) and offdiagonal (matrix, (n-1) x 2*order) of banded matrix using eps and eps2
    """
    assert order >= 2, "Order too small, should be >= 2"

    diag = np.zeros(n)
    bandw = 2*order
    offd = np.zeros((n, bandw))

    # fill diagonal
    diag[[0, n-1]] = eps2 + eps
    diag[[1, n-2]] = eps2 + 5 * eps
    diag[2:n-2] = 6 * eps

    # fill offdiagonal matrix
    offd[:, 0] = - 4 * eps
    offd[:, 1] = eps
    offd[[0, n-1], 0] = - 2 * eps

    return diag, offd


def allpass_filter_flat_new(dip, order):
    n = order * 2
    dip_clamped = np.clip(dip, -0.99, 0.99)

    b_vals = []
    for k in range(n + 1):
        val_terms = []
        for j in range(n):
            if j < n - k:
                val_terms.append((k + j + 1.0) / (2 * (2 * j + 1) * (j + 1)))
            else:
                val_terms.append(1.0 / (2 * (2 * j + 1)))
        for j in range(n):
            if j < n - k:
                val_terms.append((n - j - dip_clamped))
            else:
                val_terms.append((dip_clamped + j + 1))

        val_terms = np.array(val_terms, dtype=np.float64)
        val_terms[np.abs(val_terms) < 1e-12] = 1e-12   # avoid log(0)

        sign_term = np.sign(np.prod(val_terms))
        log_sum = np.sum(np.log(np.abs(val_terms)))
        b_val = np.exp(np.clip(log_sum, -700, 700)) * sign_term
        b_vals.append(b_val)

    b_vals = np.array(b_vals)
    b_vals /= np.linalg.norm(b_vals) + 1e-12

    return b_vals


def solve_band_mat_test(b, bandw, diag, offd):
    assert bandw % 2 == 0, "bandw must be even"
    l = u = int(bandw / 2)

    n = len(diag)
    ab = np.zeros((l + u + 1, n))

    # Fill main diagonal
    ab[u, :] = diag

    # Fill upper and lower bands from offd array columns.
    # Assuming offd[:, order-1], offd[:, order], ... correspond to offsets.
    for k in range(1, l+1):
        if k < offd.shape[1]:
            ab[u - k, :k*-1 or None] = offd[k:, l - k]
            ab[u + k, :n - k] = offd[:-k, l + k - 1]

    x = solve_banded((l, u), ab.copy(), b.copy())
    return x
