import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import laplace, gaussian_filter
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt


def local_slope_estimation(dn,
                           n_iter=5,
                           l_iter=20,
                           order=3,
                           eps_dv=0.01,
                           eps_cg=1,
                           tol_cg=1e-6,
                           rect=np.array([20, 20]),
                           slope_mask_coeff=0.05,
                           gaussian_sigma=1
                           ):
    """
    Estimate the local slope of the input data.
    """
    assert order in [3, 5], "Order must be 3 or 5"
    sigma = np.zeros_like(dn)
    n1, n2 = sigma.shape
    for _ in range(0, n_iter):
        G, P_u = allpass_filter_slope(dn, sigma, order)
        """
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(G, aspect='auto', cmap='jet')
        plt.title("G")
        plt.subplot(122)
        plt.imshow(P_u, aspect='auto', cmap='jet')
        plt.title("P_u")
        plt.show()
        """
        delta_sigma = conjgrad(b=-P_u,
                               A=G,
                               x0=None,
                               W_op=smoothing_filter,
                               rect=rect,
                               maxiter=l_iter,
                               dim=(n1, n2),
                               eps_dv=eps_dv,
                               eps_cg=eps_cg,
                               tol_cg=tol_cg,
                               )
        sigma += delta_sigma
        #print("mean sigma:", np.mean(sigma))
        #print("min/max sigma:", np.min(sigma), np.max(sigma))
    
    mask = np.abs(dn) > slope_mask_coeff * np.max(np.abs(dn))
    sigma *= mask
    sigma = np.clip(sigma, -0.5, 0.5)

    smoothed_dips = gaussian_filter(sigma, gaussian_sigma)
    smoothed_dips = laplace(smoothed_dips, mode='reflect')
    return smoothed_dips


def conjgrad(b, A, x0, W_op, rect, maxiter, dim, eps_dv, eps_cg, tol_cg,
             eps_id=1e-3):
    ns, nt = dim
    n = ns * nt

    rhs = b.flatten(order='F').astype(np.float64, copy=False)
    den = A.flatten(order='F').astype(np.float64, copy=False)

    if eps_dv is not None and eps_dv > 0.0:
        w = 1.0 / np.hypot(den, eps_dv)
        rhs = rhs * w
        den = den * w

    rhs = den * rhs
    den2 = den * den  # A^T A diagonal

    def apply_M(x):
        y = den2 * x

        Wx = W_op(x, dim, rect, adj=False)
        y += (eps_cg ** 2) * W_op(Wx, dim, rect, adj=True)

        y += eps_id * x
        return y

    x = np.zeros(n, dtype=np.float64) if x0 is None else x0.flatten(order='F').astype(np.float64, copy=False)
    r = rhs - apply_M(x)
    p = r.copy()

    r0 = np.dot(r, r)
    if r0 == 0.0:
        return x.reshape(dim, order='F')

    rsold = r0

    for i in range(maxiter):
        Ap = apply_M(p)
        denom = np.dot(p, Ap)

        if denom <= 0.0 or not np.isfinite(denom):
            break

        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = np.dot(r, r)
        relres = rsnew / r0
        #print(f"Iter: {i+1}, relres {relres}")

        if relres < tol_cg**2:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x.reshape(dim, order='F')


# Now using this


def smoothing_filter(x, dim, rect, adj):
    x = x.reshape(dim, order='F')
    rx, ry = rect
    out = np.zeros_like(x)

    if not adj:
        dxx = laplace(x, mode='reflect', axes=0)
        dyy = laplace(x, mode='reflect', axes=1)
        lap = (1 / (rx ** 2)) * dxx + (1 / (ry ** 2)) * dyy

        out = dis_double_integrate(lap, axis=0)
    else:
        out_adj_int = dis_double_diff(x, axis=0)

        # Laplacian part is self-adjoint
        dxx = laplace(out_adj_int, mode='reflect', axes=0)
        dyy = laplace(out_adj_int, mode='reflect', axes=1)
        out = (1 / (rx ** 2)) * dxx + (1 / (ry ** 2)) * dyy

        #out = dis_double_diff(out, axis=0, pad_mode='reflect')

    x = x.flatten(order='F')
    return out.flatten(order='F')


def dis_double_diff(X, axis=0, pad_mode='reflect'):
    dxx = np.diff(np.diff(X, axis=axis), axis=axis)

    pad_width = [(0, 0)] * X.ndim
    pad_width[axis] = (1, 1)
    out = np.pad(dxx, pad_width=pad_width, mode=pad_mode)

    if out.shape[axis] > X.shape[axis]:
        out = out[: X.shape[axis]]

    return out


def dis_double_integrate(X, axis=0):
    """
    Forward then backward integration
    Since target is erratic noise, filter along temporal dimension
    """
    if axis not in (0, 1):
        raise ValueError("Axes to convolve not allowed")

    Y = np.zeros_like(X)

    if axis == 0:
        for i in range(X.shape[0]):
            # Integrate forward
            Y[i, :] += np.cumsum(X[i, :])
            # Integrate backward
            Y[i, :] = np.cumsum(Y[i, ::-1])[::-1]

    if axis == 1:
        for i in range(X.shape[1]):
            # Integrate forward
            Y[:, i] += np.cumsum(X[:, i])
            # Integrate backward
            Y[:, i] = np.cumsum(Y[::-1, i])[::-1]

    return Y


def allpass_filter_slope(din, sigma, order=3):
    """
    Idea: Zero step convolution of inverted filter T with next trace, 
    minus by convolution of filter T with current trace.
    """
    ns, nt = din.shape
    p = order // 2  # pad length

    # T and T_d have shape sample x trace x order
    T = B(sigma, order, d=False)
    T_d = B(sigma, order, d=True)

    # Pad outputs wit mode 'reflect'
    din_padded = np.pad(din, ((p, p), (0, 1)), mode='reflect')
    P_u_padded = np.zeros((ns + 2 * p, nt + 1))
    G_padded = np.zeros((ns + 2 * p, nt + 1))

    # Corresponds to equation 19 in paper from H. Wang et. al
    for i2 in range(0, nt):
        for i1 in range(p, ns + p):
            cw = din_padded[i1 - p:i1 + p + 1, i2]  # window of current trace
            nw = din_padded[i1 - p:i1 + p + 1, i2 + 1]  # window of next trace

            P_u_padded[i1, i2] = (nw - cw[::-1]) @ T[i1 - p, i2, :]
            G_padded[i1, i2] = (nw - cw[::-1]) @ T_d[i1 - p, i2, :]

    P_u = P_u_padded[p:ns + p, :nt]
    G = G_padded[p:ns + p, :nt]

    return G, P_u


def B(sigma, order=3, d=False):
    """
    Returns the coefficients of a allpass filter of 3rd or 5th order
    Parameters:
    sigma : slope
    order : order of the filter
    d : bool, optional
        If True, gives the coefficients of B()
        Else, gives the coefficients of its derivative wrt sigma.
        Default is False.
    Returns:
    b : Numpy-array
        Coefficients of the filter
    """
    assert order in [3, 5], "Order must be 3 or 5"
    nx, ny = sigma.shape
    b = np.zeros((nx, ny, order))
    if order == 3:
        if not d:
            b[:, :, 0] = (1 - sigma) * (2 - sigma) / 12
            b[:, :, 1] = (2 + sigma) * (2 - sigma) / 6
            b[:, :, 2] = (1 + sigma) * (2 + sigma) / 12
        else:
            b[:, :, 0] = -(2 - sigma) / 12 - (1 - sigma) / 12
            b[:, :, 1] = (2 - sigma) / 6 - (2 + sigma) / 6
            b[:, :, 2] = (2 + sigma) / 12 + (1 + sigma) / 12
    elif order == 5:
        if not d:
            b[:, :, 0] = (1 - sigma) * (2 - sigma) * \
                (3 - sigma) * (4 - sigma) / 1680
            b[:, :, 1] = (4 - sigma) * (2 - sigma) * \
                (3 - sigma) * (4 + sigma) / 420
            b[:, :, 2] = (4 - sigma) * (3 - sigma) * \
                (3 + sigma) * (4 + sigma) / 280
            b[:, :, 3] = (4 - sigma) * (2 + sigma) * \
                (3 + sigma) * (4 + sigma) / 420
            b[:, :, 4] = (1 + sigma) * (2 + sigma) * \
                (3 + sigma) * (4 + sigma) / 1680
        else:
            b[:, :, 0] = -(2 - sigma) * (3 - sigma) * (4 - sigma)
            b[:, :, 0] -= (1 - sigma) * (3 - sigma) * (4 - sigma)
            b[:, :, 0] -= (1 - sigma) * (2 - sigma) * (4 - sigma)
            b[:, :, 0] -= (1 - sigma) * (2 - sigma) * (3 - sigma)
            b[:, :, 0] /= 1680

            b[:, :, 1] = -(2 - sigma) * (3 - sigma) * (4 + sigma)
            b[:, :, 1] -= (4 - sigma) * (3 - sigma) * (4 + sigma)
            b[:, :, 1] -= (4 - sigma) * (2 - sigma) * (4 + sigma)
            b[:, :, 1] += (4 - sigma) * (2 - sigma) * (3 - sigma)
            b[:, :, 1] /= 420

            b[:, :, 2] = -(3 - sigma) * (3 + sigma) * (4 + sigma)
            b[:, :, 2] -= (4 - sigma) * (3 + sigma) * (4 + sigma)
            b[:, :, 2] += (4 - sigma) * (3 - sigma) * (4 + sigma)
            b[:, :, 2] += (4 - sigma) * (3 - sigma) * (3 + sigma)
            b[:, :, 2] /= 280

            b[:, :, 3] = -(2 + sigma) * (3 + sigma) * (4 + sigma)
            b[:, :, 3] += (4 - sigma) * (3 + sigma) * (4 + sigma)
            b[:, :, 3] += (4 - sigma) * (2 + sigma) * (4 + sigma)
            b[:, :, 3] += (4 - sigma) * (2 + sigma) * (3 + sigma)
            b[:, :, 3] /= 420

            b[:, :, 4] = (2 + sigma) * (3 + sigma) * (4 + sigma)
            b[:, :, 4] += (1 + sigma) * (3 + sigma) * (4 + sigma)
            b[:, :, 4] += (1 + sigma) * (2 + sigma) * (4 + sigma)
            b[:, :, 4] += (1 + sigma) * (2 + sigma) * (3 + sigma)
            b[:, :, 4] /= 1680
    return b

# The functions hereafter are not used in current implementation
'''
def tik_conj_grad(num, den, x0, W_op, L_op, rect, maxiter, dim, eps_dv, eps_cg, tol_cg):
    """
    Important:
    num and den are vectors with size dim[0]*...*dim[-1]!
    """
    assert num.size == den.size, "Sizes of numerator and denominator do not match"
    assert len(rect) == 2, "Parts of radii missing"
    assert len(dim) == 2, "Must be 2D"

    args_W = {'dim': dim, 'rect': rect}

    # Normalize numerator and denominator
    num, den = weird_norm(num, den, eps_dv)

    # Initial value of x
    if x0 is None:
        x = np.zeros(num.size)
    else:
        x = x0

    # Vec(diag(den) @ num)
    b = L_op(num, den)

    # (D^T @ D + eps_cg^2 * S^T @ S) @ x
    Ax = L_op(L_op(x, den), den) + (eps_cg ** 2) * W_op(x, args_W)

    # b - Ax
    r = b - Ax
    p = r

    for _ in range(0, maxiter + 1):

        # Calculate A @ p_k
        d = L_op(L_op(p, den), den) + (eps_cg ** 2) * W_op(x, args=args_W)

        # Calculate (r^T @ r) / (p^T @ A @ p)
        alpha = np.dot(r, r) / np.dot(p, d)

        # x_(k+1) = x_k + alpha * p_k
        x += alpha * p

        # r_(k+1) = r_k - alpha * (A @ p_k)
        nr -= alpha * d

        if nr < tol_cg:
            break

        beta = np.dot(nr, nr) / np.dot(r, r)
        p = nr + beta * p
        r = nr
    return x


# Not used
def triangle_2d_lop(x, dim, rect, axes=(0, 1)):
    """
    Calculate HX, H^TX or HXH^T
    Return flatten X.
    """
    if axes not in (0, 1, (0, 1)):
        raise ValueError("Axes to convolve not allowed")

    hor_radius, ver_radius = rect

    X = x.reshape(dim, order='F')

    hx = triangle_kernel(radius=hor_radius)
    hy = triangle_kernel(radius=ver_radius)
    X_smoothed = np.zeros_like(X)

    # Calculate H @ X
    if axes == 0:
        X_smoothed = convolve2d(X, hx[None, :], mode='same', boundary='symm')
    # Calculate H^T @ X
    if axes == 1:
        X_smoothed = convolve2d(
            X_smoothed, hy[:, None], mode='same', boundary='symm')
    # Calculate H @ X @ H^T
    else:
        X_smoothed = convolve2d(X, hx[None, :], mode='same', boundary='symm')
        X_smoothed = convolve2d(
            X_smoothed, hy[:, None], mode='same', boundary='symm')

    X_smoothed = X_smoothed.flatten(order='F')
    return X_smoothed

# Not used


def triangle_kernel(radius=2):
    """
    Create a 1D triangle kernel for preconditioning.
    Should return a 1D Numpy array.
    """
    h = np.zeros(2 * radius + 1)
    for i in range(-radius, radius+1):
        h[i + radius] = 1 - np.abs(i) / radius
    return h / np.sum(h)


# Used to use this
def triangle_2d(x, dim, rect, axes=0):
    """
    For high amplitude erratic noise, only filtering along 
    temporal dimension (axis 0 most of the time) is noteworthy
    """
    if axes not in (0, 1):
        raise ValueError("Axes to convolve not allowed")

    hor_radius, ver_radius = rect

    X = x.reshape(dim, order='F')
    X_smoothed = np.zeros_like(X, dtype=np.float64)

    e0 = 1 / (hor_radius ** 2)
    e1 = 1 / (ver_radius ** 2)

    if axes == 0:
        # Calculate discrete double integration
        # X_smoothed = dis_double_integrate(X, axes=0)
        # Calculate second order derivative in horizontal direction
        X_smoothed = laplace(X, mode='reflect', axes=0)
        X_smoothed *= e0

    if axes == 1:
        # Calculate discrete double integration
        # X_smoothed = dis_double_integrate(X, axes=1)
        # Calculate second order derivative in vertical direction
        X_smoothed = laplace(X, mode='reflect', axes=1)
        X_smoothed *= e1

    return X_smoothed.flatten(order='F')


# Normal CG
def normal_cg(b, a, x0, W_op, L_op, rect, maxiter, dim, eps_dv, eps_cg, tol_cg):

    num = b.flatten(order='F')
    den = a.flatten(order='F')
    ns, nt = dim

    # num, den = weird_norm(b, a, eps_dv)
    den = np.maximum(np.abs(den), eps_cg)

    def mv(x):
        Ap = L_op(x, den)
        ATAp = L_op(Ap, den)

        Lp = eps_cg * W_op(x, dim, rect, adj=False)
        LTLp = eps_cg * W_op(Lp, dim, rect, adj=True)

        out = ATAp + LTLp

        return out

    A = LinearOperator(shape=(ns*nt, ns*nt), matvec=mv, dtype=np.float64)

    x, exit_code = cg(A=A, b=num, x0=x0, rtol=tol_cg,
                      maxiter=maxiter, callback=lambda rk: print("Residual norm:", np.linalg.norm(rk)))

    x = x.reshape(dim, order='F')
    #print(f"x: {x}, exit code: {exit_code}")

    return x


# CG-algorithm with triangle shaping regularization by Fomel, 2007


def shaping_conj_grad(b, A, x0, W_op, L_op, rect, maxiter, dim, eps_dv, eps_cg, tol_cg):
    assert b.size == A.size, "Sizes of numerator and denominator do not match"
    assert len(rect) == 2, "Parts of radii missing"
    assert len(dim) == 2, "Must be 2D"
    # num = b.flatten(order='F')
    # den = A.flatten(order='F')

    num, den = weird_norm(b, A, eps_dv)

    p = np.zeros(num.size)
    m = np.zeros(num.size)
    r = -num

    if x0 is None:
        m = np.zeros(num.size)
    else:
        m = W_op(x0, dim, rect, axes=0)
        r = L_op(m, r)

    rho_hat = 1.
    m_prev = m.copy()

    for i in range(0, maxiter):
        g_m = L_op(r, den) - eps_cg * m  # L^T * r - lambda * m
        g_p = W_op(g_m, dim, rect, adj=True) + \
            eps_cg * p  # H^T @ g_m + lambda * p
        g_m = W_op(g_p, dim, rect, adj=False)  # H @ g_p
        g_r = L_op(g_m, den)  # L * g_m
        rho = np.dot(g_p, g_p)
        if i == 0:
            beta = 0
            rho_0 = rho
            s_p = np.zeros_like(g_p)
            s_m = np.zeros_like(g_m)
            s_r = np.zeros_like(g_r)
        else:
            beta = rho/rho_hat

            if (beta < tol_cg) or (rho/rho_0 < tol_cg):
                m = m.reshape(dim, order='F')
                return m

            s_p = g_p + beta * s_p
            s_m = g_m + beta * s_m
            s_r = g_r + beta * s_r

            alpha = rho / (np.dot(s_r, s_r) + eps_cg *
                           (np.dot(s_p, s_p) - np.dot(s_m, s_m)))

            p -= alpha * s_p
            m -= alpha * s_m
            r -= alpha * s_r

            rho_hat = rho
        """
        print(f"Iter {i:03d}: "
              f"||r||={np.linalg.norm(r):.3e}, "
              f"Δm={np.linalg.norm(m - m_prev):.3e}, "
              f"ρ/ρ₀={(rho/rho_0):.3e}")
        """
        m_prev = m.copy()
    m = m.reshape(dim, order='F')
    return m

def conjgrad2(b, A, x0, W_op, L_op, rect, maxiter, dim, eps_dv, eps_cg, tol_cg):
    #num, den = weird_norm(b, A, eps_dv)

    num = b.flatten(order='F')
    den = A.flatten(order='F')

    p = np.zeros_like(num)
    x = np.zeros_like(num)
    r = -num
    if x0:
        x = x0

    dg = 0
    g0 = 0
    gnp = 0
    r0 = np.dot(r, r)

    for i in range(0, maxiter):
        gp = eps_cg * p
        gx = - eps_cg * x

        gx += L_op(r, den)

        gp += W_op(gx, dim, rect, adj=True)
        gx = W_op(gp.copy(), dim, rect, adj=False)
        gr = L_op(gx, den)

        gn = np.dot(gp, gp)

        if i == 0:
            g0 = gn
            sp = gp
            sx = gx
            sr = gr
        else:
            alpha = gn/gnp
            dg = gn/g0

            if alpha < tol_cg or dg < tol_cg:
                return x

            gp = alpha * sp + gp
            t = sp
            sp = gp
            gp = t

            gx = alpha * sx + gx
            t = sx
            sx = gx
            gx = t

            gr = alpha * sr + gr
            t = sr
            sr = gr
            gr = t

        beta = np.dot(sr, sr) + eps_cg * (np.dot(sp, sp) - np.dot(sx, sx))

        #print(f"Iter: {i+1}, res {np.dot(r, r) / r0}")

        alpha = - gn / beta

        p += alpha * sp
        x += alpha * sx
        r += alpha * sr

        gnp = gn

    x = x.reshape(dim, order='F')
    return x


def weird_norm(b, A, eps_dv):
    """
    Takes vectors num, den and returns normalized vector num, den.
    """
    num = b.copy()
    den = A.copy()
    num = num.flatten(order='F')
    den = den.flatten(order='F')

    # Normalize num and den
    if eps_dv > 0.0:
        norm = 1.0 / np.hypot(den, eps_dv)
        num *= norm
        den *= norm
    norm = np.sqrt(np.mean(den * den))
    if norm == 0.0:
        return np.zeros(num.shape), np.zeros(den.shape)
    norm = np.sqrt(num.size / norm)
    num *= norm
    den *= norm

    return num, den
'''