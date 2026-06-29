import numpy as np
from scipy.signal import convolve2d
from IDF import *
from SOMF_slope import *
from SOMF_flatten import *
import pytest

def test_triangle_kernel_basic():
    """Test that kernel sums to 1 and is symmetric."""
    radius = 3
    h = triangle_kernel(radius=radius)

    # Check length
    assert len(h) == 2 * radius + 1

    # Check normalization (sum should be 1)
    np.testing.assert_allclose(np.sum(h), 1.0)

    # Check symmetry
    np.testing.assert_allclose(h, h[::-1])


@pytest.mark.parametrize("radius", [1, 2, 5])
def test_triangle_kernel_shape_and_values(radius):
    """Test kernel shape and expected peak."""
    h = triangle_kernel(radius=radius)

    # Peak should be at center
    center_index = radius
    assert np.isclose(h[center_index], max(h))

    # Values should decrease linearly away from center
    diffs = np.diff(h[:center_index + 1])
    assert np.all(diffs <= 0)


def test_triangle_2d_lop_both_modes():
    """Test horizontal and vertical convolution modes."""
    X = np.arange(16).astype(float)

    args = {
        'dim': (4, 4),
        'rect': (1, 1)
    }

    out_hor = triangle_2d_lop(X.copy(), args, mode='hor')
    out_ver = triangle_2d_lop(X.copy(), args, mode='ver')
    out_both = triangle_2d_lop(X.copy(), args, mode='both')

   # Output must have same size as input (flattened)
    assert out_hor.shape == X.shape
    assert out_ver.shape == X.shape
    assert out_both.shape == X.shape

   # Both smoothing operations should reduce variance compared to original data.
    def var(x): return np.var(x.reshape(args['dim'], order='F'))
    assert var(out_both) < var(X)


def test_invalid_mode():
    """Ensure invalid mode raises ValueError."""
    X = np.zeros(9)
    args = {'dim': (3, 3), 'rect': (1, 1)}

    with pytest.raises(ValueError):
        triangle_2d_lop(X, args, mode='diagonal')


def test_consistency_between_modes():
    """Check that 'both' behaves like sequential hor+ver."""
    X = np.random.rand(9)
    args = {'dim': (3, 3), 'rect': (1, 1)}

    both_result = triangle_2d_lop(X.copy(), args, mode='both')

    hor_then_ver_result = triangle_2d_lop(
        triangle_2d_lop(X.copy(), args, mode='hor'),
        args,
        mode='ver'
    )

    np.testing.assert_allclose(both_result, hor_then_ver_result)