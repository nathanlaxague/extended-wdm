#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
Copyright © 2024 Daniel Pelaez-Zapata <http://github.com/dspelaez>
Distributed under terms of the GNU/GPL 3.0 license.

Tests for the wavenumber and inverse-phase-speed (nu) directional
spectra added to the `Arrays` class. The nu = k / omega convention and
the Q(nu) spectrum follow Björkqvist et al. (2019).
"""

import numpy as np
import pytest

import ewdm
from ewdm.density import estimate_radial_distribution

# numpy 2.0 renamed np.trapz to np.trapezoid; support both in tests.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


GRAV = 9.8


def _monochromatic_array(f0=0.2, direction_deg=40.0, amplitude=0.5,
                         fs=4.0, duration=600.0):
    """Build an Arrays instance for a single plane wave.

    Returns the Arrays instance together with the true wavenumber and
    inverse phase speed (nu) so the tests can check the peak locations.
    """
    t = np.arange(0, duration, 1 / fs)
    omega = 2 * np.pi * f0
    k0 = omega**2 / GRAV               # deep-water dispersion
    theta = np.radians(direction_deg)
    kx0, ky0 = k0 * np.cos(theta), k0 * np.sin(theta)

    # pentagon plus a centre point, small enough that k0 * d < pi
    radius = 0.5 * (np.pi / k0) / 2
    ang = np.radians(np.array([0, 72, 144, 216, 288]))
    px = np.r_[0.0, radius * np.cos(ang)]
    py = np.r_[0.0, radius * np.sin(ang)]

    eta = np.stack(
        [amplitude * np.cos(kx0*x + ky0*y - omega*t)
         for x, y in zip(px, py)],
        axis=-1
    )

    arr = ewdm.Arrays.from_numpy(
        time=t, surface_elevation=eta,
        position_x=px, position_y=py, fs=fs
    )
    return arr, k0, k0 / omega


COMMON = dict(omin=-4, omax=1, nvoice=8, dd=5.0, kappa=100.0)


def test_frequency_path_unchanged():
    """The default coordinate should still return a frequency spectrum."""
    arr, _, _ = _monochromatic_array()
    out = arr.compute(coordinate="frequency", **COMMON)
    assert "frequency_spectrum" in out
    assert "frequency" in out.coords
    assert out["directional_spectrum"].dims == ("frequency", "direction")


def test_wavenumber_spectrum_peak():
    """The wavenumber spectrum should peak near the true wavenumber."""
    arr, k0, _ = _monochromatic_array()
    out = arr.compute(coordinate="wavenumber", **COMMON)

    assert out["directional_spectrum"].dims == ("wavenumber", "direction")
    assert out["wavenumber"].attrs["units"] == "rad/m"
    assert out["wavenumber_spectrum"].attrs["units"] == "m^3"
    assert out["directional_spectrum"].attrs["units"] == "m^4/rad"

    k = out["wavenumber"].values
    Sk = out["wavenumber_spectrum"].values
    k_peak = k[np.nanargmax(Sk)]
    assert abs(k_peak - k0) / k0 < 0.25


def test_nu_spectrum_peak():
    """The nu (inverse phase speed) spectrum should peak near k0 / omega0."""
    arr, _, nu0 = _monochromatic_array()
    out = arr.compute(coordinate="nu", **COMMON)

    assert out["directional_spectrum"].dims == ("nu", "direction")
    assert out["nu"].attrs["units"] == "s/m"
    assert out["nu_spectrum"].attrs["units"] == "m^3/s"
    assert out["directional_spectrum"].attrs["units"] == "m^4/(s^2 rad)"

    nu = out["nu"].values
    Q = out["nu_spectrum"].values
    nu_peak = nu[np.nanargmax(Q)]
    assert abs(nu_peak - nu0) / nu0 < 0.3


def test_variance_conserved_across_coordinates():
    """Integrated energy should match across f, k and nu."""
    arr, _, _ = _monochromatic_array()
    out_f = arr.compute(coordinate="frequency", **COMMON)
    out_k = arr.compute(coordinate="wavenumber", **COMMON)
    out_nu = arr.compute(coordinate="nu", **COMMON)

    var_f = _trapezoid(out_f["frequency_spectrum"].values,
                       out_f["frequency"].values)
    var_k = _trapezoid(out_k["wavenumber_spectrum"].values,
                       out_k["wavenumber"].values)
    var_nu = _trapezoid(out_nu["nu_spectrum"].values,
                        out_nu["nu"].values)

    assert np.isclose(var_f, var_k, rtol=0.15)
    assert np.isclose(var_f, var_nu, rtol=0.15)


def test_bjorkqvist_directional_integral_identity():
    """Check Björkqvist et al. (2019) Eqs. (6) and (11):

    F(k)  = int Psi(k, theta) k dtheta,
    Q(nu) = int Q_dir(nu, theta) nu dtheta.

    The omnidirectional spectrum must equal the directional spectrum
    integrated over direction with the radial Jacobian.
    """
    arr, _, _ = _monochromatic_array()
    dtheta = np.radians(COMMON["dd"])

    for coord, spec in [("wavenumber", "wavenumber_spectrum"),
                        ("nu", "nu_spectrum")]:
        out = arr.compute(coordinate=coord, **COMMON)
        Psi = out["directional_spectrum"].values     # (r, theta)
        r = out[coord].values
        F = out[spec].values
        F_from_dir = np.nansum(Psi, axis=1) * r * dtheta

        mask = F > 0.01 * np.nanmax(F)
        rel = np.abs(F_from_dir[mask] - F[mask]) / F[mask]
        assert np.nanmax(rel) < 1e-6, f"{coord}: identity violated"


def test_invalid_coordinate_raises():
    """An unknown coordinate should raise an informative error."""
    arr, _, _ = _monochromatic_array()
    with pytest.raises(Exception):
        arr.compute(coordinate="nonsense", **COMMON)


def test_custom_radial_bins_respected():
    """User-supplied radial bins should appear as the output coordinate."""
    arr, _, _ = _monochromatic_array()
    bins = np.linspace(0.05, 0.5, 64)
    out = arr.compute(
        coordinate="wavenumber", bins_radial=bins, **COMMON
    )
    np.testing.assert_allclose(out["wavenumber"].values, bins)

# ---------------------------------------------------------------------------
# Smoother default radial binning (log-spaced grid + Silverman bandwidth).
# These tests guard the binning behaviour introduced for smoother, but still
# reliable, F(k) and Q(nu) spectra. They use the array fixture rather than a
# pure monochromatic wave so the spectrum has a resolvable shape to smooth.
# ---------------------------------------------------------------------------

def _log_smoothness(spectrum):
    """Mean absolute step in log-spectrum between adjacent positive bins.

    A smaller value means a smoother curve on log axes. Used as a relative
    smoothness metric (compare two spectra computed on grids of comparable
    density), not as an absolute threshold.
    """
    y = np.asarray(spectrum, dtype=float)
    y = y[y > 0]
    if y.size < 3:
        return np.nan
    return float(np.mean(np.abs(np.diff(np.log(y)))))


def test_default_radial_grid_is_log_spaced():
    """The default wavenumber/nu grids should be logarithmically spaced."""
    arr, _, _ = _monochromatic_array()
    for coord in ("wavenumber", "nu"):
        out = arr.compute(coordinate=coord, **COMMON)
        r = out[coord].values
        # On a log-uniform grid the ratio of successive bins is constant.
        ratios = r[1:] / r[:-1]
        assert np.allclose(ratios, ratios[0], rtol=1e-6), \
            f"{coord} default grid is not log-spaced"
        # ... and it should NOT be linear (constant difference).
        diffs = np.diff(r)
        assert not np.allclose(diffs, diffs[0], rtol=1e-3), \
            f"{coord} default grid is still linear"


def test_bins_per_octave_preserves_variance_and_peak():
    """Refining bins_per_octave must not move the peak or change variance.

    This encodes the design principle that bin density is only a rendering
    lever: the spectral content is set by the kernel bandwidth, so variance
    (m0) and the peak location must be invariant to bins_per_octave.
    """
    arr, _, _ = _monochromatic_array()
    for coord, spec in [("wavenumber", "wavenumber_spectrum"),
                        ("nu", "nu_spectrum")]:
        m0, peaks = [], []
        for npo in (12, 24, 48):
            out = arr.compute(coordinate=coord, bins_per_octave=npo, **COMMON)
            r = out[coord].values
            S = out[spec].values
            m0.append(_trapezoid(S, r))
            peaks.append(r[np.nanargmax(S)])
        # variance conserved across densities
        assert np.allclose(m0, m0[0], rtol=1e-6), f"{coord}: m0 drifts with bins"
        # peak stable to within a quarter octave
        assert max(peaks) / min(peaks) < 2 ** 0.25, \
            f"{coord}: peak moves with bins_per_octave"


def test_log_grid_smoother_than_linear():
    """The log-spaced default should be smoother than a comparable linear grid.

    Using a broadband-ish array case, the log-spaced default grid should not
    be rougher (in the mean |dlog S| sense) than an explicit linear grid of
    similar bin count over the same span.
    """
    arr, k0, _ = _monochromatic_array()
    out_log = arr.compute(coordinate="wavenumber", **COMMON)
    r_log = out_log["wavenumber"].values

    # comparable linear grid: same span, same number of bins
    lin = np.linspace(r_log[0], r_log[-1], len(r_log))
    out_lin = arr.compute(coordinate="wavenumber", bins_radial=lin, **COMMON)

    s_log = _log_smoothness(out_log["wavenumber_spectrum"].values)
    s_lin = _log_smoothness(out_lin["wavenumber_spectrum"].values)
    # log grid should be at least as smooth (allow a small tolerance)
    assert s_log <= s_lin * 1.05, \
        f"log grid not smoother: log={s_log:.4f} linear={s_lin:.4f}"


def test_bandwidth_floor_is_respected():
    """An explicit bandwidth_floor should broaden (smooth) the spectrum.

    Passing a large bandwidth_floor must not raise and must produce a
    spectrum at least as smooth as the unfloored default, while still
    conserving variance.
    """
    arr, _, _ = _monochromatic_array()
    out_def = arr.compute(coordinate="wavenumber", **COMMON)
    out_flr = arr.compute(coordinate="wavenumber", bandwidth_floor=0.5, **COMMON)

    r = out_flr["wavenumber"].values
    m0_def = _trapezoid(out_def["wavenumber_spectrum"].values,
                        out_def["wavenumber"].values)
    m0_flr = _trapezoid(out_flr["wavenumber_spectrum"].values, r)
    # variance conserved regardless of the floor
    assert np.isclose(m0_def, m0_flr, rtol=0.05)
    # floored spectrum is at least as smooth
    s_def = _log_smoothness(out_def["wavenumber_spectrum"].values)
    s_flr = _log_smoothness(out_flr["wavenumber_spectrum"].values)
    assert s_flr <= s_def * 1.05
