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
