#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
Wavenumber and inverse phase speed spectra
===========================================

The directional wave spectrum is most often expressed as a function of frequency
and direction, :math:`E(f,\\theta)`. For spatial arrays, however, `ewdm.Arrays`
solves the local wavenumber vector directly from the spatial phase gradients, so
the spectrum can equally be expressed as a function of the wavenumber magnitude
:math:`k` or the inverse phase speed :math:`\\nu = k / \\omega`. These are simply
different parametrisations of the same directional variance, and in `ewdm` they
are selected through the `coordinate` argument of the `compute` method.

The wavenumber spectrum follows the polar convention

    .. math:: F(k) = \\int \\Psi(k,\\theta)\\, k\\, d\\theta

while the inverse phase speed spectrum, after Björkqvist et al. (2019), is

    .. math:: Q(\\nu) = \\int Q(\\nu,\\theta)\\, \\nu\\, d\\theta

In this example we use the Donelan run082 test case (see Donelan et al. 2015) and
compute the directional spectrum in all three coordinates. We also build the joint
wavenumber-frequency spectrum :math:`E(k,f)` and compare it with the linear
dispersion relation and the bound second harmonic.
"""

import numpy as np
import xarray as xr

from scipy.io import loadmat
from matplotlib import pyplot as plt

import ewdm
from ewdm.plots import plot_directional_spectrum

# loading matlab file
mat_fname = "../../data/donelan_run82.mat"
mat_data = loadmat(mat_fname, simplify_cells=True)

sampling_rate = mat_data["fs"]
time = np.arange(len(mat_data["eta"])) / sampling_rate
elements = np.arange(len(mat_data["x"]))

# creating dataset
dataset = xr.Dataset(
    data_vars = {
        "surface_elevation": (["time", "element"], mat_data["eta"]),
        "position_x": ("element", mat_data["x"]),
        "position_y": ("element", mat_data["y"])
    },
    coords = {"time": time, "element": elements},
    attrs = {"sampling_rate": sampling_rate}
)
print(dataset)

# %%
# Frequency-direction spectrum (baseline)
# ---------------------------------------
#
# The default `coordinate="frequency"` reproduces the familiar frequency-direction
# spectrum `E(f, theta)`.
spec = ewdm.Arrays(dataset)
output_f = spec.compute(cross_wavelet=True, solver="least-squares", kappa=36)
print(output_f)

fig, ax = plt.subplots(figsize=(6,5))
plot_directional_spectrum(
    output_f.directional_spectrum, ax=ax, levels=None, colorbar=True,
    axes_kw={"rmin": 0.1, "rmax": 1.2, "rstep": 0.2, "angle": 135},
    cbar_kw={"label": "$E(f,\\theta)$ [m$^2$/Hz/deg]"}
)
fig.subplots_adjust(right=0.80)

# %%
# Wavenumber-direction spectrum
# -----------------------------
#
# Setting `coordinate="wavenumber"` maps the same variance onto a wavenumber axis,
# producing `Psi(k, theta)` and the omnidirectional `F(k)`. The radial grid is
# log-spaced by default (roughly 1/12 to 9 rad/m). We reuse `plot_directional_spectrum`
# with `frqs="wavenumber"` so the polar radius is the wavenumber coordinate.
output_k = spec.compute(
    cross_wavelet=True, solver="least-squares", kappa=36,
    coordinate="wavenumber"
)
print(output_k)

fig, ax = plt.subplots(figsize=(6,5))
plot_directional_spectrum(
    output_k.directional_spectrum, frqs="wavenumber", ax=ax, levels=None,
    colorbar=True,
    axes_kw={"rmin": 0.5, "rmax": 4.0, "rstep": 1.0, "angle": 135},
    cbar_kw={"label": "$\\Psi(k,\\theta)$ [m$^4$/rad]"}
)
ax.set_ylabel("$k$ [rad/m]")
fig.subplots_adjust(right=0.80)

# %%
# Inverse-phase-speed (nu) spectrum
# ---------------------------------
#
# Setting `coordinate="nu"` yields the inverse-phase-speed spectrum `Q(nu, theta)`
# and `Q(nu)` of Björkqvist et al. (2019), where `nu = k / omega` has units of s/m.
# The default radial grid spans roughly 1/50 to 2 s/m.
output_nu = spec.compute(
    cross_wavelet=True, solver="least-squares", kappa=36,
    coordinate="nu"
)
print(output_nu)

fig, ax = plt.subplots(figsize=(6,5))
plot_directional_spectrum(
    output_nu.directional_spectrum, frqs="nu", ax=ax, levels=None,
    colorbar=True,
    axes_kw={"rmin": 0.1, "rmax": 1.0, "rstep": 0.2, "angle": 135},
    cbar_kw={"label": "$Q(\\nu,\\theta)$ [m$^4$/(s$^2$ rad)]"}
)
ax.set_ylabel("$\\nu$ [s/m]")
fig.subplots_adjust(right=0.80)

# %%
# Wavenumber-frequency spectrum
# -----------------------------
#
# Since the array solves a wavenumber for every frequency and time, we can also
# build the joint wavenumber-frequency spectrum :math:`E(k,f)`. We reconstruct the
# local wavenumber and the wavelet power from the same building blocks used
# internally by `compute`, and map the variance onto a wavenumber grid using the
# radial kernel of `ewdm.density`. The energy follows the linear dispersion
# relation, with a weaker signature along the bound second-harmonic curve at half
# the free-wave wavenumber.
from ewdm.parameters import GRAV
from ewdm.density import _gaussian_radial_kde
from ewdm.plots import _get_cmap

# reconstruct the per-(frequency, time) wavenumber and wavelet power
coeffs = spec.wavelet_coefficients(dataset)
dx = spec.array_geometry(dataset)
dphi = spec.phase_differences(coeffs, cross_wavelet=True)
dphi = (dphi - np.pi) % (2 * np.pi) - np.pi
kx, ky, _ = spec.compute_wavenumbers(dx, dphi, solver="least-squares")
wavenumber = np.hypot(kx, ky)                       # (frequency, time)
power = (np.abs(coeffs) ** 2).mean("element").data  # (frequency, time)

# map the variance onto a wavenumber grid, frequency by frequency. The
# per-frequency wavenumber samples are smoothed with the radial kernel of
# `ewdm.density` into the distribution P(k|f), normalised to unit area over k.
# The wavenumber-frequency spectral density is then E(k,f) = S(f) P(k|f), in
# m^3/Hz, so that the variance is preserved: int int E(k,f) dk df = m0.
freqs = spec.freqs
Sf = output_f["frequency_spectrum"].data          # frequency spectrum, m^2/Hz
kbins = np.linspace(0.05, 6.0, 240)
Pkf = np.array([
    _gaussian_radial_kde(wavenumber[i], kbins, weights=power[i])
    for i in range(len(freqs))
])
Ekf = Sf[:, None] * Pkf   # per-magnitude spectral density, m^3/Hz

# linear dispersion and bound second harmonic (deep water)
k_linear = (2 * np.pi * freqs) ** 2 / GRAV
k_bound = k_linear / 2

# express the spectral density in decibels relative to 1 m^3/Hz
Ekf_db = 10 * np.log10(Ekf, where=Ekf > 0, out=np.full_like(Ekf, np.nan))
vmax = np.nanmax(Ekf_db)

fig, ax = plt.subplots(figsize=(6,5))
pc = ax.pcolormesh(
    kbins, freqs, Ekf_db, cmap=_get_cmap(), shading="gouraud",
    vmin=vmax - 25, vmax=vmax
)
ax.plot(k_linear, freqs, color="r", lw=1.5, label="linear dispersion")
ax.plot(k_bound, freqs, color="r", ls="--", lw=1.5, label="bound 2nd harmonic")
ax.set(
    xlim=(0, 6), ylim=(0, 1.5), xlabel="wavenumber [rad/m]",
    ylabel="frequency [Hz]"
)
ax.legend(loc="lower right")
fig.colorbar(pc, ax=ax, label="spectral density [dB rel. 1 m$^3$/Hz]")

# %%
# Omnidirectional spectra
# -----------------------
#
# Each `compute` call also returns the omnidirectional spectrum integrated over
# direction. The three representations carry the same total variance `m0`; only
# the independent variable changes.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11,3.5), layout="constrained")

ax1.plot(output_f["frequency"], output_f["frequency_spectrum"])
ax1.set(xscale="log", yscale="log",
        xlabel="frequency [Hz]", ylabel="$F(f)$ [m$^2$/Hz]")

ax2.plot(output_k["wavenumber"], output_k["wavenumber_spectrum"])
kmax = np.pi / np.hypot(dx[:, 0], dx[:, 1]).max()
ax2.axvline(kmax, color="0.5", ls="--", lw=1, label="$k_{max}=\\pi/b_{max}$")
ax2.legend(loc="lower left", fontsize="small")
ax2.set(xscale="log", yscale="log",
        xlabel="wavenumber [rad/m]", ylabel="$F(k)$ [m$^3$]")

ax3.plot(output_nu["nu"], output_nu["nu_spectrum"])
ax3.set(xscale="log", yscale="log",
        xlabel="inverse phase speed [s/m]", ylabel="$Q(\\nu)$ [m$^3$/s]")

# %%
# .. note::
#
#    The wavenumber spectrum :math:`F(k)` is only meaningful below the dashed grey
#    line at :math:`k_{max}=\pi/b_{max}`, where :math:`b_{max}` is the largest
#    element separation in the array. Beyond this cutoff the array can no longer
#    resolve the phase differences between elements, so the steep roll-off of
#    :math:`F(k)` (and the high-:math:`\nu` tail of :math:`Q(\nu)`) is an artefact
#    of the array resolution rather than a feature of the wave field.
