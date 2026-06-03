#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8


import numpy as np
import xarray as xr
import logging

from .parameters import VARIABLE_NAMES

# numpy 2.0 renamed `np.trapz` to `np.trapezoid` and removed the old
# name. Use the new name when available and fall back on older numpy.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


# von mises kernel desnsity estimation
def _vonmises_kde(arr, bins, kappa, weights=None):
    """Return Kernel-Density Estimation using von Mises distribution.

    When ``weights`` is given (per-sample, e.g. wavelet power), each kernel is
    weighted by it before summing, so the estimate is the variance-weighted
    distribution (the canonical WDM construction) rather than the unweighted
    sample occurrence. Normalised to unit integral over ``bins`` either way."""

    m = np.isfinite(arr)
    a = arr[m]
    w = None if weights is None else np.asarray(weights)[m]

    # define input parameters
    x = np.radians(bins[:, None] - a[None, :])

    # integrate (optionally weighted) vonmises kernels
    ker = np.exp(kappa * np.cos(x))
    if w is not None:
        ker = ker * w[None, :]
    kde = ker.sum(axis=1) / (2 * np.pi * np.i0(kappa))
    kde /= _trapezoid(kde, x=bins)
    return kde


def _gaussian_kde(arr, bins, bandwidth='silverman'):
    """Return Kernel-Density Estimation using Gaussian distribution"""

    if bandwidth == 'silverman':
        bandwidth = 1.06 * arr.std() * len(arr) ** (-1./5.)

    x = (bins[:, None] - arr[None, :]) / bandwidth
    fac = 1 / np.sqrt(2 * np.pi)

    kde = (fac * np.exp(-0.5 * x**2)).sum(axis=1)
    return kde / (len(arr) * bandwidth)


# function to get histogram along freq axis
def _get_density(arr, bins, kappa, weights=None):
    if np.isnan(arr).all():
        return np.zeros_like(bins, dtype="float") * np.nan
    else:
        if kappa is not None:
            return _vonmises_kde(arr, bins=bins, kappa=kappa, weights=weights)
        else:
            m = np.isfinite(arr)
            w = None if weights is None else np.asarray(weights)[m]
            bins_edges = np.r_[bins, bins[-1]+np.diff(bins)[0]]
            return np.histogram(arr[m], bins=bins_edges, weights=w,
                                density=True)[0]


# function to actually estimate the spectrum
def estimate_directional_distribution(power, theta, dd, kappa,
                                      power_weighted=False):
    """Construct directional distribution function from local wave directions.

    power_weighted (bool): if True, weight each (frequency, time) direction by
    its wavelet power when forming D(f, theta) -- i.e. bin the variance, the
    canonical WDM construction -- instead of the unweighted sample occurrence.
    Leaves the frequency spectrum S(f) = <power>_t unchanged.
    """

    # array of directiontions where dd is the directional resolution
    bins = np.arange(-180, 180, dd)

    # directional distribution function
    if power_weighted:
        th = np.asarray(theta); pw = np.asarray(power)
        D = np.stack([_get_density(th[f], bins, kappa, weights=pw[f])
                      for f in range(th.shape[0])])
    else:
        D = np.apply_along_axis(
            _get_density, arr=theta, bins=bins, kappa=kappa, axis=1
        )

    # determine average wavelet power
    S = power.mean("time").data

    # array containing directional wave spectra
    E = S[:,None] * D

    # return dataset
    output_ds = xr.Dataset(
        data_vars = {
            "directional_spectrum": (["frequency", "direction"], E.data),
            "directional_distribution": (["frequency", "direction"], D.data),
            "frequency_spectrum": (["frequency"], S)
        },
        coords = {
            "frequency": power["frequency"].data,
            "direction": bins,
        }
    )
    output_ds["frequency"].attrs = VARIABLE_NAMES["frequency"]
    output_ds["direction"].attrs = VARIABLE_NAMES["direction"]
    for var in output_ds:
        output_ds[var].attrs = VARIABLE_NAMES[var]

    return output_ds


# kernel density estimation along a radial (non-periodic) axis {{{
def _gaussian_radial_kde(arr, bins, bandwidth="silverman", bandwidth_floor=None,
                         weights=None):
    """Return Gaussian Kernel-Density Estimation along a radial axis.

    Unlike the directional case, the radial coordinate (wavenumber or
    inverse phase speed) is not periodic, so a plain Gaussian kernel is
    used instead of the von Mises kernel. The estimate is normalised so
    that it integrates to unity over `bins`.

    Args:
        arr (np.ndarray): Sample values along the radial axis.
        bins (np.ndarray): Radial bin centers where the density is
            evaluated.
        bandwidth (str or float): Either a number giving the kernel
            bandwidth or the string ``silverman`` to use Silverman's
            rule of thumb (default).
        bandwidth_floor (float, optional): Optional physical lower bound on
            the kernel bandwidth, in the units of the radial coordinate.
            When given, the bandwidth is never allowed to fall below this
            value. Intended for callers who want to impose a known
            resolution limit; note this should be a *kernel width*, not an
            aperture wavelength such as ``2*pi / L_baseline`` (which is far
            wider than a sensible kernel and over-smooths). If None, the
            data-driven (Silverman) bandwidth is used and only floored at
            the local bin spacing to keep the kernel resolvable for
            degenerate, near-zero-spread samples.

    Returns:
        np.ndarray: Normalised density evaluated at `bins`.
    """

    m = np.isfinite(arr)
    arr = arr[m]
    w = None if weights is None else np.asarray(weights)[m]

    if bandwidth == "silverman":
        bandwidth = 1.06 * arr.std() * len(arr) ** (-1. / 5.)

    # the Silverman bandwidth collapses to zero when the radial samples
    # have little or no spread (e.g. a near-monochromatic sea state). In
    # that case the Gaussian kernel becomes a delta function that misses
    # the discrete `bins` grid. We floor the bandwidth so the kernel stays
    # resolvable. An explicit `bandwidth_floor` (a kernel width) may be
    # supplied by the caller to impose a known resolution limit; otherwise
    # we fall back to the local bin spacing, which keeps the kernel
    # resolvable without biasing the well-sampled, broadband case.
    if bandwidth_floor is not None and np.isfinite(bandwidth_floor):
        bandwidth = max(bandwidth, float(bandwidth_floor))
    else:
        bin_spacing = np.median(np.diff(bins)) if len(bins) > 1 else 1.0
        bandwidth = max(bandwidth, bin_spacing)

    # guard against a degenerate (zero-spread) sample
    if bandwidth == 0 or not np.isfinite(bandwidth):
        return np.zeros_like(bins, dtype="float")

    x = (bins[:, None] - arr[None, :]) / bandwidth
    fac = 1 / np.sqrt(2 * np.pi)
    ker = fac * np.exp(-0.5 * x**2)
    if w is not None:
        ker = ker * w[None, :]
    kde = ker.sum(axis=1) / (len(arr) * bandwidth)

    # normalise to unit integral over the radial bins. np.trapezoid was
    # introduced in numpy 2.0 as the replacement for np.trapz; fall back
    # to the old name on numpy 1.x.
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = _trapz(kde, x=bins)
    if area > 0:
        kde /= area
    return kde


def _relative_radial_kde(arr, bins, bandwidth, weights=None):
    """Gaussian KDE in ``log`` radial coordinate: a constant *relative*
    (fractional) kernel width, so the smoothing is a fixed percentage of the
    radial value rather than a fixed absolute width. This avoids the absolute
    kernel over-smoothing the steep short-wave (high radial) tail. ``bandwidth``
    is the fractional log-space standard deviation (e.g. 0.08). Normalised to
    unit integral over the linear ``bins``. ``weights`` (per-sample) weights
    each kernel before summing (variance-weighted distribution)."""
    m = np.isfinite(arr) & (arr > 0)
    a = arr[m]
    w = None if weights is None else np.asarray(weights)[m]
    if a.size == 0:
        return np.zeros_like(bins, dtype="float")
    x = (np.log(bins)[:, None] - np.log(a)[None, :]) / bandwidth
    ker = np.exp(-0.5 * x**2)
    if w is not None:
        ker = ker * w[None, :]
    kde = ker.sum(axis=1) / bins
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = _trapz(kde, x=bins)
    return kde / area if area > 0 else kde


def _histogram_radial(arr, bins, weights=None):
    """Raw (un-smoothed) density: a histogram of the samples into bins whose
    edges are the mid-points of ``bins``, divided by bin width and normalised to
    unit integral. ``weights`` (per-sample, e.g. power) gives the variance-
    weighted histogram. Use to inspect the bare measured-radial distribution
    without any kernel smoothing."""
    m = np.isfinite(arr) & (arr > 0)
    a = arr[m]
    w = None if weights is None else np.asarray(weights)[m]
    if a.size == 0:
        return np.zeros_like(bins, dtype="float")
    mids = 0.5 * (bins[:-1] + bins[1:])
    edges = np.concatenate(([bins[0] - (mids[0] - bins[0])], mids,
                            [bins[-1] + (bins[-1] - mids[-1])]))
    dens = np.histogram(a, bins=edges, weights=w)[0] / np.diff(edges)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = _trapz(dens, x=bins)
    return dens / area if area > 0 else dens


# function to get radial density along the frequency axis
def _get_radial_density(arr, bins, bandwidth, bandwidth_floor=None,
                        bandwidth_mode="absolute", weights=None):
    if np.isnan(arr).all():
        return np.zeros_like(bins, dtype="float") * np.nan
    if bandwidth_mode == "relative":
        return _relative_radial_kde(arr, bins=bins, bandwidth=bandwidth,
                                    weights=weights)
    elif bandwidth_mode == "histogram":
        return _histogram_radial(arr, bins=bins, weights=weights)
    return _gaussian_radial_kde(arr, bins=bins, bandwidth=bandwidth,
                                bandwidth_floor=bandwidth_floor, weights=weights)
# }}}


# estimate spectrum on a (radial, direction) grid {{{
def estimate_radial_distribution(
        power, theta, radial, radial_name, bins_radial,
        dd, kappa, bandwidth="silverman", bandwidth_floor=None,
        bandwidth_mode="absolute", power_weighted=False
    ):
    """Construct a directional spectrum on a (radial, direction) grid.

    This is the wavenumber/inverse-phase-speed analogue of
    :func:`estimate_directional_distribution`. The energy contained in
    each frequency band, ``S(f) = <power>_t``, is redistributed onto a
    radial axis (wavenumber ``k`` or inverse phase speed ``nu = k /
    omega``, following Björkqvist et al., 2019) according to the local
    radial estimates obtained per ``(frequency, time)`` sample, and onto
    the directional axis using the same von Mises kernel as the
    frequency-direction path. The mapping conserves variance, i.e.
    ``sum E(r, theta) dr dtheta == sum S(f) df`` up to the discretisation
    of the kernels.

    Args:
        power (xr.DataArray): Wavelet power with dims ``(frequency,
            time)``, identical to the array used by
            :func:`estimate_directional_distribution`.
        theta (xr.DataArray): Local wave direction in degrees with dims
            ``(frequency, time)``.
        radial (xr.DataArray): Local radial coordinate (e.g. wavenumber
            in rad/m, or inverse phase speed ``nu`` in s/m) with dims
            ``(frequency, time)``.
        radial_name (str): Name of the radial coordinate. It must be a
            key in :data:`ewdm.parameters.VARIABLE_NAMES` and is used to
            label the output dimension and the radial spectrum. For
            ``nu`` the radial spectrum follows the ``Q(nu)`` convention
            of Björkqvist et al. (2019).
        bins_radial (np.ndarray): Bin centres for the radial axis.
        dd (float): Directional resolution in degrees.
        kappa (float): Smoothness parameter for the von Mises kernel used
            along the directional axis.
        bandwidth (str or float): Bandwidth for the Gaussian kernel used
            along the radial axis. Default is Silverman's rule.
        bandwidth_floor (float, optional): Physical lower bound on the
            radial-kernel bandwidth (e.g. the array wavenumber resolution
            ``2*pi / L_baseline`` for wavenumber, or its slowness
            equivalent for ``nu``). Prevents the smoothed spectrum from
            implying structure finer than the array can resolve. When
            None, the kernel is floored at the local bin spacing instead.
            Only used when ``bandwidth_mode == "absolute"``.
        bandwidth_mode (str): How ``bandwidth`` is interpreted along the
            radial axis. ``"absolute"`` (default) is a fixed-width Gaussian
            kernel in the radial coordinate (Silverman or a number).
            ``"relative"`` is a Gaussian in ``log`` radial with a constant
            *fractional* width (``bandwidth`` is the fractional log-space
            sigma, e.g. 0.08), which smooths a fixed percentage of k/nu and
            so preserves the slope of a steep short-wave tail. ``"histogram"``
            applies no smoothing (raw weighted histogram), exposing the bare
            measured-radial distribution.
        power_weighted (bool): if True, weight each (frequency, time) radial and
            directional sample by its wavelet power when forming R(f, r) and
            D(f, theta) -- i.e. bin the variance (the canonical WDM
            construction) rather than the unweighted sample occurrence. The
            low-power high-radial scatter is then no longer over-counted, which
            preserves the steep short-wave (saturation) slope. The total
            variance and S(f) are unchanged.

    Returns:
        xr.Dataset: Dataset containing the directional spectrum
        ``directional_spectrum`` on the ``(radial_name, direction)``
        grid, the radial spectrum ``<radial_name>_spectrum`` (``Q`` when
        ``radial_name == "nu"``), and the directional distribution
        ``directional_distribution``.
    """

    # directional and radial bin arrays
    bins_dir = np.arange(-180, 180, dd)

    # directional D(f, theta) and radial R(f, r) distribution functions.
    # When power_weighted, each (f, t) sample is weighted by its wavelet power
    # (variance binning) -- looped per frequency so radial/theta and the weights
    # stay paired; otherwise the vectorised apply_along_axis path is used.
    th = np.asarray(theta); rad = np.asarray(radial)
    if power_weighted:
        pw = np.asarray(power)
        D = np.stack([_get_density(th[f], bins_dir, kappa, weights=pw[f])
                      for f in range(th.shape[0])])
        R = np.stack([_get_radial_density(rad[f], bins_radial, bandwidth,
                                          bandwidth_floor, bandwidth_mode,
                                          weights=pw[f])
                      for f in range(rad.shape[0])])
    else:
        D = np.apply_along_axis(
            _get_density, arr=th, bins=bins_dir, kappa=kappa, axis=1
        )
        R = np.apply_along_axis(
            _get_radial_density, arr=rad, bins=bins_radial,
            bandwidth=bandwidth, bandwidth_floor=bandwidth_floor,
            bandwidth_mode=bandwidth_mode, axis=1
        )

    # average wavelet power per frequency, S(f). This is the same
    # quantity used by the frequency-direction path, so the absolute
    # scaling is inherited unchanged.
    S = power.mean("time").data

    # total variance per frequency band, S(f) df. Summed over frequency
    # this equals the sea surface variance (m0).
    df = np.gradient(power["frequency"].data)
    Sdf = S * df

    # Energy density on the (radial, direction) grid. D is a density that
    # integrates to one over direction in *degrees* (sum D dd == 1) and R
    # integrates to one over the radial axis (int R dr == 1). Hence
    #     rho(r, theta) = sum_f S(f) df R(f, r) D(f, theta)
    # satisfies  sum_r sum_theta rho dr dd == sum_f S df == m0, i.e. it
    # conserves the total variance.
    rho = np.einsum("f,fr,fd->rd", Sdf, R, D)

    # Omnidirectional radial spectrum, F(k) (m^3) or Q(nu) (m^3/s),
    # following Björkqvist et al. (2019). Integrating the energy density
    # over direction recovers the omnidirectional spectrum such that
    # int F(k) dk == int Q(nu) dnu == m0. This is equivalent to their
    # Eqs. (6) and (11), int Psi(r, theta) r dtheta, once the radial
    # Jacobian is reintroduced below.
    radial_spectrum = rho.sum(axis=1) * dd

    # Directional spectrum following the polar convention of Björkqvist
    # et al. (2019, Eqs. 6 and 11), in which the radial Jacobian is
    # factored out of the *directional* spectrum:
    #     Psi(k, theta)    [m^4/rad]      with  F(k)  = int Psi k dtheta
    #     Q_dir(nu, theta) [m^4/(s^2 rad)] with Q(nu) = int Q_dir nu dtheta
    # We convert the degree-based density to a per-radian density and
    # divide by the radial coordinate, guarding against r = 0.
    r = bins_radial.astype("float")
    r_safe = np.where(r == 0, np.nan, r)
    deg_per_rad = 180.0 / np.pi
    directional_spectrum = (rho * deg_per_rad) / r_safe[:, None]

    # return dataset
    output_ds = xr.Dataset(
        data_vars = {
            "directional_spectrum": (
                [radial_name, "direction"], directional_spectrum
            ),
            "directional_distribution": (
                ["frequency", "direction"], D
            ),
            f"{radial_name}_spectrum": ([radial_name], radial_spectrum),
        },
        coords = {
            "frequency": power["frequency"].data,
            radial_name: bins_radial,
            "direction": bins_dir,
        }
    )
    for coord in ("frequency", "direction", radial_name):
        if coord in VARIABLE_NAMES:
            output_ds[coord].attrs = VARIABLE_NAMES[coord]
    for var in output_ds.data_vars:
        if var in VARIABLE_NAMES:
            output_ds[var].attrs = VARIABLE_NAMES[var]

    # The directional spectrum carries coordinate-specific units in the
    # polar convention of Björkqvist et al. (2019): Psi(k, theta) is in
    # m^4/rad and Q_dir(nu, theta) is in m^4/(s^2 rad). Override the
    # generic metadata accordingly.
    _dir_units = {
        "wavenumber": "m^4/rad",
        "nu": "m^4/(s^2 rad)",
    }
    if radial_name in _dir_units:
        output_ds["directional_spectrum"].attrs = {
            "standard_name": f"sea_surface_{radial_name}_directional_wave_spectrum",
            "long_name": (
                "Directional wavenumber wave spectrum Psi(k, theta)"
                if radial_name == "wavenumber"
                else "Directional inverse-phase-speed wave spectrum Q(nu, theta)"
            ),
            "units": _dir_units[radial_name],
        }

    return output_ds
# }}}
