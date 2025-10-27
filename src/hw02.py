###############################################
# IMPORTS AND SETUP
###############################################
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

np.set_printoptions(suppress=True)
# plt.rcParams["text.usetex"] = True

###############################################
# CONSTANTS
###############################################
mu = 3.986004418e5  # km^3/s^2 (gravitational parameter for Earth)
omega_EN = 7.2911e-5  # rad/s (Earth rotation)
R_E = 6378.0  # km (spherical Earth)

# Problem 2
X_OE_0 = np.array(
    [
        7000.0,  # a [km]
        0.05,  # e
        np.deg2rad(45.0),  # i [rad]
        np.deg2rad(30.0),  # omega [rad]
        np.deg2rad(60.0),  # Omega [rad]
        np.deg2rad(0.0),  # true anomaly f [rad]
    ]
)

# Problem 3 + 4
station_ll = (np.deg2rad(35.2967), np.deg2rad(-116.9141))  # lat, lon [rad]

X1 = np.array([R_E + 420.0, 0.007, np.deg2rad(51.6), 0.0, np.deg2rad(215.0), 0.0])
X2 = np.array([26560.0, 0.02, np.deg2rad(55.0), 0.0, np.deg2rad(215.0), 0.0])
X3 = np.array(
    [26600.0, 0.74, np.deg2rad(63.4), np.deg2rad(270.0), np.deg2rad(80.0), 0.0]
)
X4 = np.array([42164.0, 0.00, 0.0, 0.0, np.deg2rad(35.0), 0.0])  # GEO


###############################################
# OPTIONAL
###############################################
def _propagate_states(oe0, t_array):
    a, e, inc, omega, Omega, f, r = _oe_propagate(oe0, t_array)
    # build r/v in each frame
    r_PF = []
    r_ECI = []
    r_ECEF = []
    for ti, fi in zip(t_array, f):
        # PF
        p = a * (1 - e**2)
        rpf = (p / (1 + e * np.cos(fi))) * np.array([np.cos(fi), np.sin(fi), 0.0])
        r_PF.append(rpf)
        # ECI
        C313 = sp.spatial.transform.Rotation.from_euler(
            "ZXZ", [Omega, inc, omega]
        ).as_matrix()
        r_ECI.append(C313 @ rpf)
        # ECEF
        gamma = omega_EN * ti
        C3 = sp.spatial.transform.Rotation.from_euler("Z", -gamma).as_matrix()
        r_ECEF.append(C3 @ (C313 @ rpf))
    return np.array(r_PF), np.array(r_ECI), np.array(r_ECEF)


def _plot_az_el(az, el, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    el_plot = el.copy()
    el_plot[el_plot < 0.0] = np.nan
    r = 90.0 - el_plot
    ax.plot(az, r)
    ax.set_rticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["90°", "75°", "60°", "45°", "30°", "15°", "0°"])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"{name} Azimuth vs Elevation")
    return fig


def _mean_motion(a, mu=mu):
    """
    Mean motion n [rad/s] for semi-major axis a [km] and mu [km^3/s^2].
    Works with scalars or numpy arrays.
    """
    a = np.asarray(a, dtype=float)
    return np.sqrt(mu / a**3)


def _oe_propagate(oe0, t, mu=mu, return_E_M=False):
    """
    Two-body Keplerian propagation (elliptical, e<1).

    Parameters
    ----------
    oe0 : array-like [a, e, i, omega, Omega, f0]  (angles in radians)
    t   : float or array-like times [s]
    mu  : gravitational parameter [km^3/s^2]

    Returns
    -------
    a, e, i, omega, Omega, f(t), r(t)
    (optionally E(t), M(t) if return_E_M=True)
    """
    a, e, inc, omega, Omega, f0 = [float(x) for x in oe0]
    t = np.atleast_1d(t).astype(float)
    n = _mean_motion(a, mu)

    if e < 1e-12:
        # circular limit: E=M, f = M; radius constant = a
        M = f0 + n * t
        E = M
        f = np.mod(M, 2 * np.pi)
        r = np.full_like(f, a, dtype=float)
    else:
        # initial E0 from f0
        E0 = 2.0 * np.arctan(np.tan(f0 / 2.0) * np.sqrt((1 - e) / (1 + e)))
        M0 = E0 - e * np.sin(E0)
        M = M0 + n * t

        # Newton-Raphson (vectorized) to solve E - e sin E = M
        E = M.copy()
        for _ in range(15):
            f_E = E - e * np.sin(E) - M
            f_Ep = 1.0 - e * np.cos(E)
            dE = -f_E / f_Ep
            E = E + dE
            if np.all(np.abs(dE) < 1e-12):
                break

        # true anomaly and radius
        ce, se = np.cos(E), np.sin(E)
        f = np.mod(np.arctan2(np.sqrt(1 - e * e) * se, ce - e), 2 * np.pi)
        r = a * (1 - e * ce)

    if return_E_M:
        return a, e, inc, omega, Omega, f, r, E, (E - e * np.sin(E))
    else:
        return a, e, inc, omega, Omega, f, r


def _site_ecef_from_geodetic(lat, lon, alt=0.0, model="spherical"):
    """
    Convert geodetic latitude/longitude and altitude to ECEF (km).

    Parameters
    ----------
    lat : float
        Geodetic latitude [rad]
    lon : float
        Geodetic longitude [rad]
    alt : float, optional
        Altitude above reference surface [km], default 0
    model : {"spherical", "wgs84"}, optional
        Earth model. "spherical" uses your R_E constant.
        "wgs84" uses the WGS-84 ellipsoid.

    Returns
    -------
    np.ndarray, shape (3,)
        ECEF position [km]
    """
    lat = float(lat)
    lon = float(lon)
    alt = float(alt)

    if model.lower() == "wgs84":
        # WGS-84 parameters (km)
        a = 6378.137  # semi-major axis
        f = 1.0 / 298.257223563  # flattening
        e2 = f * (2.0 - f)  # eccentricity squared

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        N = a / np.sqrt(1.0 - e2 * sin_lat**2)  # prime vertical radius of curvature
        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1.0 - e2) + alt) * sin_lat
        return np.array([x, y, z], dtype=float)

    # spherical Earth (uses your global R_E)
    cos_lat = np.cos(lat)
    x = (R_E + alt) * cos_lat * np.cos(lon)
    y = (R_E + alt) * cos_lat * np.sin(lon)
    z = (R_E + alt) * np.sin(lat)
    return np.array([x, y, z], dtype=float)


def _az_el_range_series(oe, dt=60.0):
    # one period
    a = oe[0]
    T = 2 * np.pi / _mean_motion(a)
    t = np.arange(0.0, T + dt, dt)

    # station ECEF
    lat, long = station_ll
    r_site_ecef = _site_ecef_from_geodetic(lat, long, alt=0.0)

    # spacecraft ECEF positions
    _, r_eci, r_ecef = _propagate_states(oe, t)

    az = np.zeros_like(t)
    el = np.zeros_like(t)
    rng = np.zeros_like(t)
    for k in range(t.size):
        C = sp.spatial.transform.Rotation.from_euler(
            "ZYZ", [-np.pi / 2, (lat - np.pi / 2), -long]
        ).as_matrix()
        e, n, u = C @ (r_ecef[k] - r_site_ecef)
        rng[k] = np.linalg.norm(np.array([e, n, u], dtype=float))
        el[k] = np.arcsin(u / rng[k])
        az[k] = np.arctan2(e, n) % (2 * np.pi)
    # degrees for elevation, az in radians for the polar helper
    return t, az, np.rad2deg(el), rng


###############################################
# REQUIRED FUNCTIONS
###############################################


###############################################
# Problem 1
###############################################
# REQUIRED --- Problem 1a
def Peri_2_ECI(X_P, oe):
    a, e, inc, omega, Omega, f = [float(x) for x in oe]
    r_pf, v_pf = X_P[:3], X_P[3:]
    C = sp.spatial.transform.Rotation.from_euler("ZXZ", [Omega, inc, omega]).as_matrix()
    r_eci = C @ r_pf
    v_eci = C @ v_pf
    return np.concatenate([r_eci, v_eci])


# REQUIRED --- Problem 1b
def ECI_2_ECEF(X_N, t, theta_0=0.0):
    r_eci, v_eci = X_N[:3], X_N[3:]
    gamma_g = omega_EN * t + theta_0
    C = sp.spatial.transform.Rotation.from_euler("Z", -gamma_g).as_matrix()
    r_ecef = C @ r_eci
    v_ecef = C @ v_eci
    return np.concatenate([r_ecef, v_ecef])


# REQUIRED --- Problem 1c
def ECEF_2_TOPO(X_E, R_site_E):
    r_ecef = X_E[:3]
    # infer lat/lon from site ECEF (spherical)
    x, y, z = R_site_E
    long = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x * x + y * y))
    rho = r_ecef - R_site_E
    C = sp.spatial.transform.Rotation.from_euler(
        "ZYZ", [-np.pi / 2, (lat - np.pi / 2), -long]
    ).as_matrix()
    enu = C @ rho
    return enu  # [e, n, u] in km


###############################################
# Problem 2
###############################################


# REQUIRED --- Problem 2a
def plot_orbits():
    T24 = 24 * 3600.0
    t = np.linspace(0.0, T24, 2000)
    r_pf, r_eci, r_ecef = _propagate_states(X_OE_0, t)

    figs = []

    # Perifocal
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot(r_pf[:, 0], r_pf[:, 1], r_pf[:, 2])
    ax1.set_xlabel("p [km]")
    ax1.set_ylabel("q [km]")
    ax1.set_zlabel("w [km]")
    ax1.set_title("Trajectory in Perifocal Frame")
    ax1.axis("equal")
    figs.append(fig1)

    # ECI
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot(r_eci[:, 0], r_eci[:, 1], r_eci[:, 2])
    ax2.set_xlabel("I [km]")
    ax2.set_ylabel("J [km]")
    ax2.set_zlabel("K [km]")
    ax2.set_title("Trajectory in ECI Frame")
    ax2.axis("equal")
    figs.append(fig2)

    # ECEF
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot(r_ecef[:, 0], r_ecef[:, 1], r_ecef[:, 2])
    ax3.set_xlabel("x_E [km]")
    ax3.set_ylabel("y_E [km]")
    ax3.set_zlabel("z_E [km]")
    ax3.set_title("Trajectory in ECEF Frame")
    ax3.axis("equal")
    figs.append(fig3)

    return figs


# REQUIRED --- Problem 2b
def describe_orbits():
    return (
        "Perifocal: orbit is a fixed ellipse in the spacecraft orbital plane\n"
        "\tGreat for two-body analytics and interpreting anomalies (θ), but has no Earth context.\n"
        "ECI: ellipse is fixed in inertial space (no Earth rotation)\n"
        "\tGood for multi-body perturbations and inter-frame transforms; longitude/latitude are not obvious.\n"
        "ECEF: Earth-fixed axes rotate with the planet; the trajectory appears to sweep over the rotating Earth\n"
        "\tThis is the natural frame for **ground tracks**, access windows, and station visibility, but motion\n"
        "\tmixes orbital dynamics with Earth rotation."
    )


###############################################
# Problem 3
###############################################
def _ground_track_for_oe(oe, n_periods=3, dt=60.0):
    a, e, inc, omega, Omega, f0 = oe
    T = 2 * np.pi / _mean_motion(a)
    t = np.arange(0.0, n_periods * T + dt, dt)
    r_pf, r_eci, r_ecef = _propagate_states(oe, t)
    # geocentric lat/lon from ECEF
    x, y, z = r_ecef[:, 0], r_ecef[:, 1], r_ecef[:, 2]
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arctan2(z, np.sqrt(x * x + y * y)))
    # wrap longitudes to [-180,180]
    lon = (lon + 180.0) % 360.0 - 180.0
    return t, lat, lon, r_ecef


# REQUIRED --- Problem 3a
def plot_groundtracks():
    oes = [X1, X2, X3, X4]
    names = ["X1 (LEO)", "X2 (MEO)", "X3 (Molniya)", "X4 (GEO)"]
    figs = []

    for k, (oe, name) in enumerate(zip(oes, names), start=1):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.2)
        _, lat, lon, _ = _ground_track_for_oe(oe, n_periods=3, dt=60.0)
        ax.plot(lon, lat, transform=ccrs.PlateCarree(), linewidth=0.8)
        ax.set_title(f"{name} — Ground Track (3 periods)")
        # ax.set_xlim(-180, 180)
        # ax.set_ylim(-90, 90)
        figs.append(fig)

    return figs


def _perigee_ssp_region(oe):
    # approximate SSP at perigee at t=0 (theta=0)
    a, e, inc, omega, Omega, f0 = oe.copy()
    f_p = 0.0
    rpf = (a * (1 - e**2) / (1 + e * np.cos(f_p))) * np.array(
        [np.cos(f_p), np.sin(f_p), 0.0]
    )
    C = sp.spatial.transform.Rotation.from_euler("ZXZ", [Omega, inc, omega]).as_matrix()
    r_eci = C @ rpf
    C = sp.spatial.transform.Rotation.from_euler("Z", 0.0).as_matrix()
    r_ecef = C @ r_eci
    x, y, z = r_ecef
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arctan2(z, np.sqrt(x * x + y * y)))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon


# REQUIRED --- Problem 3b
def describe_spacecraft_proximity():
    # closest to Earth occurs near perigee; give coarse regions at t=0
    regions = []
    names = ["X1 (LEO)", "X2 (MEO)", "X3 (Molniya)", "X4 (GEO)"]
    for name, oe in zip(names, [X1, X2, X3, X4]):
        lat, lon = _perigee_ssp_region(oe)
        regions.append(f"{name}: perigee SSP ≈ lat {lat:+.1f}°, lon {lon:+.1f}°")
    return (
        "Closest approach to Earth occurs at perigee. Approximate subsatellite points (at t=0) are:\n"
        + "\n".join(regions)
    )


# REQUIRED --- Problem 3c
def describe_orbit_use_case():
    return (
        "X1 (LEO): human spaceflight / Earth observation; frequent revisits, moderate global coverage\n"
        "X2 (MEO): GNSS constellation-style navigation; near 12-hour period gives repeating ground tracks\n"
        "X3 (Molniya): long dwell over high northern latitudes—ideal for comms and ISR in high-lat regions\n"
        "X4 (GEO): continuous coverage over a fixed longitude—telecom, weather, and broadcast"
    )


###############################################
# Problem 4
###############################################
# REQUIRED --- Problem 4a
def plot_az_el_measurements():
    figs = []
    names = ["X1 (LEO)", "X2 (MEO)", "X3 (Molniya)", "X4 (GEO)"]
    for name, oe in zip(names, [X1, X2, X3, X4]):
        _, az, el_deg, _ = _az_el_range_series(oe, dt=60.0)
        figs.append(_plot_az_el(az, el_deg, f"{name}"))
    return figs


# REQUIRED --- Problem 4b
def plot_range():
    figs = []
    names = ["X1 (LEO)", "X2 (MEO)", "X3 (Molniya)", "X4 (GEO)"]
    for name, oe in zip(names, [X1, X2, X3, X4]):
        t, az, el_deg, rng = _az_el_range_series(oe, dt=60.0)

        # mask below 10 deg elevation
        rng_plot = rng.copy()
        rng_plot[el_deg < 10.0] = np.nan

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t / 3600.0, rng_plot)
        ax.set_xlabel("Time [hours]")
        ax.set_ylabel("Range [km]")
        ax.set_title(f"{name} Range vs Time (masked for el < 10°)")
        ax.grid(True, alpha=0.3)
        figs.append(fig)
    return figs


# REQUIRED --- Problem 4c
def describe_visibility():
    # simple visibility check
    vis = []
    names = ["X1 (LEO)", "X2 (MEO)", "X3 (Molniya)", "X4 (GEO)"]
    for name, oe in zip(names, [X1, X2, X3, X4]):
        _, _, el_deg, _ = _az_el_range_series(oe, dt=60.0)
        vis.append((name, np.any(el_deg >= 10.0)))
    lines = [
        f"{n}: {'VISIBLE' if ok else 'NOT VISIBLE'} during one orbit from Goldstone."
        for (n, ok) in vis
    ]
    return "\n".join(lines)


###############################################
# Main
###############################################
def main():
    print("enae441-hw02")

    # 1a
    with open("../outputs/text/s01a.txt", "w", encoding="utf-8") as f:
        f.write(
            np.array2string(
                Peri_2_ECI(
                    np.array([6650.0, 0.0, 0.0, -0.0, 7.933279, 0.0]),
                    np.array(
                        [
                            7000.0,
                            0.05,
                            np.deg2rad(45.0),
                            np.deg2rad(30.0),
                            np.deg2rad(60.0),
                            0.0,
                        ]
                    ),
                ),
                max_line_width=120,
            )
        )

    # 1b
    with open("../outputs/text/s01b.txt", "w", encoding="utf-8") as f:
        f.write(
            np.array2string(
                ECI_2_ECEF(
                    np.array(
                        [
                            843.396119,
                            6163.065024,
                            2351.130047,
                            -6.190576,
                            -1.00615,
                            4.858121,
                        ]
                    ),
                    10.0,
                    0.0,
                ),
                max_line_width=120,
            )
        )

    # 1c
    with open("../outputs/text/s01c.txt", "w", encoding="utf-8") as f:
        f.write(
            np.array2string(
                ECEF_2_TOPO(
                    np.array(
                        [
                            847.889447,
                            6162.448457,
                            2351.130047,
                            -6.191308,
                            -1.001636,
                            4.858121,
                        ]
                    ),
                    np.array([-2356.308343, -4641.706037, 3685.276117]),
                ),
                max_line_width=120,
            )
        )

    # 2a
    plot_orbits()

    # 2b
    with open("../outputs/text/s02b.txt", "w", encoding="utf-8") as f:
        f.write(describe_orbits())

    # 3a
    plot_groundtracks()

    # 3b
    with open("../outputs/text/s03b.txt", "w", encoding="utf-8") as f:
        f.write(describe_spacecraft_proximity())

    # 3c
    with open("../outputs/text/s03c.txt", "w", encoding="utf-8") as f:
        f.write(describe_orbit_use_case())

    # 4a
    plot_az_el_measurements()

    # 4b
    plot_range()

    # 4c
    with open("../outputs/text/s04c.txt", "w", encoding="utf-8") as f:
        f.write(describe_visibility())

    plt.show()


if __name__ == "__main__":
    main()
