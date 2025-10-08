import matplotlib.pyplot as plt
import numpy as np

# CONSTANTS
# Gravitational parameter for Earth in km^3/s^2
mu = 3.986004418e5
omega_EN = 7.2911 * 10**-5  # rad /s


# problem 2
X_OE_0 = np.array(
    [
        7000,
        0.05,
        np.deg2rad(45),
        np.deg2rad(30),
        np.deg2rad(60),
        np.deg2rad(0),
    ]
)


# Problem 3 + 4
R_E = 6378
station_coords = (35.2967, -116.9141)  # deg

X1 = np.array(
    [
        420 + R_E,
        0.007,
        np.deg2rad(51.6),
        np.deg2rad(0),
        np.deg2rad(215),
        np.deg2rad(0),
    ]
)
X2 = np.array(
    [
        26560,
        0.02,
        np.deg2rad(55),
        np.deg2rad(0),
        np.deg2rad(215),
        np.deg2rad(0),
    ]
)
X3 = np.array(
    [
        26600,
        0.74,
        np.deg2rad(63.4),
        np.deg2rad(270),
        np.deg2rad(80),
        np.deg2rad(0),
    ]
)
X4 = np.array(
    [
        42164,
        0.0,
        np.deg2rad(0),
        np.deg2rad(0),
        np.deg2rad(35),
        np.deg2rad(0),
    ]
)


###############################################
# OPTIONAL FUNCTIONS TO AID IN DEBUGGING
# These are not graded functions, but may help you debug your code.
# Keep the function signatures the same if you want autograder feedback!
###############################################


def ECI_2_OE(X, mu):
    return np.array([a, e, i, omega, Omega, f])


def OE_2_Perifocal(oe, mu):
    return X_P


def ECI_2_Peri(x_N, oe):
    return X_P


def ECEF_2_LatLong(X_ECEF_t):
    return np.rad2deg(lat), np.rad2deg(long)


def plot_az_el(az, el, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # Mask out any negative elevations (below horizon)
    mask = el < 0
    el_plot = el.copy()
    el_plot[mask] = np.nan

    # Azimuth in Radians
    # Elevation in Degrees
    r = 90 - el_plot
    ax.plot(az, r)

    # set the r ticks to be 90 at center and increment down by 15
    ax.set_rticks([0, 15, 30, 45, 60, 75, 90])  # Define the tick positions

    # Overwrite the labels
    ax.set_yticklabels(["90°", "75°", "60°", "45°", "30°", "15°", "0°"])

    ax.set_theta_zero_location("N")  # Azimuth 0 points to North
    ax.set_theta_direction(-1)  # Clockwise azimuth

    ax.set_title(f"{name} Azimuth vs Elevation")


###############################################
# REQUIRED FUNCTIONS FOR AUTOGRADER
# Keep the function signatures the same!!
###############################################


#######################
# Problem 1
#######################


# REQUIRED --- 1a
def Peri_2_ECI(X_P, oe):
    return X_N


# REQUIRED --- 1b
def ECI_2_ECEF(X_N, t, theta_0=0.0):
    return X_E


# REQUIRED --- 1c
def ECEF_2_TOPO(X_E, R_site_E):
    return X_T


#######################
# Problem 2
#######################


# REQUIRED --- Problem 2a
def plot_orbits():
    return figs


# REQUIRED --- Problem 2b
def describe_orbits():
    return """
        Write your answer here.

        **Note**: You can split your answers across multiple lines 
        to improve readability (the TAs will thank you!)
    """


#######################
# Problem 3
#######################


# REQUIRED --- Problem 3a
def plot_groundtracks():
    return figs


# REQUIRED --- Problem 3b
def describe_spacecraft_proximity():
    return """Write your answer here"""


# REQUIRED --- Problem 3c
def describe_orbit_use_case():
    return """Write your answer here"""


#######################
# Problem 4
#######################


# REQUIRED --- Problem 4a
def plot_az_el_measurements():
    # use plot_az_el function
    return figs


# REQUIRED Problem 4b:
def plot_range():
    return figs


# REQUIRED Problem 4c:
def describe_visibility():
    return """Write your answer here"""


###############################################
# Main Script to test / debug your code
# This will not be run by the autograder
# the individual functions above will be called and tested
###############################################


def main():
    # REQUIRED --- 1a
    fcn1 = Peri_2_ECI

    # REQUIRED --- 1b
    fcn2 = ECI_2_ECEF

    # REQUIRED --- 1c
    fcn3 = ECEF_2_TOPO

    # Problem 2
    plot_orbits()
    describe_orbits()

    # Problem 3
    plot_groundtracks()
    describe_spacecraft_proximity()
    describe_orbit_use_case()

    # Problem 4
    plot_az_el_measurements()
    plot_range()
    describe_visibility()
    plt.show()


if __name__ == "__main__":
    main()
