import matplotlib.pyplot as plt
import numpy as np


def g_acc(radius, mass):
    # Computes acceleration of first body caused by gravitational force of a second body with mass "mass".
    # Vector "radius" points from the second body to the first.
    if np.linalg.norm(radius) < 1.0:
        return np.array([0.0, 0.0])  # the body exerts no force on itself
    else:
        G = 6.674e-11
        dist = np.linalg.norm(radius)
        return G * mass * radius / (dist ** 3)  # acceleration


def solve(x1, x2, v1, v2, m, dt):
    # Solves the N-body problem
    # gets array of masses of all bodies, size of time step dt and also
    # arrays of their coordinates (x1,x2) and velocities (v1,v2) with set initial conditions (and zeros from line 2)
    t_steps = x1.shape[0]  # number of time steps (set by number of lines of input variables)
    n_bodies = x1.shape[1]  # number N of bodies (set by number of columns of input variables)
    for t in range(1, t_steps):
        for j in range(n_bodies):
            a = np.array([0.0, 0.0])
            for k in range(n_bodies):
                radius = np.array([x1[t - 1, k] - x1[t - 1, j], x2[t - 1, k] - x2[t - 1, j]])
                a = a + g_acc(radius, m[k])
                # Euler-Cromer
                v1[t, j] = v1[t - 1, j] + a[0] * dt
                v2[t, j] = v2[t - 1, j] + a[1] * dt
                x1[t, j] = x1[t - 1, j] + v1[t, j] * dt
                x2[t, j] = x2[t - 1, j] + v2[t, j] * dt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Basic settings
    nBodies = 3
    time_steps = 386
    t_step = 24 * 3600  # time step in seconds (one time step takes 24 hours now)
    # Allocation with zeros
    coordinates_x = np.zeros((time_steps, nBodies))  # coordinates x
    coordinates_y = np.zeros((time_steps, nBodies))  # coordinates y
    velocities_x = np.zeros((time_steps, nBodies))  # velocities (first component of the vector)
    velocities_y = np.zeros((time_steps, nBodies))  # velocities (second component of the vector)
    masses = np.zeros(nBodies)  # masses

    # Define all our bodies (mass, two components of position and two components of velocity for each)
    # Sun (only mass is not zero)
    masses[0] = 1.989e30  # (kilograms)
    # Earth
    coordinates_x[0, 1] = 149.6e9  # right from the Sun (m)
    velocities_y[0, 1] = 30300.0  # upwards (m/s)
    masses[1] = 5.972e24  # (kg)
    # Moon
    coordinates_x[0, 2] = coordinates_x[0, 1] + 384.4e6  # right from the Earth and Sun (m)
    velocities_y[0, 2] = velocities_y[0, 1] + 1022.0  # upwards (m/s)
    masses[2] = 7.348e22  # (kg)

    # Get the solution!
    solve(coordinates_x, coordinates_y, velocities_x, velocities_y, masses, t_step)

    # Prepare our figure for all trajectories
    fig1 = plt.figure("Trajectories")
    # Paint It Black 🎵
    plt.style.use('dark_background')
    ax = fig1.add_subplot(111)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')
    # Trajectories
    plt.plot(coordinates_x[:, 0], coordinates_y[:, 0], "-o", color=(1.0, 1.0, 0.7), label="Sun")
    plt.plot(coordinates_x[:, 2], coordinates_y[:, 2], "-", color=(0.8, 0.8, 0.0), label="Moon")
    plt.plot(coordinates_x[:, 1], coordinates_y[:, 1], "-", color=(0.0, 0.0, 1.0), label="Earth")
    # Labels
    plt.legend()
    plt.xlabel('x (metres)')
    plt.ylabel('y (metres)')
    plt.gca().set_aspect('equal', adjustable='box')  # optionally add adjustable='box'
    plt.grid(color=(0.3, 0.3, 0.3))
    # Draw it
    plt.show()
