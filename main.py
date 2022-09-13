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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Basic settings
    nBodies = 3
    time_steps = 386
    dt = 24 * 3600  # time step in seconds (one time step takes 24 hours now)
    # Allocation with zeros
    x1 = np.zeros((time_steps, nBodies))  # coordinates x
    x2 = np.zeros((time_steps, nBodies))  # coordinates y
    v1 = np.zeros((time_steps, nBodies))  # velocities (first component of the vector)
    v2 = np.zeros((time_steps, nBodies))  # velocities (second component of the vector)
    m = np.zeros(nBodies)  # masses

    # Define all our bodies (mass, two components of position and two components of velocity for each)
    # Sun (only mass is not zero)
    m[0] = 1.989e30  # (kilograms)
    # Earth
    x1[0, 1] = 149.6e9  # right from the Sun (m)
    v2[0, 1] = 30300.0  # upwards (m/s)
    m[1] = 5.972e24  # (kg)
    # Moon
    x1[0, 2] = x1[0, 1] + 384.4e6  # right from the Earth and Sun (m)
    v2[0, 2] = v2[0, 1] + 1022.0  # upwards (m/s)
    m[2] = 7.348e22  # (kg)

    for t in range(1, time_steps):
        for j in range(nBodies):
            a = np.array([0.0, 0.0])
            for k in range(nBodies):
                radius = np.array([x1[t - 1, k] - x1[t - 1, j], x2[t - 1, k] - x2[t - 1, j]])
                a = a + g_acc(radius, m[k])
                # Euler-Cromer
                v1[t, j] = v1[t - 1, j] + a[0] * dt
                v2[t, j] = v2[t - 1, j] + a[1] * dt
                x1[t, j] = x1[t - 1, j] + v1[t, j] * dt
                x2[t, j] = x2[t - 1, j] + v2[t, j] * dt

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
    plt.plot(x1[:, 0], x2[:, 0], "-o", color=(1.0, 1.0, 0.7), label="Sun")
    plt.plot(x1[:, 2], x2[:, 2], "-", color=(0.8, 0.8, 0.0), label="Moon")
    plt.plot(x1[:, 1], x2[:, 1], "-", color=(0.0, 0.0, 1.0), label="Earth")
    # Labels
    plt.legend()
    plt.xlabel('x (metres)')
    plt.ylabel('y (metres)')
    plt.gca().set_aspect('equal', adjustable='box')  # optionally add adjustable='box'
    plt.grid(color=(0.3, 0.3, 0.3))
    # Draw it
    plt.show()
