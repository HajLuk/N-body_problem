import matplotlib.pyplot as plt
import numpy as np
import random


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
    random.seed()
    # Basic settings
    nBodies = random.randrange(16)
    print("Calculating trajectories of", nBodies, "bodies…")
    time_steps = 500
    t_step = 24 * 3600  # time step in seconds (one time step takes 24 hours now)
    # Allocation with zeros
    coordinates_x = np.zeros((time_steps, nBodies))  # coordinates x
    coordinates_y = np.zeros((time_steps, nBodies))  # coordinates y
    velocities_x = np.zeros((time_steps, nBodies))  # velocities (first component of the vector)
    velocities_y = np.zeros((time_steps, nBodies))  # velocities (second component of the vector)
    masses = np.zeros(nBodies)  # masses

    for body in range(nBodies):
        masses[body] = random.random() * 2.0e30
        coordinates_x[0, body] = random.random() * 1.0e12
        coordinates_y[0, body] = random.random() * 1.0e12
        velocities_x[0, body] = random.random() * 0.0
        velocities_y[0, body] = random.random() * 0.0

    # Get the solution!
    solve(coordinates_x, coordinates_y, velocities_x, velocities_y, masses, t_step)

    # Prepare our figure for all trajectories
    fig1 = plt.figure("Trajectories")
    # Trajectories
    for body in range(nBodies):
        r = random.random()
        g = random.random()
        b = random.random()
        plt.plot(coordinates_x[:, body], coordinates_y[:, body], "-",
                 color=(r, g, b), label="body #" + str(body))
        plt.plot(coordinates_x[time_steps-1, body], coordinates_y[time_steps-1, body], "o",
                 color=(r, g, b))
    # Labels
    plt.legend()
    plt.xlabel('x (metres)')
    plt.ylabel('y (metres)')
    plt.gca().set_aspect('equal', adjustable='box')  # optionally add adjustable='box'
    plt.grid()
    # Draw it
    plt.show()
