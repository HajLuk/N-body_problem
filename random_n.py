import matplotlib.pyplot as plt
import numpy as np
import random


def generate_bodies(x1, x2, v1, v2, m):
    """
    Generate pseudorandom values for all n bodies (coordinates, initial velocities and masses).
    Properly allocated empty fields are expected on the input.
    :param x1: x coordinates
    :param x2: y coordinates
    :param v1: x velocities
    :param v2: y velocities
    :param m: masses
    :return:
    """
    n_bodies = x1.shape[1]  # number N of bodies (set by number of columns of input variables)
    for body in range(n_bodies):
        m[body] = random.random() * 2.0e30
        x1[0, body] = random.random() * 1.0e12
        x2[0, body] = random.random() * 1.0e12
        v1[0, body] = random.random() * 0.0
        v2[0, body] = random.random() * 0.0


def g_acc(radius, mass):
    """
    Computes acceleration of first body caused by gravitational force of a second body with mass "mass".
    Vector "radius" points from the second body to the first.
    :param radius: radius vector from body 2 to body 1
    :param mass: mass of body 2
    :return:
    """
    if np.linalg.norm(radius) < 1.0:  # distance is very small so it is the same body
        return np.array([0.0, 0.0])  # the body exerts no force on itself
    else:
        G = 6.674e-11  # Gravitational constant
        dist = np.linalg.norm(radius)
        return G * mass * radius / (dist ** 3)  # acceleration


def solve(x1, x2, v1, v2, m, dt):
    """
    Solves the N-body problem
    Gets array of masses of all bodies, size of time step dt and also
    arrays of their coordinates (x1,x2) and velocities (v1,v2) with set initial conditions (and zeros from line 2)
    :param x1: x coordinates
    :param x2: y coordinates
    :param v1: x velocities
    :param v2: y velocities
    :param m: masses
    :param dt: time step
    :return:
    """
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


def draw_all(x1, x2):
    """
    Plots the trajectories of all points with coordinates (x1,x2)
    :param x1: x coordinates
    :param x2: y coordinates
    :return:
    """
    t_steps = x1.shape[0]  # number of time steps (set by number of lines of input variables)
    n_bodies = x1.shape[1]  # number N of bodies (set by number of columns of input variables)
    # Prepare our figure for all trajectories
    plt.figure("Trajectories")
    # Trajectories
    for body in range(n_bodies):
        rgb = (random.random(), random.random(), random.random())
        plt.plot(x1[:, body], x2[:, body], "-",
                 color=rgb)
        plt.plot(x1[t_steps-1, body], x2[t_steps-1, body], "o",
                 color=rgb, label="body #" + str(body))
    # Labels
    plt.legend()
    plt.xlabel('x (metres)')
    plt.ylabel('y (metres)')
    plt.gca().set_aspect('equal', adjustable='box')  # optionally add adjustable='box'
    plt.grid()
    # Draw it
    plt.show()


if __name__ == '__main__':
    # Basic settings
    random.seed()  # seed our RNG
    nBodies = random.randrange(16)  # we solve the N-body problem and N is this number
    print("Calculating trajectories of", nBodies, "bodies…")
    time_steps = 500  # how many time steps we want to calculate?
    t_step = 24 * 3600  # time step in seconds (one time step takes 24 hours now)

    # Allocation with zeros
    coordinates_x = np.zeros((time_steps, nBodies))  # coordinates x
    coordinates_y = np.zeros((time_steps, nBodies))  # coordinates y
    velocities_x = np.zeros((time_steps, nBodies))  # velocities (first component of the vector)
    velocities_y = np.zeros((time_steps, nBodies))  # velocities (second component of the vector)
    masses = np.zeros(nBodies)  # masses

    # Generate all n bodies
    generate_bodies(coordinates_x, coordinates_y, velocities_x, velocities_y, masses)
    # Get the solution!
    solve(coordinates_x, coordinates_y, velocities_x, velocities_y, masses, t_step)
    # Plot everything
    draw_all(coordinates_x, coordinates_y)
