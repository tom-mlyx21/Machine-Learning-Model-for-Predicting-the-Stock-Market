

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:41:21 2024

@author: terry
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:52:33 2024

@author: terry
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
m_Sgr = 8.26e36  # Mass of the Sgr A* (kg)
m_S2 = 2.8e31  # Mass of the S2 (kg)

# Initial conditions for elliptical orbit
x_S2 = 1.75e13  # Initial x-position of S2 (m)
y_S2 = 0.0  # Initial y-position of S2 (m)
vx_S2 = 0.0  # Initial x-velocity of S2 (m/s)
vy_S2 = 7.7095e6  # Initial y-velocity of S2 (m/s)

# Time parameters
dt = 10 * 86400.0  # Time step (s), corresponds to one day
t_max = 10 * 365 * 16 * 24 * 3600.0  # Total simulation time (s)
n_steps = int(t_max / dt)  # Number of time steps

# Arrays to store positions and velocities
x_S2_values = np.zeros(n_steps)
y_S2_values = np.zeros(n_steps)

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 8))
line, = ax.plot([], [], lw=1, label='S2 Orbit', color='r')
ax.scatter(0, 0, color='black', label='Sgr A*')
ax.set_xlabel('x-position (m)')
ax.set_ylabel('y-position (m)')
ax.set_title('S2 orbit around Sgr A* using Newtonian physics')
ax.set_aspect('equal', adjustable='box')
ax.legend()
ax.grid(True)


# Function to initialize the plot
def init():
    line.set_data([], [])
    return line,


# Initial orbit radians
orb_rad = 0.0


# Function to update the plot for each frame using RK4 method
def update(frame):
    global x_S2, y_S2, vx_S2, vy_S2, orb_rad

    # Define the RK4 method
    def rk4(x, y, vx, vy, dt):
        r = np.array([x, y])
        v = np.array([vx, vy])

        k1r = v
        k1v = -G * m_Sgr / np.linalg.norm(r) ** 3 * r

        k2r = v + 0.5 * dt * k1v
        k2v = -G * m_Sgr / np.linalg.norm(r + 0.5 * dt * k1r) ** 3 * (r + 0.5 * dt * k1r)

        k3r = v + 0.5 * dt * k2v
        k3v = -G * m_Sgr / np.linalg.norm(r + 0.5 * dt * k2r) ** 3 * (r + 0.5 * dt * k2r)

        k4r = v + dt * k3v
        k4v = -G * m_Sgr / np.linalg.norm(r + dt * k3r) ** 3 * (r + dt * k3r)

        xf = x + (dt / 6.0) * (k1r[0] + 2.0 * k2r[0] + 2.0 * k3r[0] + k4r[0])
        yf = y + (dt / 6.0) * (k1r[1] + 2.0 * k2r[1] + 2.0 * k3r[1] + k4r[1])
        vxf = vx + (dt / 6.0) * (k1v[0] + 2.0 * k2v[0] + 2.0 * k3v[0] + k4v[0])
        vyf = vy + (dt / 6.0) * (k1v[1] + 2.0 * k2v[1] + 2.0 * k3v[1] + k4v[1])

        return xf, yf, vxf, vyf

    # Update Star's position and velocity using RK4 method
    x_S2, y_S2, vx_S2, vy_S2 = rk4(x_S2, y_S2, vx_S2, vy_S2, dt)
    r = (x_S2 ** 2 + y_S2 ** 2) ** (1 / 2)
    mu = 1.451e14 * (1 - (0.88) ** 2)

    # Precess the orbit
    orb_precession = (6 * np.pi * G * m_Sgr) / ((3 * 10 ** 8) ** 2 * mu) * np.pi / 180
    if r == 1.75e13:
        x_S2 = x_S2 + orb_precession
        y_S2 = y_S2 + orb_precession
    # precession = orb_precession * (2 * np.pi) / (365 * 24 * 3600 * 16)
    # orb_rad += precession * dt  # Update orbit radians
    # x_S2, y_S2 = x_S2 * np.cos(orb_rad) - y_S2 * np.sin(orb_rad), x_S2 * np.sin(orb_rad) + y_S2 * np.cos(orb_rad)

    # Store S2's position
    x_S2_values[frame] = x_S2
    y_S2_values[frame] = y_S2

    # Update the plot
    if frame > 0 and np.linalg.norm([x_S2 - x_S2_values[0], y_S2 - y_S2_values[0]]) > 1e9:
        line.set_data(x_S2_values[:frame + 1], y_S2_values[:frame + 1])

    return line,


# Create animation
ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True)

# convert plot labels to astronomical units
m2km = lambda x, _: f'{int(x / 1.496e+11):g}'
ax.xaxis.set_major_formatter(m2km)
ax.yaxis.set_major_formatter(m2km)

# Set new limits for better visibility
ax.set_xlim(-3e14, 1e14)
ax.set_ylim(-3e14, 1e14)

plt.show()