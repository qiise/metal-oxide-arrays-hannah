import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Defining the particle as an agent
class Particle:
    def __init__(self, x, y, velocity):
        self.x = x
        self.y = y
        self.state = "unassembled"  # Initial state
        self.transformation_time = None
        self.velocity = velocity  # Velocity as a tuple (vx, vy)

    def update(self, t, k0, f_x):
        """Updates state based on the Avrami equation and moves the particle."""
        if self.state == "unassembled":
            P = 1 - math.exp(-k0 * f_x * t**2)  # Transformation probability
            if random.random() < P:
                self.state = "assembled"
                self.transformation_time = t

        # Move the particle if it's still unassembled
        if self.state == "unassembled":
            self.x += self.velocity[0]
            self.y += self.velocity[1]

            # Keep particles inside the boundary (bouncing effect)
            if self.x < 0 or self.x > 10:
                self.velocity = (-self.velocity[0], self.velocity[1])
                self.x = max(0, min(self.x, 10))

            if self.y < 0 or self.y > 1:
                self.velocity = (self.velocity[0], -self.velocity[1])
                self.y = max(0, min(self.y, 1))

# Defining the model
class SelfAssemblyModel:
    def __init__(self, N, width, length, k0, alpha, x_p, sigma, velocity_mag):
        self.width = width
        self.length = length
        self.k0 = k0  # Base rate constant
        self.alpha = alpha  # Scaling factor
        self.x_p = x_p  # Peak location for concentration profile
        self.sigma = sigma  # Spread of peak
        self.velocity_mag = velocity_mag  # Magnitude of velocity

        self.particles = [
            Particle(
                random.uniform(0, length),
                random.uniform(0, width),
                (random.uniform(-velocity_mag, velocity_mag), random.uniform(-velocity_mag, velocity_mag))
            ) for _ in range(N)
        ]
        self.time = 0

    def correction_factor(self, x):
        return 1 + self.alpha * math.exp(-((x - self.x_p) ** 2) / (2 * self.sigma**2))

    def step(self):
        self.time += 1
        for particle in self.particles:
            f_x = self.correction_factor(particle.x)
            particle.update(self.time, self.k0, f_x)

# Visualization function
def visualize_simulation(model, num_steps):
    fig, ax = plt.subplots()
    ax.set_xlim(0, model.length)
    ax.set_ylim(0, model.width)
    ax.set_xlabel("Channel Length (x)")
    ax.set_ylabel("Channel Width (y)")
    ax.set_title("Self-Assembly of Metal Oxide Array")

    unassembled_scatter = ax.scatter([], [], color="red", label="Unassembled")
    assembled_scatter = ax.scatter([], [], color="blue", label="Assembled")
    ax.legend()

    def update(frame):
        model.step()

        unassembled_x = [p.x for p in model.particles if p.state == "unassembled"]
        unassembled_y = [p.y for p in model.particles if p.state == "unassembled"]
        assembled_x = [p.x for p in model.particles if p.state == "assembled"]
        assembled_y = [p.y for p in model.particles if p.state == "assembled"]

        unassembled_scatter.set_offsets(np.column_stack((unassembled_x, unassembled_y)))
        assembled_scatter.set_offsets(np.column_stack((assembled_x, assembled_y)))

        return unassembled_scatter, assembled_scatter

    ani = FuncAnimation(fig, update, frames=num_steps, interval=100, blit=False)  # Set blit=False
    plt.show()

# Model parameters
N = 100  # Number of particles
width = 1  # Channel width
length = 10  # Channel length
k0 = 0.1  # Base rate constant
alpha = 0  # Scaling factor for velocity correction
x_p = 3.0  # Peak location for concentration profile
sigma = 1.0  # Spread of the peak
velocity_mag = 0.05  # Small velocity for particle motion

model = SelfAssemblyModel(N, width, length, k0, alpha, x_p, sigma, velocity_mag)

# Run the simulation and visualize
num_steps = 100  # Number of time steps
visualize_simulation(model, num_steps)

        




