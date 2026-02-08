import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import config as cfg
import physics as phys

class SPHSimulation:
    def __init__(self):
        self.head = np.full(cfg.GRID_SIZE * cfg.GRID_SIZE, -1, dtype=np.int32)
        self.next_particle = np.full(cfg.N_PARTICLES, -1, dtype=np.int32)
        self.mass_per_particle = cfg.MASS / cfg.N_PARTICLES
        self.steps_per_frame = 1
        self.remnant_mask = np.zeros(cfg.N_PARTICLES, dtype=bool)

        # Matplotlib Setup
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(9, 8), facecolor='black')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Background Stars
        rng = np.random.default_rng(123)
        bg_x = rng.uniform(-4, 4, 300)
        bg_y = rng.uniform(-4, 4, 300)
        bg_s = rng.uniform(0.1, 1.5, 300)
        bg_alpha = rng.uniform(0.1, 0.6, 300)
        self.ax.scatter(bg_x, bg_y, s=bg_s, c='white', alpha=bg_alpha, zorder=0)

        # Star Temperature Colormap
        colors = ['#550000', '#ff0000', '#ff8800', '#ffff00', '#ffffff', '#aaddff']
        nodes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.star_cmap = LinearSegmentedColormap.from_list("star_temp", list(zip(nodes, colors)))

        # Particle Scatter Plot
        self.scat = self.ax.scatter([], [], s=6, c=[], cmap=self.star_cmap,
                                    norm=LogNorm(vmin=0.01, vmax=100), alpha=1.0, zorder=15, animated=True)

        # Colorbar
        cbar = plt.colorbar(self.scat, ax=self.ax, fraction=0.046, pad=0.04)
        cbar.set_label('Temperature (Internal Energy)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Neutron Star Remnant Scatter
        self.remnant_scat = self.ax.scatter([], [], s=40, c='white',
                                            edgecolors='cyan', linewidth=1.5, zorder=20, animated=True)

        # UI Text
        self.text_status = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes,
                                        color="white", fontsize=10, family='monospace', animated=True)
        self.text_inst = self.ax.text(0.02, 0.02,
                                      "SPACE: Implode | [ / ] : Speed | R: Reset",
                                      transform=self.ax.transAxes, color="gray", fontsize=9, animated=True)

        print("Compiling Physics Engine...")
        self.reset()

        # JIT Warmup
        phys.build_grid(self.pos, self.head, self.next_particle)
        rho, p = phys.compute_density_pressure(self.pos, self.u, self.mass_per_particle, self.head, self.next_particle)
        phys.compute_gravity(self.pos, self.mass_per_particle, cfg.G_BASE)
        phys.compute_sph_forces(self.pos, self.vel, rho, p, self.mass_per_particle, self.head, self.next_particle)
        print("Ready.")

    def reset(self):
        rng = np.random.default_rng(42)
        r_gauss = np.abs(rng.normal(0, 0.6, cfg.N_PARTICLES))
        r_gauss = np.clip(r_gauss, 0, 2.5)

        theta = rng.random(cfg.N_PARTICLES) * 2 * np.pi
        self.pos = np.zeros((cfg.N_PARTICLES, 2))
        self.pos[:, 0] = r_gauss * np.cos(theta)
        self.pos[:, 1] = r_gauss * np.sin(theta)

        self.vel = np.zeros((cfg.N_PARTICLES, 2))

        # Initial rotational velocity
        self.vel[:, 0] = -self.pos[:, 1] * 0.5
        self.vel[:, 1] = self.pos[:, 0] * 0.5

        self.u = np.full(cfg.N_PARTICLES, 1.0)

        self.damping = True
        self.state = "STABLE"
        self.step_count = 0
        self.time = 0.0
        self.current_G = cfg.G_BASE
        self.remnant_mask[:] = False
        self.collapse_start_time = 0.0

        if hasattr(self, 'remnant_scat'):
            self.remnant_scat.set_offsets(np.empty((0, 2)))

    def on_key(self, event):
        if event.key == ' ':
            self.initiate_collapse()
        elif event.key == 'r':
            self.reset()
        elif event.key == 'q':
            plt.close(self.fig)
        elif event.key == ']':
            self.steps_per_frame = min(self.steps_per_frame + 1, 20)
        elif event.key == '[':
            self.steps_per_frame = max(self.steps_per_frame - 1, 1)

    def initiate_collapse(self):
        if self.state != "STABLE":
            return
        self.state = "COLLAPSING"
        self.damping = False
        self.collapse_start_time = self.time
        self.current_G = cfg.G_BASE * cfg.COLLAPSE_GRAVITY_MULT
        print("Phase 1: CORE COLLAPSE INITIATED")

    def trigger_explosion(self):
        if self.state != "COLLAPSING":
            return
        self.state = "EXPLODING"
        self.current_G = cfg.G_BASE

        r = np.sqrt(np.sum(self.pos ** 2, axis=1))
        sorted_indices = np.argsort(r)

        # Form Neutron Star Core
        remnant_idx = sorted_indices[:cfg.REMNANT_COUNT]
        self.remnant_mask[remnant_idx] = True
        self.vel[remnant_idx] *= 0.1
        self.u[remnant_idx] = 500.0

        # Inject Shockwave Energy
        shell_start = cfg.REMNANT_COUNT
        shell_end = cfg.REMNANT_COUNT + 1000
        shell_idx = sorted_indices[shell_start:shell_end]

        self.u[shell_idx] += cfg.SUPERNOVA_INJECT_E

        dirs = self.pos[shell_idx] / (r[shell_idx, None] + 1e-5)
        self.vel[shell_idx] = dirs * 40.0
        print("Phase 2: NEUTRON STAR FORMED - SHOCKWAVE EJECTED")

    def update(self, frame):
        rho = None
        for _ in range(self.steps_per_frame):
            # Half-step integration
            phys.build_grid(self.pos, self.head, self.next_particle)
            rho, pressure = phys.compute_density_pressure(self.pos, self.u, self.mass_per_particle, self.head,
                                                     self.next_particle)
            acc_grav = phys.compute_gravity(self.pos, self.mass_per_particle, self.current_G)
            acc_sph, du_dt = phys.compute_sph_forces(self.pos, self.vel, rho, pressure, self.mass_per_particle,
                                                self.head, self.next_particle)

            acc = acc_grav + acc_sph
            if self.damping and self.state == "STABLE":
                acc -= self.vel * (1.0 - cfg.DAMP_FACTOR)

            dt = phys.get_dt(rho, pressure, self.vel)
            self.time += dt

            vel_half = self.vel + 0.5 * acc * dt
            u_half = self.u + 0.5 * du_dt * dt
            u_half = np.maximum(u_half, cfg.MIN_TEMP)
            self.pos += vel_half * dt

            # Full-step completion
            phys.build_grid(self.pos, self.head, self.next_particle)
            rho, pressure = phys.compute_density_pressure(self.pos, u_half, self.mass_per_particle, self.head,
                                                     self.next_particle)
            acc_grav = phys.compute_gravity(self.pos, self.mass_per_particle, self.current_G)
            acc_sph, du_dt = phys.compute_sph_forces(self.pos, vel_half, rho, pressure, self.mass_per_particle,
                                                self.head, self.next_particle)

            acc = acc_grav + acc_sph
            if self.damping and self.state == "STABLE":
                acc -= vel_half * (1.0 - cfg.DAMP_FACTOR)

            self.vel = vel_half + 0.5 * acc * dt
            self.u = u_half + 0.5 * du_dt * dt
            self.u = np.maximum(self.u, cfg.MIN_TEMP)

            # State Transitions
            if self.state == "COLLAPSING":
                self.u *= cfg.COLLAPSE_COOL_RATE
                max_rho = np.max(rho)
                duration = self.time - self.collapse_start_time
                if duration > cfg.COLLAPSE_MIN_DURATION and max_rho > cfg.NUCLEAR_DENSITY:
                    self.trigger_explosion()

            elif self.state == "EXPLODING":
                self.vel[self.remnant_mask] *= 0.8
                self.pos[self.remnant_mask] *= 0.98
                self.u[self.remnant_mask] = 500.0

        # Update Visuals
        self.scat.set_offsets(self.pos)
        self.scat.set_array(self.u)

        dynamic_sizes = 3.0 + 12.0 * np.clip(self.u / 200.0, 0, 1.0)
        self.scat.set_sizes(dynamic_sizes)

        if self.state == "EXPLODING":
            remnant_pos = self.pos[self.remnant_mask]
            self.remnant_scat.set_offsets(remnant_pos)
            pulse_size = 40.0 + 10.0 * np.sin(self.step_count * 0.2)
            self.remnant_scat.set_sizes([pulse_size] * len(remnant_pos))
        else:
            self.remnant_scat.set_offsets(np.empty((0, 2)))

        status = f"State: {self.state:<10} | Step: {self.step_count} | Rho_Max: {np.max(rho):.1f}"
        color = "white"
        if self.state == "COLLAPSING":
            color = "#ff4444"
        if self.state == "EXPLODING":
            color = "#ffff44"

        self.text_status.set_text(status)
        self.text_status.set_color(color)

        self.step_count += 1
        return self.scat, self.remnant_scat, self.text_status, self.text_inst

    def run(self):
        self.ani = FuncAnimation(self.fig, self.update, frames=None, interval=1, blit=True, cache_frame_data=False)
        plt.show()