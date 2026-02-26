from itertools import combinations
import numpy as np


class Gas:
    def __init__(self, width: float, height: float, particles: np.ndarray = None, n_particles: int = None):
        self.width = width
        self.height = height
        
        if particles is not None:
            self.particles = np.array(particles, dtype=float)
            self.n = len(self.particles)
        elif n_particles is not None:
            self.n = n_particles
            self.particles = self.__get_random_distribution()
        else:
            raise ValueError("Must provide either a list of particles or n_particles.")
            
        self.speed = np.random.randn(self.n, 2)
    
    def step(self, dt: float = 1e-2, n_grid: int = 10, radius: float = 1.0):
        self.particles += self.speed * dt
        self.__check_wall_colisions()
        self.__check_particle_colisions(n_grid=n_grid, radius=radius)

    def get_particles(self) -> np.ndarray:
        return self.particles
    
    def compute_entropy(self, n_grid: int) -> float:
        cell_indices = self.__get_grid_indices(n_grid=n_grid)

        _, counts = np.unique(cell_indices, return_counts=True)
        probabilities = counts / self.n
        entropy = -np.sum(probabilities * np.log(probabilities))        
        return float(entropy)
    
    def simulate(self, n_steps: int, dt: float, n_grid, radius):
        pass

    def __get_grid_indices(self, n_grid: int) -> np.ndarray:
        delta_w = self.width / n_grid
        delta_h = self.height / n_grid

        w_idx = (self.particles[:, 0] // delta_w).astype(int)
        h_idx = (self.particles[:, 1] // delta_h).astype(int)
        
        w_idx = np.clip(w_idx, 0, n_grid - 1)
        h_idx = np.clip(h_idx, 0, n_grid - 1)
        
        cell_indices = n_grid * w_idx + h_idx
        return cell_indices

    def __check_wall_colisions(self):
        hit_left = self.particles[:, 0] <= 0
        hit_right = self.particles[:, 0] >= self.width
        self.speed[hit_left | hit_right, 0] *= -1
        
        hit_bottom = self.particles[:, 1] <= 0
        hit_top = self.particles[:, 1] >= self.height
        self.speed[hit_bottom | hit_top, 1] *= -1
        
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.width)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.height)

    def __get_random_distribution(self) -> np.ndarray:
        return np.random.rand(self.n, 2) * np.array([self.width, self.height])

    def __check_particle_colisions(self, n_grid: int, radius: float = 1.0):
        cell_indices = self.__get_grid_indices(n_grid=n_grid)
        
        sort_idx = np.argsort(cell_indices)
        sorted_cells = cell_indices[sort_idx]
        
        unique_cells, start_indices = np.unique(sorted_cells, return_index=True)
        cells_particle_indices = np.split(sort_idx, start_indices[1:])
        
        
        for cell_p_idx in cells_particle_indices:
            if len(cell_p_idx) < 2:
                continue
                
            for i, j in combinations(cell_p_idx, 2):
                pos_i = self.particles[i]
                pos_j = self.particles[j]
                
                r_vec = pos_i - pos_j
                dist_sq = np.sum(r_vec**2)
                
                min_dist = 2 * radius
                if dist_sq < min_dist**2 and dist_sq > 0:
                    v_i = self.speed[i]
                    v_j = self.speed[j]
                    v_rel = v_i - v_j
                    
                    dot_product = np.dot(v_rel, r_vec)
                    
                    if dot_product < 0:
                        impulse = (dot_product / dist_sq) * r_vec
                        
                        self.speed[i] -= impulse
                        self.speed[j] += impulse
