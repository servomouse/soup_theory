import random
import json


class Jar:
    """Represents a 3D water volume of 1024x1024x1024 cells."""

    SIZE = 1024
    REMOVAL_THRESHOLD = 0.001
    GRADIENT_THRESHOLD = 0.001

    def __init__(self, num_compounds=256):
        with open("rates.json") as f:
            rates = json.loads(f.read())
        self.decay_rates = rates["decay_rates"]
        self.diffusion_rates = rates["diffusion_rates"]
        self.num_compounds = num_compounds

        # Memory storage: compound_idx -> {(x, y, z): amount}
        # Only non-zero/active cells are tracked here.
        self.grid = {i: {} for i in range(num_compounds)}

    def add_compound(self, compound_idx, cell_coords, amount):
        """Adds a specific amount of a compound to a target 3D cell coordinate."""
        if not self._is_valid_coord(cell_coords):
            raise ValueError(f"Coordinates {cell_coords} fall outside the jar bounds.")

        current_amount = self.grid[compound_idx].get(cell_coords, 0.0)
        self.grid[compound_idx][cell_coords] = current_amount + amount

    def get_gradient(self, compound_idx, cell_coords):
        """Calculates direction (-1 or 1) of the gradient along X, Y, and Z axes."""
        x, y, z = cell_coords
        compound_grid = self.grid[compound_idx]

        # Read neighbor concentrations (0.0 if not present in memory)
        c_center = compound_grid.get((x, y, z), 0.0)
        c_x_plus = compound_grid.get((x + 1, y, z), 0.0)
        c_x_minus = compound_grid.get((x - 1, y, z), 0.0)
        c_y_plus = compound_grid.get((x, y + 1, z), 0.0)
        c_y_minus = compound_grid.get((x, y - 1, z), 0.0)
        c_z_plus = compound_grid.get((x, y, z + 1), 0.0)
        c_z_minus = compound_grid.get((x, y, z - 1), 0.0)

        # Estimate gradient directional slope (positive shift minus negative shift)
        grad_x = c_x_plus - c_x_minus
        grad_y = c_y_plus - c_y_minus
        grad_z = c_z_plus - c_z_minus

        # Helper to determine orientation: return -1 if negative, otherwise 1
        def direction(grad_val):
            if abs(grad_val) < self.GRADIENT_THRESHOLD:
                return 1
            return 1 if grad_val > 0 else -1

        return (direction(grad_x), direction(grad_y), direction(grad_z))

    def tick(self):
        """Simulates one time step: diffuses compounds, applies decay, and prunes small values."""
        for compound_idx in range(self.num_compounds):
            current_cells = self.grid[compound_idx]
            if not current_cells:
                continue

            next_cells = {}
            diffusion_rate = self.diffusion_rates[compound_idx]
            decay_factor = 1.0 - self.decay_rates[compound_idx]

            # 1. Distribute amounts via simple 6-neighbor 3D diffusion
            for (x, y, z), amount in current_cells.items():
                # Amount migrating out equally into 6 orthogonal neighbor cells
                diffused_out = amount * diffusion_rate
                retained_amount = amount - diffused_out
                amount_per_neighbor = diffused_out / 6.0

                # Keep local remaining amount
                next_cells[(x, y, z)] = next_cells.get((x, y, z), 0.0) + retained_amount

                # Spread to adjacent neighbors within the boundary
                neighbors = [
                    (x + 1, y, z), (x - 1, y, z),
                    (x, y + 1, z), (x, y - 1, z),
                    (x, y, z + 1), (x, y, z - 1)
                ]
                for nx, ny, nz in neighbors:
                    if self._is_valid_coord((nx, ny, nz)):
                        next_cells[(nx, ny, nz)] = next_cells.get((nx, ny, nz), 0.0) + amount_per_neighbor

            # 2. Apply decay rate & prune values below threshold
            pruned_cells = {}
            for coord, amount in next_cells.items():
                decayed_amount = amount * decay_factor
                if decayed_amount >= self.REMOVAL_THRESHOLD:
                    pruned_cells[coord] = decayed_amount

            self.grid[compound_idx] = pruned_cells

    def _is_valid_coord(self, cell_coords):
        """Checks whether coordinate sits within grid bounds."""
        x, y, z = cell_coords
        return 0 <= x < self.SIZE and 0 <= y < self.SIZE and 0 <= z < self.SIZE
