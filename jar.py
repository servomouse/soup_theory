import numpy as np
import json
import math

CELL_RADIUS = 0.5

class Jar:
    def __init__(self, jar_config):
        self.cube_size = jar_config["size"]
        self.num_morphogenes = jar_config["num_morphogenes"]
        self.threshold = 0.001
        self.current_tick = 0

        with open(jar_config["rates_file"]) as f:
            rates_data = json.loads(f.read())

        self.morphogenes = [
            {
                "releases": [],
                "diffusion_rate": rates_data["diffusion_rates"][i],
                "decay_rate": rates_data["decay_rates"][i]
            } for i in range(self.num_morphogenes)
        ]

    def add_morphogene(self, cell_coords, morph_idx, amount) -> None:
        x, y, z = cell_coords
        self.morphogenes[morph_idx]["releases"].append({"amount": amount, "coords": [x, y, z], "ticks": 0})

    def tick(self) -> None:
        for morph in self.morphogenes:
            self.update_sources(morph)

    def get_total_concentration(self, cell_coords, morph_idx):
        x, y, z = cell_coords
        if not (0 <= x <= self.cube_size and 0 <= y <= self.cube_size and 0 <= z <= self.cube_size):
            return 0.0

        total_concentration = 0.0
        morph = self.morphogenes[morph_idx]

        for source in morph["releases"]:
            amount = source["amount"]
            source_coords = source["coords"]
            ticks = source["ticks"]

            # Step 1: Distance and expansion boundary check
            distance = math.dist(source_coords, cell_coords)
            plume_radius = CELL_RADIUS + (morph["diffusion_rate"] * ticks)

            if distance > plume_radius:
                continue  # Target is outside the sphere

            # Step 2: Calculate uniform concentration (Amount / Sphere Volume)
            volume = (4.0 / 3.0) * math.pi * (plume_radius ** 3)
            base_concentration = amount / volume

            # Step 3: Apply per-tick decay
            decay_factor = (1.0 - morph["decay_rate"]) ** ticks
            concentration = base_concentration * decay_factor

            # Step 4: Accumulate if above threshold
            if concentration >= self.threshold:
                total_concentration += concentration

        return total_concentration

    def update_sources(self, morph):
        active_sources = []

        for source in morph["releases"]:
            # Step 1: Increment simulation tick
            source["ticks"] += 1

            # Step 2: Calculate expanded plume volume
            plume_radius = CELL_RADIUS + (morph["diffusion_rate"] * source["ticks"])
            volume = (4.0 / 3.0) * math.pi * (plume_radius ** 3)

            # Step 3: Calculate current uniform concentration inside the sphere
            base_concentration = source["amount"] / volume
            decay_factor = (1.0 - morph["decay_rate"]) ** source["ticks"]
            current_concentration = base_concentration * decay_factor

            # Step 4: Keep only sources that still meet the threshold
            if current_concentration >= self.threshold:
                active_sources.append(source)

        # Modify the original list in-place safely
        morph["releases"][:] = active_sources

    def get_default_direction(self, target_coords):
        """Calculates fallback direction pointing toward the center along the most displaced axis."""
        center = self.cube_size / 2.0
        x, y, z = target_coords

        offsets = [center - x, center - y, center - z]
        abs_offsets = [abs(o) for o in offsets]
        max_distance = max(abs_offsets)

        # If target is at the exact center, return [0, 1, 0]
        if max_distance == 0:
            return [0, 1, 0]

        # Find the axis farthest from center and build a unit vector toward center
        for i, offset in enumerate(offsets):
            if abs(offset) == max_distance:
                direction = [0, 0, 0]
                direction[i] = 1 if offset > 0 else -1
                return direction

        return [0, 1, 0]

    def get_gradient(self, cell_coords, morph_idx):
        default_direction = self.get_default_direction(cell_coords)

        try:
            x, y, z = cell_coords

            # Step 1: Get concentration at the 6 neighboring face cells
            c_px = self.get_total_concentration([x + 1, y, z], morph_idx)
            c_nx = self.get_total_concentration([x - 1, y, z], morph_idx)

            c_py = self.get_total_concentration([x, y + 1, z], morph_idx)
            c_ny = self.get_total_concentration([x, y - 1, z], morph_idx)

            c_pz = self.get_total_concentration([x, y, z + 1], morph_idx)
            c_nz = self.get_total_concentration([x, y, z - 1], morph_idx)

            # Step 2: Compute change along each axis
            grad_x = c_px - c_nx
            grad_y = c_py - c_ny
            grad_z = c_pz - c_nz

            # Step 3: Pair gradient magnitudes with their 1D direction vectors
            axis_gradients = [
                (abs(grad_x), [1 if grad_x > 0 else -1, 0, 0]),
                (abs(grad_y), [0, 1 if grad_y > 0 else -1, 0]),
                (abs(grad_z), [0, 0, 1 if grad_z > 0 else -1]),
            ]

            # Step 4: Find the axis with the largest gradient magnitude
            max_magnitude, max_direction = max(axis_gradients, key=lambda item: item[0])

            # Step 5: Return direction if steep enough, otherwise default
            if max_magnitude >= self.threshold:
                return max_direction
            return default_direction

        except Exception:
            # Fallback if target/neighbors cannot be calculated
            return default_direction


if __name__ == "__main__":
    j = Jar({
        "size": 256,
        "num_morphogenes": 32,
        "rates_file": "rates_test.json"
    })
    j.add_morphogene([10, 10, 10], 3, 128)
    for i in range(100):
        print(f"Concentration at step {i}: {j.get_total_concentration([10, 10, 10], 3):.3f}", end="")
        print(f" {j.get_total_concentration([11, 10, 10], 3):.3f}", end="")
        print(f" {j.get_total_concentration([12, 10, 10], 3):.3f}", end="")
        print(f" {j.get_total_concentration([13, 10, 10], 3):.3f}", end="")
        print(f" {j.get_total_concentration([14, 10, 10], 3):.3f}", end="")
        print(f" {j.get_total_concentration([15, 10, 10], 3):.3f}")
        j.tick()
