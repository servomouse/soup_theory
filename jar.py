import numpy as np


class Jar:

    def __init__(self, size: int = 1024, num_morphogenes: int = 256, threshold: float = 1e-4,):
        self.size = size
        self.num_morphogenes = num_morphogenes
        self.threshold = threshold
        self.current_tick = 0

        self.diffusion_speeds = np.random.uniform(1.0, 10.0, size=num_morphogenes)
        self.decay_rates = np.random.uniform(0.01, 0.1, size=num_morphogenes)
        self.releases = {i: [] for i in range(num_morphogenes)}

    def add_morphogene(self,cell_coords: tuple[int, int, int], morph_id: int, amount: float,) -> None:
        x, y, z = cell_coords
        self.releases[morph_id].append({
            "pos": np.array([x, y, z], dtype=np.float64),
            "amount": float(amount),
            "release_tick": self.current_tick,
        })

    def tick(self) -> None:
        self.current_tick += 1
        for morph_id in range(self.num_morphogenes):
            decay = self.decay_rates[morph_id]
            active_sources = []
            for source in self.releases[morph_id]:
                source["amount"] *= 1.0 - decay
                if source["amount"] >= self.threshold:
                    active_sources.append(source)
            self.releases[morph_id] = active_sources

    def get_gradient(self, cell_coords: tuple[int, int, int], morph_id: int) -> tuple[int, int, int]:
        sources = self.releases[morph_id]
        if not sources:
            return (1, 1, 1)

        target = np.array(cell_coords, dtype=np.float64)
        D = self.diffusion_speeds[morph_id]

        positions = np.array([s["pos"] for s in sources])
        amounts = np.array([s["amount"] for s in sources])
        release_ticks = np.array([s["release_tick"] for s in sources])

        dt = np.maximum(1, self.current_tick - release_ticks + 1)
        diff = target - positions
        r2 = np.sum(diff**2, axis=1)

        norm_factor = (4.0 * np.pi * D * dt) ** 1.5
        conc_i = (amounts / norm_factor) * np.exp(-r2 / (4.0 * D * dt))
        total_conc = np.sum(conc_i)

        if total_conc < self.threshold:
            return (1, 1, 1)

        grad_components = conc_i[:, np.newaxis] * (
            -diff / (2.0 * D * dt[:, np.newaxis])
        )
        grad_vec = np.sum(grad_components, axis=0)

        gx = 1 if grad_vec[0] >= 0 else -1
        gy = 1 if grad_vec[1] >= 0 else -1
        gz = 1 if grad_vec[2] >= 0 else -1

        return (gx, gy, gz)

    def get_max_gradient(self, cell_coords: tuple[int, int, int], morph_id: int) -> tuple[int, int, int]:
        """Calculates the dominant gradient axis at the cell coordinates.

        Returns a vector with a single non-zero entry (1 or -1) along the axis
        with the highest absolute gradient magnitude. Default: (0, 1, 0).
        """
        sources = self.releases[morph_id]
        if not sources:
            return (0, 1, 0)

        target = np.array(cell_coords, dtype=np.float64)
        D = self.diffusion_speeds[morph_id]

        positions = np.array([s["pos"] for s in sources])
        amounts = np.array([s["amount"] for s in sources])
        release_ticks = np.array([s["release_tick"] for s in sources])

        dt = np.maximum(1, self.current_tick - release_ticks + 1)
        diff = target - positions
        r2 = np.sum(diff**2, axis=1)

        norm_factor = (4.0 * np.pi * D * dt) ** 1.5
        conc_i = (amounts / norm_factor) * np.exp(-r2 / (4.0 * D * dt))
        total_conc = np.sum(conc_i)

        # Fallback to default if total concentration is below threshold
        if total_conc < self.threshold:
            return (0, 1, 0)

        grad_components = conc_i[:, np.newaxis] * (-diff / (2.0 * D * dt[:, np.newaxis]))
        grad_vec = np.sum(grad_components, axis=0)

        abs_grads = np.abs(grad_vec)
        max_idx = int(np.argmax(abs_grads))

        # Fallback if all gradient components evaluate to zero
        if abs_grads[max_idx] == 0:
            return (0, 1, 0)

        res = [0, 0, 0]
        res[max_idx] = 1 if grad_vec[max_idx] >= 0 else -1

        return tuple(res)
