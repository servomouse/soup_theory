import os
import numpy as np
from PIL import Image

from jar import Jar 

def compute_plane_slice(
    jar: Jar,
    morph_id: int,
    plane: str,
    fixed_coord: int,
    resolution: int = 256,
) -> np.ndarray:
    """Vectorized calculation of gradients across a 2D slice plane."""
    sources = jar.releases[morph_id]
    H, W = resolution, resolution

    # Coordinate grids across full Jar dimensions (1024x1024 plane)
    axis_range = np.linspace(0, jar.size - 1, resolution)
    grid_a, grid_b = np.meshgrid(axis_range, axis_range)
    fixed_grid = np.full((H, W), fixed_coord, dtype=np.float64)

    if plane == "XY":
        coords = np.stack([grid_a, grid_b, fixed_grid], axis=-1)  # (H, W, 3)
    elif plane == "XZ":
        coords = np.stack([grid_a, fixed_grid, grid_b], axis=-1)  # (H, W, 3)
    elif plane == "YZ":
        coords = np.stack([fixed_grid, grid_a, grid_b], axis=-1)  # (H, W, 3)

    if not sources:
        # Default below threshold returns white background (255, 255, 255)
        return np.full((H, W, 3), 255, dtype=np.uint8)

    D = jar.diffusion_speeds[morph_id]
    total_conc = np.zeros((H, W), dtype=np.float64)
    total_grad = np.zeros((H, W, 3), dtype=np.float64)

    for source in sources:
        pos = source["pos"]
        amount = source["amount"]
        dt = max(1, jar.current_tick - source["release_tick"] + 1)

        diff = coords - pos  # Displacement shape: (H, W, 3)
        r2 = np.sum(diff**2, axis=-1)

        norm_factor = (4.0 * np.pi * D * dt) ** 1.5
        conc_i = (amount / norm_factor) * np.exp(-r2 / (4.0 * D * dt))

        total_conc += conc_i
        total_grad += conc_i[..., np.newaxis] * (
            -diff / (2.0 * D * dt)
        )

    # Gradient Direction to Color Mapping
    # R: X-axis, G: Y-axis, B: Z-axis (255 for >=0 positive gradient, 0 for negative)
    r = np.where(total_grad[..., 0] >= 0, 255, 0).astype(np.uint8)
    g = np.where(total_grad[..., 1] >= 0, 255, 0).astype(np.uint8)
    b = np.where(total_grad[..., 2] >= 0, 255, 0).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)

    # Below threshold -> fallback default gradient (1, 1, 1) = White (255, 255, 255)
    below_thresh = total_conc < jar.threshold
    rgb[below_thresh] = [255, 255, 255]

    return rgb


def run_visualization_simulation(
    num_ticks: int = 50, output_dir: str = "slice_images"
):
    os.makedirs(output_dir, exist_ok=True)

    jar = Jar(size=1024, num_morphogenes=256, threshold=1e-4)

    # Custom diffusion parameters for morphogene 0 for smooth visualization
    morph_id = 0
    jar.diffusion_speeds[morph_id] = 150.0  # Diffusion speed
    jar.decay_rates[morph_id] = 0.04  # Decay rate

    # Release morphogene at the center of the 1024^3 jar
    center = (512, 512, 512)
    jar.add_morphogene(center, morph_id=morph_id, amount=1000000.0)

    print(f"Generating slice images in folder: '{output_dir}/'...")

    for step in range(num_ticks):
        for plane in ["XY", "XZ", "YZ"]:
            # Extract fixed coordinate for slice (center of jar)
            fixed_coord = 512

            rgb_image = compute_plane_slice(
                jar,
                morph_id=morph_id,
                plane=plane,
                fixed_coord=fixed_coord,
                resolution=256,
            )

            filename = f"{plane}_{step:03d}.bmp"
            filepath = os.path.join(output_dir, filename)

            # Save as BMP file
            img = Image.fromarray(rgb_image, mode="RGB")
            img.save(filepath)

        print(f"Saved tick {step:02d}/{num_ticks - 1}")
        jar.tick()

    print("\nVisualization slice generation complete!")


if __name__ == "__main__":
    run_visualization_simulation(num_ticks=40)