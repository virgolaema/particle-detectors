"""Plotting helpers for the Bethe-Bloch notebook."""

import numpy as np
import matplotlib.pyplot as plt

from lib import calculate_bethe_bloch, material_db as _default_material_db


def plot_materials_comparison(particle, materials_list, energies, material_db=None):
    """Plot Bethe-Bloch for different materials with fixed particle."""
    if material_db is None:
        material_db = _default_material_db

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(materials_list)))

    for material, color in zip(materials_list, colors):
        try:
            dEdx = calculate_bethe_bloch(energies, particle, material, material_db)

            # Plot only valid points
            valid_mask = ~np.isnan(dEdx) & (dEdx > 0)
            if np.any(valid_mask):
                plt.loglog(
                    energies[valid_mask],
                    dEdx[valid_mask],
                    label=material,
                    color=color,
                    linewidth=2.5,
                )
        except Exception as exc:
            print(f"Warning: Could not calculate for {material}: {exc}")

    plt.xlabel("Kinetic Energy (MeV)", fontsize=14)
    plt.ylabel(r"$-dE/dx$ (MeV·cm²/g)", fontsize=14)
    plt.title(
        f"Bethe-Bloch Energy Loss: {particle['name']} in Different Materials",
        fontsize=16,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_particles_comparison(material, particles_list, energies, material_db=None):
    """Plot Bethe-Bloch for different particles in fixed material."""
    if material_db is None:
        material_db = _default_material_db

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(particles_list)))

    for particle, color in zip(particles_list, colors):
        try:
            dEdx = calculate_bethe_bloch(energies, particle, material, material_db)

            # Plot only valid points
            valid_mask = ~np.isnan(dEdx) & (dEdx > 0)
            if np.any(valid_mask):
                label = f"{particle['name']} (m={particle['mass']:.1f} MeV/c²)"
                plt.loglog(
                    energies[valid_mask],
                    dEdx[valid_mask],
                    label=label,
                    color=color,
                    linewidth=2.5,
                )
        except Exception as exc:
            print(f"Warning: Could not calculate for {particle['name']}: {exc}")

    plt.xlabel("Kinetic Energy (MeV)", fontsize=14)
    plt.ylabel(r"$-dE/dx$ (MeV·cm²/g)", fontsize=14)
    plt.title(f"Bethe-Bloch Energy Loss: Different Particles in {material}", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comprehensive_grid(materials_list, particles_list, energies, material_db=None):
    """Create a comprehensive grid plot of all combinations."""
    if material_db is None:
        material_db = _default_material_db

    n_materials = len(materials_list)
    n_particles = len(particles_list)

    fig, axes = plt.subplots(
        n_particles,
        n_materials,
        figsize=(4 * n_materials, 3 * n_particles),
        sharex=True,
        sharey=True,
    )

    if n_particles == 1:
        axes = axes.reshape(1, -1)
    if n_materials == 1:
        axes = axes.reshape(-1, 1)

    for i, particle in enumerate(particles_list):
        for j, material in enumerate(materials_list):
            ax = axes[i, j]

            try:
                dEdx = calculate_bethe_bloch(energies, particle, material, material_db)
                valid_mask = ~np.isnan(dEdx) & (dEdx > 0)

                if np.any(valid_mask):
                    ax.loglog(energies[valid_mask], dEdx[valid_mask], "b-", linewidth=2)

                ax.set_title(f"{particle['name']} in {material}", fontsize=11)
                ax.grid(True, alpha=0.3)

                if i == n_particles - 1:
                    ax.set_xlabel("Energy (MeV)", fontsize=10)
                if j == 0:
                    ax.set_ylabel(r"$-dE/dx$ (MeV·cm²/g)", fontsize=10)

            except Exception as exc:
                ax.text(
                    0.5,
                    0.5,
                    f"Error:\n{str(exc)[:30]}...",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

    plt.tight_layout()
    plt.show()


def interactive_plotter(
    materials_list,
    particles_list,
    energy_min,
    energy_max,
    num_points,
    material_db=None,
):
    """Interactive function to quickly plot specific cases."""
    if material_db is None:
        material_db = _default_material_db

    print("Interactive Bethe-Bloch Plotter")
    print("Available materials:", materials_list)
    print("Available particles:", [p["name"] for p in particles_list])

    material = input("Enter material name: ")
    particle_name = input("Enter particle name: ")

    # Find particle
    particle = None
    for item in particles_list:
        if item["name"].lower() == particle_name.lower():
            particle = item
            break

    if particle is None:
        print(f"Particle {particle_name} not found!")
        return

    # Generate energies and plot
    energies = np.logspace(np.log10(energy_min), np.log10(energy_max), num_points)

    try:
        dEdx = calculate_bethe_bloch(energies, particle, material, material_db)
        valid_mask = ~np.isnan(dEdx) & (dEdx > 0)

        plt.figure(figsize=(10, 6))
        plt.loglog(energies[valid_mask], dEdx[valid_mask], "b-", linewidth=2.5)
        plt.xlabel("Kinetic Energy (MeV)", fontsize=14)
        plt.ylabel(r"$-dE/dx$ (MeV·cm²/g)", fontsize=14)
        plt.title(f"Bethe-Bloch: {particle['name']} in {material}", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as exc:
        print(f"Error: {exc}")
