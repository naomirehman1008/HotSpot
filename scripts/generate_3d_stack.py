#!/usr/bin/env python3
"""
HotSpot 3D Stacked Chip Input File Generator

Generates input files (.lcf, .flp, .ptrace, .config, run.sh) for HotSpot
thermal simulation of 3D stacked chips.

Each tier in the stack consists of (bottom to top within the tier):
  1. Silicon die (active, power-dissipating)
  2. Metal/dielectric interconnect layer (modeled via Effective Medium
     Approximation / Rule of Mixtures)
  3. TSV/bonding layer connecting to the next tier (heterogeneous blocks
     for detailed_3D mode)

The topmost tier replaces the TSV/bonding layer with a TIM layer to the
heat sink.

Usage:
    python generate_3d_stack.py -n 4                     # 4-tier stack
    python generate_3d_stack.py -n 2 -o my_stack         # custom output dir
    python generate_3d_stack.py -n 3 --power-per-layer 5 # 5W per tier
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass


@dataclass
class StackConfig:
    """Configuration for 3D stacked chip input file generation.

    The ``num_layers`` field is the primary tunable parameter. All other
    fields provide physical defaults that will become individually tunable
    in future iterations.
    """

    # -- Primary tunable parameter ------------------------------------------
    num_layers: int = 2

    # -- Chip geometry (meters) ---------------------------------------------
    chip_width: float = 0.010   # 10 mm
    chip_height: float = 0.010  # 10 mm

    # -- Silicon die --------------------------------------------------------
    silicon_thickness: float = 50e-6      # 50 um (thinned for 3D stacking)
    silicon_conductivity: float = 130.0   # W/(m-K)
    silicon_specific_heat: float = 1.75e6 # J/(m^3-K)

    # -- Metal / dielectric (BEOL) layer ------------------------------------
    # Individual material properties used for EMA calculation.
    metal_dielectric_thickness: float = 10e-6  # 10 um
    metal_conductivity: float = 401.0          # copper, W/(m-K)
    metal_specific_heat: float = 3.42e6        # copper, J/(m^3-K)
    dielectric_conductivity: float = 1.4       # SiO2, W/(m-K)
    dielectric_specific_heat: float = 1.65e6   # SiO2, J/(m^3-K)
    metal_fill_fraction: float = 0.30          # volume fraction of metal

    # -- TSV / bonding layer ------------------------------------------------
    tsv_bonding_thickness: float = 20e-6  # 20 um
    # TSV material (copper)
    tsv_conductivity: float = 401.0       # W/(m-K)
    tsv_specific_heat: float = 3.42e6     # J/(m^3-K)
    # Bonding dielectric / underfill (bulk properties of the layer)
    bonding_conductivity: float = 0.2     # W/(m-K)
    bonding_specific_heat: float = 2.0e6  # J/(m^3-K)
    # TSV geometry
    num_tsv_strips: int = 2       # number of horizontal TSV strip regions
    tsv_area_fraction: float = 0.05  # fraction of bonding layer area that is TSV

    # -- Top TIM layer (between topmost tier and heat sink) -----------------
    tim_thickness: float = 20e-6      # 20 um
    tim_conductivity: float = 4.0     # W/(m-K)
    tim_specific_heat: float = 4.0e6  # J/(m^3-K)

    # -- Power --------------------------------------------------------------
    power_per_layer: float = 10.0  # watts per silicon layer (uniform)
    num_power_samples: int = 1     # number of time-step rows in ptrace

    # -- Grid model ---------------------------------------------------------
    grid_rows: int = 64
    grid_cols: int = 64

    # -- Simulation ---------------------------------------------------------
    ambient_temp: float = 318.15       # kelvin (45 C)
    sampling_interval: float = 3.333e-6  # seconds

    # -- Output -------------------------------------------------------------
    output_dir: str = "generated_3d_stack"


# ---------------------------------------------------------------------------
# Effective Medium Approximation (Rule of Mixtures)
# ---------------------------------------------------------------------------

def effective_medium(k1: float, p1: float,
                     k2: float, p2: float,
                     f1: float) -> tuple[float, float]:
    """Compute effective thermal properties via the Rule of Mixtures.

    For a composite of two materials with volume fraction *f1* of material 1:

        k_eff = f1 * k1 + (1 - f1) * k2
        p_eff = f1 * p1 + (1 - f1) * p2

    Parameters
    ----------
    k1, k2 : float
        Thermal conductivity of material 1 and 2 [W/(m-K)].
    p1, p2 : float
        Volumetric heat capacity of material 1 and 2 [J/(m^3-K)].
    f1 : float
        Volume fraction of material 1 (0-1).

    Returns
    -------
    (k_eff, p_eff)
    """
    k_eff = f1 * k1 + (1.0 - f1) * k2
    p_eff = f1 * p1 + (1.0 - f1) * p2
    return k_eff, p_eff


# ---------------------------------------------------------------------------
# Floorplan generators
# ---------------------------------------------------------------------------

def _flp_line(name: str, width: float, height: float,
              left_x: float, bottom_y: float,
              specific_heat: float | None = None,
              resistivity: float | None = None) -> str:
    """Format a single .flp line."""
    line = f"{name}\t{width:.6e}\t{height:.6e}\t{left_x:.6e}\t{bottom_y:.6e}"
    if specific_heat is not None and resistivity is not None:
        line += f"\t{specific_heat:.6e}\t{resistivity:.6e}"
    return line + "\n"


def generate_silicon_flp(cfg: StackConfig, tier: int, path: str) -> list[str]:
    """Single monolithic block covering the full chip area."""
    name = f"silicon_{tier}"
    with open(path, "w") as fh:
        fh.write(f"# Silicon die – tier {tier}\n")
        fh.write(_flp_line(name, cfg.chip_width, cfg.chip_height, 0.0, 0.0))
    return [name]


def generate_metal_dielectric_flp(cfg: StackConfig, tier: int,
                                  path: str) -> list[str]:
    """Single monolithic block; layer-level EMA properties are set in the LCF."""
    name = f"metal_dielectric_{tier}"
    with open(path, "w") as fh:
        fh.write(f"# Metal/dielectric (BEOL) – tier {tier}\n")
        fh.write(f"# Effective medium: metal fill fraction = {cfg.metal_fill_fraction}\n")
        fh.write(_flp_line(name, cfg.chip_width, cfg.chip_height, 0.0, 0.0))
    return [name]


def generate_tsv_bonding_flp(cfg: StackConfig, tier: int,
                             path: str) -> list[str]:
    """Heterogeneous floorplan with alternating dielectric and TSV strips.

    TSV strips carry per-block R-C overrides for ``detailed_3D`` mode.
    The bulk layer properties (set in the LCF) correspond to the bonding
    dielectric; TSV strips override with copper-like values.
    """
    names: list[str] = []

    total_tsv_height = cfg.tsv_area_fraction * cfg.chip_height
    tsv_strip_height = total_tsv_height / cfg.num_tsv_strips

    total_dielectric_height = cfg.chip_height - total_tsv_height
    num_dielectric_blocks = cfg.num_tsv_strips + 1
    dielectric_block_height = total_dielectric_height / num_dielectric_blocks

    tsv_resistivity = 1.0 / cfg.tsv_conductivity

    with open(path, "w") as fh:
        fh.write(f"# TSV/bonding layer – tier {tier}\n")
        fh.write(f"# Heterogeneous: dielectric blocks + TSV strips (detailed_3D)\n")
        fh.write(f"# TSV area fraction = {cfg.tsv_area_fraction}\n")

        y = 0.0
        dielectric_idx = 0

        for strip in range(cfg.num_tsv_strips):
            # --- dielectric block ---
            name = f"bond_d_{tier}_{dielectric_idx}"
            fh.write(_flp_line(name, cfg.chip_width, dielectric_block_height,
                               0.0, y))
            names.append(name)
            y += dielectric_block_height
            dielectric_idx += 1

            # --- TSV strip (custom R-C) ---
            name = f"tsv_{tier}_{strip}"
            fh.write(_flp_line(name, cfg.chip_width, tsv_strip_height,
                               0.0, y,
                               specific_heat=cfg.tsv_specific_heat,
                               resistivity=tsv_resistivity))
            names.append(name)
            y += tsv_strip_height

        # --- final dielectric block (absorbs any FP rounding) ---
        remaining = cfg.chip_height - y
        name = f"bond_d_{tier}_{dielectric_idx}"
        fh.write(_flp_line(name, cfg.chip_width, remaining, 0.0, y))
        names.append(name)

    return names


def generate_tim_top_flp(cfg: StackConfig, path: str) -> list[str]:
    """Single block TIM layer between top tier and heat sink."""
    name = "tim_top"
    with open(path, "w") as fh:
        fh.write("# Top TIM layer (chip to heat sink)\n")
        fh.write(_flp_line(name, cfg.chip_width, cfg.chip_height, 0.0, 0.0))
    return [name]


# ---------------------------------------------------------------------------
# Layer configuration file (.lcf)
# ---------------------------------------------------------------------------

def _lcf_layer(fh, layer_num: int, lateral: str, power: str,
               specific_heat: float, resistivity: float,
               thickness: float, flp_file: str,
               comment: str = "") -> None:
    """Write one layer entry to an LCF file."""
    if comment:
        fh.write(f"# {comment}\n")
    fh.write(f"{layer_num}\n")
    fh.write(f"{lateral}\n")
    fh.write(f"{power}\n")
    fh.write(f"{specific_heat:.6e}\n")
    fh.write(f"{resistivity:.6e}\n")
    fh.write(f"{thickness:.6e}\n")
    fh.write(f"{flp_file}\n\n")


def generate_lcf(cfg: StackConfig, flp_files: list[str],
                 path: str) -> None:
    """Generate the layer configuration file.

    Layer ordering (bottom to top)::

        For each tier 0 .. N-1:
            silicon  (Power = Y)
            metal/dielectric  (Power = N, EMA properties)
            tsv/bonding  (Power = N, heterogeneous)   [omitted for top tier]
        Top tier ends with:
            TIM  (Power = N)
    """
    k_eff, p_eff = effective_medium(
        cfg.metal_conductivity, cfg.metal_specific_heat,
        cfg.dielectric_conductivity, cfg.dielectric_specific_heat,
        cfg.metal_fill_fraction,
    )
    r_eff = 1.0 / k_eff

    with open(path, "w") as fh:
        fh.write("# Layer Configuration File – 3D Stacked Chip\n")
        fh.write("# Generated by generate_3d_stack.py\n")
        fh.write("#\n")
        fh.write("# <Layer Number>\n")
        fh.write("# <Lateral heat flow Y/N?>\n")
        fh.write("# <Power Dissipation Y/N?>\n")
        fh.write("# <Specific heat capacity in J/(m^3K)>\n")
        fh.write("# <Resistivity in (m-K)/W>\n")
        fh.write("# <Thickness in m>\n")
        fh.write("# <floorplan file>\n\n")

        layer_num = 0
        flp_idx = 0

        for tier in range(cfg.num_layers):
            # --- silicon ---
            _lcf_layer(fh, layer_num, "Y", "Y",
                       cfg.silicon_specific_heat,
                       1.0 / cfg.silicon_conductivity,
                       cfg.silicon_thickness,
                       flp_files[flp_idx],
                       comment=f"Tier {tier}: Silicon die")
            layer_num += 1
            flp_idx += 1

            # --- metal / dielectric (EMA) ---
            _lcf_layer(fh, layer_num, "Y", "N",
                       p_eff, r_eff,
                       cfg.metal_dielectric_thickness,
                       flp_files[flp_idx],
                       comment=(f"Tier {tier}: Metal/dielectric "
                                f"(EMA k_eff={k_eff:.2f} W/(m-K), "
                                f"fill={cfg.metal_fill_fraction})"))
            layer_num += 1
            flp_idx += 1

            if tier < cfg.num_layers - 1:
                # --- TSV / bonding ---
                _lcf_layer(fh, layer_num, "Y", "N",
                           cfg.bonding_specific_heat,
                           1.0 / cfg.bonding_conductivity,
                           cfg.tsv_bonding_thickness,
                           flp_files[flp_idx],
                           comment=f"Tier {tier}: TSV/bonding (heterogeneous)")
                layer_num += 1
                flp_idx += 1
            else:
                # --- top TIM ---
                _lcf_layer(fh, layer_num, "Y", "N",
                           cfg.tim_specific_heat,
                           1.0 / cfg.tim_conductivity,
                           cfg.tim_thickness,
                           flp_files[flp_idx],
                           comment="Top TIM (to heat sink)")
                layer_num += 1
                flp_idx += 1


# ---------------------------------------------------------------------------
# Power trace (.ptrace)
# ---------------------------------------------------------------------------

def generate_ptrace(cfg: StackConfig, silicon_names: list[str],
                    path: str) -> None:
    """Generate a power trace with uniform power on each silicon layer.

    Only blocks from power-dissipating layers (silicon) appear in the
    ptrace.  Each silicon layer is a single block, so it receives the
    full ``power_per_layer`` value.
    """
    with open(path, "w") as fh:
        fh.write("\t".join(silicon_names) + "\n")
        power_values = [f"{cfg.power_per_layer}" for _ in silicon_names]
        row = "\t".join(power_values)
        for _ in range(cfg.num_power_samples):
            fh.write(row + "\n")


# ---------------------------------------------------------------------------
# HotSpot configuration file
# ---------------------------------------------------------------------------

def generate_config(cfg: StackConfig, lcf_file: str, path: str) -> None:
    with open(path, "w") as fh:
        fh.write("# HotSpot configuration – 3D stacked chip\n")
        fh.write("# Generated by generate_3d_stack.py\n\n")

        fh.write("\t# chip specs\n")
        fh.write(f"\t-t_chip\t\t\t\t{cfg.silicon_thickness:.6e}\n")
        fh.write(f"\t-k_chip\t\t\t\t{cfg.silicon_conductivity}\n")
        fh.write(f"\t-p_chip\t\t\t\t{cfg.silicon_specific_heat:.6e}\n")
        fh.write(f"\t-thermal_threshold\t354.95\n\n")

        fh.write("\t# heat sink\n")
        fh.write("\t-c_convec\t\t\t140.4\n")
        fh.write("\t-r_convec\t\t\t0.1\n")
        fh.write("\t-s_sink\t\t\t\t0.06\n")
        fh.write("\t-t_sink\t\t\t\t0.0069\n")
        fh.write("\t-k_sink\t\t\t\t400.0\n")
        fh.write("\t-p_sink\t\t\t\t3.55e6\n\n")

        fh.write("\t# heat spreader\n")
        fh.write("\t-s_spreader\t\t\t0.03\n")
        fh.write("\t-t_spreader\t\t\t0.001\n")
        fh.write("\t-k_spreader\t\t\t400.0\n")
        fh.write("\t-p_spreader\t\t\t3.55e6\n\n")

        fh.write("\t# interface material\n")
        fh.write(f"\t-t_interface\t\t{cfg.tim_thickness:.6e}\n")
        fh.write(f"\t-k_interface\t\t{cfg.tim_conductivity}\n")
        fh.write(f"\t-p_interface\t\t{cfg.tim_specific_heat:.6e}\n\n")

        fh.write("\t# secondary path (disabled)\n")
        fh.write("\t-model_secondary\t0\n\n")

        fh.write("\t# simulation\n")
        fh.write(f"\t-ambient\t\t\t{cfg.ambient_temp}\n")
        fh.write(f"\t-init_temp\t\t\t{cfg.ambient_temp}\n")
        fh.write(f"\t-sampling_intvl\t\t{cfg.sampling_interval:.6e}\n")
        fh.write("\t-base_proc_freq\t\t3e+09\n")
        fh.write("\t-dtm_used\t\t\t0\n")
        fh.write("\t-model_type\t\t\tgrid\n")
        fh.write("\t-leakage_used\t\t0\n\n")

        fh.write("\t# grid model\n")
        fh.write(f"\t-grid_rows\t\t\t{cfg.grid_rows}\n")
        fh.write(f"\t-grid_cols\t\t\t{cfg.grid_cols}\n")
        fh.write(f"\t-grid_layer_file\t{lcf_file}\n")
        fh.write("\t-grid_map_mode\t\tcenter\n")


# ---------------------------------------------------------------------------
# Run script
# ---------------------------------------------------------------------------

def generate_run_script(cfg: StackConfig, config_file: str,
                        ptrace_file: str, lcf_file: str,
                        path: str) -> None:
    with open(path, "w") as fh:
        fh.write("#!/usr/bin/env bash\n")
        fh.write("# Run script – 3D stacked chip simulation\n")
        fh.write("# Generated by generate_3d_stack.py\n\n")
        fh.write("set -e\n\n")

        fh.write('HOTSPOT="${HOTSPOT:-../../hotspot}"\n\n')

        fh.write("rm -f *.init\n")
        fh.write("rm -rf outputs\n")
        fh.write("mkdir -p outputs\n\n")

        fh.write("# Steady-state\n")
        fh.write(f"$HOTSPOT -c {config_file} -p {ptrace_file}"
                 f" -grid_layer_file {lcf_file}"
                 f" -model_type grid -detailed_3D on"
                 f" -steady_file outputs/steady.out"
                 f" -grid_steady_file outputs/grid_steady.out\n\n")

        fh.write("cp outputs/steady.out initial.init\n\n")

        fh.write("# Transient\n")
        fh.write(f"$HOTSPOT -c {config_file} -p {ptrace_file}"
                 f" -grid_layer_file {lcf_file}"
                 f" -init_file initial.init -model_type grid -detailed_3D on"
                 f" -o outputs/transient.ttrace"
                 f" -grid_transient_file outputs/grid_transient.ttrace\n")

    os.chmod(path, 0o755)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def generate_3d_stack(cfg: StackConfig) -> None:
    """Generate all HotSpot input files for a 3D stacked chip."""
    os.makedirs(cfg.output_dir, exist_ok=True)

    flp_files: list[str] = []       # filenames in layer order (for LCF)
    silicon_names: list[str] = []   # block names of power-dissipating layers

    for tier in range(cfg.num_layers):
        # --- silicon ---
        fname = f"silicon_{tier}.flp"
        names = generate_silicon_flp(
            cfg, tier, os.path.join(cfg.output_dir, fname))
        flp_files.append(fname)
        silicon_names.extend(names)

        # --- metal / dielectric ---
        fname = f"metal_dielectric_{tier}.flp"
        generate_metal_dielectric_flp(
            cfg, tier, os.path.join(cfg.output_dir, fname))
        flp_files.append(fname)

        # --- TSV / bonding  OR  top TIM ---
        if tier < cfg.num_layers - 1:
            fname = f"tsv_bonding_{tier}.flp"
            generate_tsv_bonding_flp(
                cfg, tier, os.path.join(cfg.output_dir, fname))
        else:
            fname = "tim_top.flp"
            generate_tim_top_flp(cfg, os.path.join(cfg.output_dir, fname))
        flp_files.append(fname)

    # --- LCF ---
    lcf_file = "stack.lcf"
    generate_lcf(cfg, flp_files, os.path.join(cfg.output_dir, lcf_file))

    # --- power trace ---
    ptrace_file = "power.ptrace"
    generate_ptrace(cfg, silicon_names,
                    os.path.join(cfg.output_dir, ptrace_file))

    # --- HotSpot config ---
    config_file = "hotspot.config"
    generate_config(cfg, lcf_file,
                    os.path.join(cfg.output_dir, config_file))

    # --- run script ---
    generate_run_script(cfg, config_file, ptrace_file, lcf_file,
                        os.path.join(cfg.output_dir, "run.sh"))

    # --- summary ---
    k_eff, p_eff = effective_medium(
        cfg.metal_conductivity, cfg.metal_specific_heat,
        cfg.dielectric_conductivity, cfg.dielectric_specific_heat,
        cfg.metal_fill_fraction,
    )

    total_lcf_layers = cfg.num_layers * 3  # silicon + MD + (TSV or TIM)
    print(f"Generated {cfg.num_layers}-tier 3D stack in '{cfg.output_dir}/'")
    print(f"  LCF layers:           {total_lcf_layers}")
    print(f"  Silicon layers:       {cfg.num_layers}  (power-dissipating)")
    print(f"  Metal/dielectric:     {cfg.num_layers}  (EMA)")
    print(f"  TSV/bonding layers:   {cfg.num_layers - 1}")
    print(f"  TIM top layer:        1")
    print()
    print(f"  Metal/dielectric EMA (fill = {cfg.metal_fill_fraction}):")
    print(f"    k_eff = {k_eff:.4f} W/(m-K)")
    print(f"    p_eff = {p_eff:.6e} J/(m^3-K)")
    print()
    print("  Files:")
    for f in sorted(os.listdir(cfg.output_dir)):
        print(f"    {f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HotSpot input files for 3D stacked chips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n", "--num-layers", type=int, default=2,
        help="Number of silicon tiers in the stack",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="generated_3d_stack",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--chip-width", type=float, default=0.010,
        help="Chip width in meters",
    )
    parser.add_argument(
        "--chip-height", type=float, default=0.010,
        help="Chip height in meters",
    )
    parser.add_argument(
        "--power-per-layer", type=float, default=10.0,
        help="Uniform power per silicon layer in Watts",
    )

    args = parser.parse_args()

    if args.num_layers < 1:
        parser.error("--num-layers must be >= 1")

    cfg = StackConfig(
        num_layers=args.num_layers,
        output_dir=args.output_dir,
        chip_width=args.chip_width,
        chip_height=args.chip_height,
        power_per_layer=args.power_per_layer,
    )

    generate_3d_stack(cfg)


if __name__ == "__main__":
    main()
