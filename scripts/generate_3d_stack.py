#!/usr/bin/env python3
"""
HotSpot 3D Stacked Chip Input File Generator

Generates input files (.lcf, .flp, .ptrace, .config, run.sh) for HotSpot
thermal simulation of 3D stacked chips.

Each tier in the stack consists of (bottom to top within the tier):
  1. Silicon die (active, power-dissipating)
  2. Per-metal-level BEOL layers (M1-M9, each with its own EMA-computed
     effective thermal conductivity based on pitch/width/fill at that level)
  3. TSV/bonding layer connecting to the next tier (heterogeneous blocks
     for detailed_3D mode)

The topmost tier replaces the TSV/bonding layer with a TIM layer to the
heat sink. 
NAOMI FIXME: this should be a BGA or something to the substrate.
The heat sink goes on the silicon side I think.

BEOL layer parameters default to ASAP7 7nm PDK values (9 metal levels in
4 groups: M1-M3 @36nm pitch, M4-M5 @48nm, M6-M7 @64nm, M8-M9 @80nm).
Each metal level becomes a separate HotSpot thermal layer with individually
computed effective thermal conductivity via the Rule of Mixtures.

Usage:
    python generate_3d_stack.py -n 4                     # 4-tier stack
    python generate_3d_stack.py -n 2 -o my_stack         # custom output dir
    python generate_3d_stack.py -n 3 --power-per-layer 5 # 5W per tier
    python generate_3d_stack.py -n 2 --beol-preset asap7 # explicit PDK preset

TODO:
- Joule Heating?
- Thermal coupling layer? (https://www.imec-int.com/en/articles/mitigating-thermal-bottleneck-advanced-interconnects)
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field


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

    # -- BEOL metal/dielectric layers ----------------------------------------
    # Each metal level is modeled as a separate thermal layer with its own
    # EMA-computed effective conductivity.  The ``beol_layers`` list defines
    # the stack from M1 (closest to silicon) upward.
    #
    # Geometry defaults from ASAP7 7nm PDK (9 metal levels, 4 pitch groups).
    # Source: asap7_pdk_r1p7 config files (m1-m3.json, m4-m5.json, etc.)
    #   - Fill fraction = width / pitch = 0.50 (at minimum pitch)
    #   - Layer thickness = metal height = AR * width  (AR = 2 for ASAP7)
    #
    # Each entry: (name, pitch_nm, width_nm, height_nm, fill_fraction)
    # The ILD thickness between metal levels equals the metal height.
    beol_layers: list = field(default_factory=lambda: [
        # ASAP7: M1-M3 group (36 nm pitch, EUV patterned)
        ("M1", 36, 18, 36, 0.50),
        ("M2", 36, 18, 36, 0.50),
        ("M3", 36, 18, 36, 0.50),
        # ASAP7: M4-M5 group (48 nm pitch, SADP patterned)
        ("M4", 48, 24, 48, 0.50),
        ("M5", 48, 24, 48, 0.50),
        # ASAP7: M6-M7 group (64 nm pitch, SADP patterned)
        ("M6", 64, 32, 64, 0.50),
        ("M7", 64, 32, 64, 0.50),
        # ASAP7: M8-M9 group (80 nm pitch, top-level routing)
        ("M8", 80, 40, 80, 0.50),
        ("M9", 80, 40, 80, 0.50),
    ])
    #
    # THERMAL CONDUCTIVITY VALUES
    # NOTE: The ASAP7 PDK does not provide thermal conductivity — only
    # electrical parameters (rho, e_r).  The defaults below are *derived
    # estimates*, not measured/PDK values.  Override with --metal-conductivity
    # and --dielectric-conductivity if you have better numbers.
    #
    # TODO: Find published measurement data for Cu thermal conductivity at
    #       nanoscale wire dimensions and for the specific low-k ILD
    #       formulation assumed by ASAP7.  Candidates:
    #       - Im et al., IEEE TED 2005 (k vs e_r correlation)
    #       - Kuo et al., "Thermal conductivity of ultra low-k dielectrics"
    #         Microelectronic Engineering 2003
    #       - imec BTE-FEM studies (calibrated to self-heating measurements)
    #
    # Metal thermal conductivity [W/(m-K)].
    # Derived via Wiedemann-Franz from ASAP7 effective rho = 5 uOhm-cm:
    #   k_eff = k_bulk_Cu * (rho_bulk / rho_eff) = 401 * (1.7/5) ~ 136
    # Bulk Cu = 401 W/(m-K) at rho = 1.7 uOhm-cm; at 7nm, grain-boundary
    # and surface scattering raise rho to ~5 uOhm-cm (per ASAP7 PDK).
    metal_conductivity: float = 136.0          # ESTIMATE, W/(m-K)
    metal_specific_heat: float = 3.42e6        # copper, J/(m^3-K)
    # Low-k dielectric thermal conductivity [W/(m-K)].
    # ASAP7 specifies e_r = 2.7 (SiCOH / organosilicate glass).
    # Thermal conductivity is NOT in the PDK.  This estimate is from the
    # literature correlation between e_r and k for SiCOH materials:
    #   e_r ~ 2.7 => k ~ 0.20-0.30 W/(m-K)  (midpoint used)
    # For reference, SiO2 (e_r ~ 4.0) has k ~ 1.4 W/(m-K) — the low-k
    # materials trade thermal conductivity for lower capacitance.
    dielectric_conductivity: float = 0.25      # ESTIMATE, W/(m-K)
    dielectric_specific_heat: float = 1.65e6   # J/(m^3-K)

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
    tsv_density: int = 900        # TSVs per chip (must be a perfect square)
    tsv_diameter: float = 10e-6   # 10 um TSV diameter
    tsv_keepout_zone: float = 5e-6  # 5 um keep-out zone radius from TSV edge
 
    # -- Top TIM layer (between topmost tier and heat sink) -----------------
    tim_thickness: float = 20e-6      # 20 um
    tim_conductivity: float = 4.0     # W/(m-K)
    tim_specific_heat: float = 4.0e6  # J/(m^3-K)

    # -- Power --------------------------------------------------------------
    power_per_layer: float = 10.0  # watts per silicon layer (uniform) NAOMI FIXME: should be W/mm^2
    num_power_samples: int = 1     # number of time-step rows in ptrace

    # -- Grid model ---------------------------------------------------------
    grid_rows: int = 16
    grid_cols: int = 16

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
# Validation
# ---------------------------------------------------------------------------

def validate_tsv_config(cfg: StackConfig) -> None:
    """Validate TSV geometry parameters.

    Checks:
    1. ``tsv_density`` is a perfect square (TSVs on a uniform grid).
    2. Adjacent TSV keep-out zones do not overlap, i.e. the center-to-center
       pitch is at least ``tsv_diameter + 2 * tsv_keepout_zone``.
    """
    # --- perfect square ---
    n = int(round(math.isqrt(cfg.tsv_density)))
    if n * n != cfg.tsv_density:
        sys.exit(f"Error: --tsv-density ({cfg.tsv_density}) must be a perfect "
                 f"square (e.g. 4, 9, 16, 25, 100, 900, ...)")

    if cfg.tsv_density == 0:
        return  # nothing else to check

    # --- KOZ overlap ---
    # Uniform grid: n TSVs across each dimension of the chip.
    pitch = cfg.chip_width / n  # assumes square chip
    min_pitch = cfg.tsv_diameter + 2.0 * cfg.tsv_keepout_zone

    if pitch < min_pitch:
        pitch_um = pitch * 1e6
        min_um = min_pitch * 1e6
        sys.exit(f"Error: TSV keep-out zones overlap. "
                 f"Pitch = {pitch_um:.1f} um (from {n}x{n} grid on "
                 f"{cfg.chip_width * 1e3:.1f} mm chip) but minimum "
                 f"non-overlapping pitch = {min_um:.1f} um "
                 f"(diameter {cfg.tsv_diameter * 1e6:.1f} + "
                 f"2 x KOZ {cfg.tsv_keepout_zone * 1e6:.1f} um). "
                 f"Reduce --tsv-density or --tsv-keepout-zone.")


# ---------------------------------------------------------------------------
# Integration density
# ---------------------------------------------------------------------------

@dataclass
class IntegrationDensity:
    """Per-tier integration density metrics."""

    tier: int
    chip_area: float           # total chip area (m^2)
    num_tsvs: int              # number of TSVs passing through this tier
    tsv_footprint: float       # total TSV cross-section area (m^2)
    keepout_area: float        # total keep-out zone area (m^2), excludes TSV itself
    excluded_area: float       # tsv_footprint + keepout_area (m^2)
    usable_area: float         # chip_area - excluded_area (m^2)
    density_pct: float         # usable_area / chip_area * 100


def compute_integration_density(cfg: StackConfig) -> list[IntegrationDensity]:
    """Compute per-tier integration density.

    TSVs are placed on a uniform sqrt(N) x sqrt(N) grid.  Each TSV
    occupies a circular cross-section of diameter ``tsv_diameter`` and
    imposes a keep-out zone of radius ``tsv_keepout_zone`` from the TSV
    edge where active devices cannot be placed.

    Tiers 0 .. N-2 each have TSVs; the top tier (N-1) has none.

    Returns a list of :class:`IntegrationDensity`, one per tier.
    """
    chip_area = cfg.chip_width * cfg.chip_height
    num_tsvs = cfg.tsv_density
    tsv_radius = cfg.tsv_diameter / 2.0
    tsv_cross_section = math.pi * tsv_radius ** 2

    # Area excluded per TSV: circle of radius (tsv_radius + koz)
    exclusion_radius = tsv_radius + cfg.tsv_keepout_zone
    exclusion_per_tsv = math.pi * exclusion_radius ** 2

    total_tsv_footprint = num_tsvs * tsv_cross_section
    total_exclusion = num_tsvs * exclusion_per_tsv
    total_keepout_only = total_exclusion - total_tsv_footprint

    results: list[IntegrationDensity] = []
    for tier in range(cfg.num_layers):
        if tier < cfg.num_layers - 1:
            usable = chip_area - total_exclusion
            results.append(IntegrationDensity(
                tier=tier,
                chip_area=chip_area,
                num_tsvs=num_tsvs,
                tsv_footprint=total_tsv_footprint,
                keepout_area=total_keepout_only,
                excluded_area=total_exclusion,
                usable_area=usable,
                density_pct=usable / chip_area * 100.0,
            ))
        else:
            # Top tier: no TSVs pass through
            results.append(IntegrationDensity(
                tier=tier,
                chip_area=chip_area,
                num_tsvs=0,
                tsv_footprint=0.0,
                keepout_area=0.0,
                excluded_area=0.0,
                usable_area=chip_area,
                density_pct=100.0,
            ))

    return results


def print_integration_density(densities: list[IntegrationDensity]) -> None:
    """Print integration density report to stdout."""
    print("  Integration density (usable silicon area):")
    for d in densities:
        if d.num_tsvs > 0:
            # Convert m^2 to mm^2 for readability (1 m^2 = 1e6 mm^2)
            excl_mm2 = d.excluded_area * 1e6
            tsv_mm2 = d.tsv_footprint * 1e6
            koz_mm2 = d.keepout_area * 1e6
            print(f"    Tier {d.tier}: {d.density_pct:6.2f}%  "
                  f"({d.num_tsvs} TSVs, "
                  f"excluded {excl_mm2:.4f} mm^2 = "
                  f"TSV {tsv_mm2:.4f} + "
                  f"KOZ {koz_mm2:.4f} mm^2)")
        else:
            print(f"    Tier {d.tier}: {d.density_pct:6.2f}%  (no TSVs)")
    # Overall average
    avg = sum(d.density_pct for d in densities) / len(densities)
    print(f"    Average:  {avg:6.2f}%")


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


def generate_beol_layer_flp(cfg: StackConfig, tier: int,
                            metal_name: str, path: str) -> list[str]:
    """Single monolithic block for one BEOL metal level; EMA properties in LCF."""
    name = f"beol_{metal_name}_{tier}"
    with open(path, "w") as fh:
        fh.write(f"# BEOL {metal_name} – tier {tier}\n")
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

    chip_area = cfg.chip_width * cfg.chip_height
    tsv_radius = cfg.tsv_diameter / 2.0
    tsv_area_fraction = cfg.tsv_density * math.pi * tsv_radius ** 2 / chip_area

    total_tsv_height = tsv_area_fraction * cfg.chip_height
    tsv_strip_height = total_tsv_height / cfg.num_tsv_strips

    total_dielectric_height = cfg.chip_height - total_tsv_height
    num_dielectric_blocks = cfg.num_tsv_strips + 1
    dielectric_block_height = total_dielectric_height / num_dielectric_blocks

    tsv_resistivity = 1.0 / cfg.tsv_conductivity

    with open(path, "w") as fh:
        fh.write(f"# TSV/bonding layer – tier {tier}\n")
        fh.write(f"# Heterogeneous: dielectric blocks + TSV strips (detailed_3D)\n")
        fh.write(f"# TSV density = {cfg.tsv_density} ({int(round(math.isqrt(cfg.tsv_density)))}x{int(round(math.isqrt(cfg.tsv_density)))} grid), area fraction = {tsv_area_fraction:.6f}\n")

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


def compute_beol_ema(cfg: StackConfig) -> list[tuple[str, float, float, float, float]]:
    """Compute per-metal-level EMA thermal properties.

    Returns a list of (name, thickness_m, k_eff, p_eff, fill) for each
    BEOL metal level defined in ``cfg.beol_layers``.
    """
    results = []
    for (name, _pitch_nm, _width_nm, height_nm, fill) in cfg.beol_layers:
        thickness = height_nm * 1e-9  # nm -> m
        k_eff, p_eff = effective_medium(
            cfg.metal_conductivity, cfg.metal_specific_heat,
            cfg.dielectric_conductivity, cfg.dielectric_specific_heat,
            fill,
        )
        results.append((name, thickness, k_eff, p_eff, fill))
    return results


def generate_lcf(cfg: StackConfig, flp_files: list[str],
                 path: str) -> None:
    """Generate the layer configuration file.

    Layer ordering (bottom to top)::

        For each tier 0 .. N-1:
            silicon  (Power = Y)
            M1 BEOL layer  (Power = N, per-level EMA)
            M2 BEOL layer  (Power = N, per-level EMA)
            ...
            M9 BEOL layer  (Power = N, per-level EMA)
            tsv/bonding  (Power = N, heterogeneous)   [omitted for top tier]
        Top tier ends with:
            TIM  (Power = N)
    """
    beol_ema = compute_beol_ema(cfg)

    with open(path, "w") as fh:
        fh.write("# Layer Configuration File – 3D Stacked Chip\n")
        fh.write("# Generated by generate_3d_stack.py\n")
        fh.write(f"# BEOL: {len(cfg.beol_layers)} metal levels per tier "
                 f"(ASAP7 7nm defaults)\n")
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

            # --- per-metal-level BEOL layers ---
            for (mname, thickness, k_eff, p_eff, fill_frac) in beol_ema:
                _lcf_layer(fh, layer_num, "Y", "N",
                           p_eff, 1.0 / k_eff,
                           thickness,
                           flp_files[flp_idx],
                           comment=(f"Tier {tier}: {mname} "
                                    f"(EMA k_eff={k_eff:.2f} W/(m-K), "
                                    f"fill={fill_frac:.2f}, "
                                    f"t={thickness*1e9:.0f} nm)"))
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
                # FIXME: this should go on the silicon side!!
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
    validate_tsv_config(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    beol_ema = compute_beol_ema(cfg)
    num_beol = len(cfg.beol_layers)

    flp_files: list[str] = []       # filenames in layer order (for LCF)
    silicon_names: list[str] = []   # block names of power-dissipating layers

    for tier in range(cfg.num_layers):
        # --- silicon ---
        fname = f"silicon_{tier}.flp"
        names = generate_silicon_flp(
            cfg, tier, os.path.join(cfg.output_dir, fname))
        flp_files.append(fname)
        silicon_names.extend(names)

        # --- per-metal-level BEOL layers ---
        for (mname, _thickness, _k_eff, _p_eff, _fill) in beol_ema:
            fname = f"beol_{mname}_{tier}.flp"
            generate_beol_layer_flp(
                cfg, tier, mname, os.path.join(cfg.output_dir, fname))
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
    # layers per tier: 1 silicon + N beol + 1 (tsv or tim)
    layers_per_tier = 1 + num_beol + 1
    total_lcf_layers = cfg.num_layers * layers_per_tier
    total_beol_thickness = sum(t for (_, t, _, _, _) in beol_ema)

    print(f"Generated {cfg.num_layers}-tier 3D stack in '{cfg.output_dir}/'")
    print(f"  LCF layers:           {total_lcf_layers}")
    print(f"  Silicon layers:       {cfg.num_layers}  (power-dissipating)")
    print(f"  BEOL layers/tier:     {num_beol}  ({num_beol * cfg.num_layers} total)")
    print(f"  TSV/bonding layers:   {cfg.num_layers - 1}")
    print(f"  TIM top layer:        1")
    print()
    print(f"  BEOL stack ({num_beol} metal levels, "
          f"total thickness = {total_beol_thickness*1e9:.0f} nm "
          f"= {total_beol_thickness*1e6:.3f} um):")
    print(f"    {'Level':<6} {'Pitch':>7} {'Width':>7} {'Height':>8} "
          f"{'Fill':>6} {'k_eff':>10} {'r_eff':>12}")
    print(f"    {'':─<6} {'(nm)':─>7} {'(nm)':─>7} {'(nm)':─>8} "
          f"{'':─>6} {'W/(m-K)':─>10} {'(m-K)/W':─>12}")
    for (mname, pitch, width, height, fill) in cfg.beol_layers:
        k_eff, _ = effective_medium(
            cfg.metal_conductivity, cfg.metal_specific_heat,
            cfg.dielectric_conductivity, cfg.dielectric_specific_heat,
            fill,
        )
        print(f"    {mname:<6} {pitch:>7} {width:>7} {height:>8} "
              f"{fill:>6.2f} {k_eff:>10.2f} {1.0/k_eff:>12.6f}")
    print(f"    Metal k = {cfg.metal_conductivity:.1f} W/(m-K) "
          f"(ESTIMATE: Wiedemann-Franz from ASAP7 rho=5 uOhm-cm)")
    print(f"    Dielectric k = {cfg.dielectric_conductivity:.2f} W/(m-K) "
          f"(ESTIMATE: literature correlation for e_r~2.7 SiCOH)")

    n_side = int(round(math.isqrt(cfg.tsv_density)))
    pitch_um = (cfg.chip_width / n_side) * 1e6 if n_side > 0 else 0.0
    print()
    print(f"  TSV geometry (diameter = {cfg.tsv_diameter * 1e6:.1f} um, "
          f"KOZ = {cfg.tsv_keepout_zone * 1e6:.1f} um):")
    print(f"    Grid:  {n_side}x{n_side} = {cfg.tsv_density} TSVs  "
          f"(pitch = {pitch_um:.1f} um)")
    print()
    densities = compute_integration_density(cfg)
    print_integration_density(densities)
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
    parser.add_argument(
        "--tsv-density", type=int, default=900,
        help="Number of TSVs per chip (must be a perfect square, e.g. 900 = 30x30 grid)",
    )
    parser.add_argument(
        "--tsv-diameter", type=float, default=10e-6,
        help="TSV diameter in meters (default: 10 um)",
    )
    parser.add_argument(
        "--tsv-keepout-zone", type=float, default=5e-6,
        help="Keep-out zone radius from TSV edge in meters (default: 5 um)",
    )
    parser.add_argument(
        "--beol-config", type=str, default=None,
        help=("Path to a JSON file defining BEOL metal layers. "
              "Format: list of [name, pitch_nm, width_nm, height_nm, fill]. "
              "Overrides the built-in ASAP7 defaults."),
    )
    parser.add_argument(
        "--metal-conductivity", type=float, default=136.0,
        help=("Metal thermal conductivity in W/(m-K). "
              "Default 136 is an ESTIMATE derived via Wiedemann-Franz "
              "from ASAP7 rho=5 uOhm-cm — not a PDK/measured value."),
    )
    parser.add_argument(
        "--dielectric-conductivity", type=float, default=0.25,
        help=("ILD thermal conductivity in W/(m-K). "
              "Default 0.25 is an ESTIMATE from literature correlation "
              "for e_r~2.7 SiCOH — not a PDK/measured value."),
    )
    parser.add_argument(
        "--grid-size", type=int, default=None,
        help="Grid resolution (sets both grid_rows and grid_cols to N)",
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
        tsv_density=args.tsv_density,
        tsv_diameter=args.tsv_diameter,
        tsv_keepout_zone=args.tsv_keepout_zone,
        metal_conductivity=args.metal_conductivity,
        dielectric_conductivity=args.dielectric_conductivity,
    )

    if args.grid_size is not None:
        cfg.grid_rows = args.grid_size
        cfg.grid_cols = args.grid_size

    # Load custom BEOL config if provided
    if args.beol_config:
        with open(args.beol_config, "r") as f:
            beol_data = json.load(f)
        cfg.beol_layers = [
            (str(entry[0]), int(entry[1]), int(entry[2]),
             int(entry[3]), float(entry[4]))
            for entry in beol_data
        ]

    generate_3d_stack(cfg)


if __name__ == "__main__":
    main()
