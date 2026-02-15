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

Optionally, dummy thermal pillars made of metal can span the entire stack
to provide additional vertical heat conduction paths.  When enabled, every
layer's floorplan becomes heterogeneous with pillar strip regions carrying
copper-like R-C overrides.

Usage:
    python generate_3d_stack.py -n 4                     # 4-tier stack
    python generate_3d_stack.py -n 2 -o my_stack         # custom output dir
    python generate_3d_stack.py -n 3 --power-per-layer 5 # 5W per tier
    python generate_3d_stack.py -n 2 --use-thermal-pillars  # with pillars
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Strip specification (for heterogeneous floorplans)
# ---------------------------------------------------------------------------

@dataclass
class StripSpec:
    """Specification for a type of strip insertion in a heterogeneous floorplan.

    Used to represent TSV strips, thermal pillar strips, or any other
    heterogeneous region that overrides the bulk layer R-C properties.
    """
    name_prefix: str      # block name prefix, e.g. "tsv_0" → "tsv_0_0", "tsv_0_1"
    num_strips: int       # number of strips of this type
    area_fraction: float  # total area fraction occupied by all strips of this type
    specific_heat: float  # J/(m^3-K) — per-block R-C override
    resistivity: float    # (m-K)/W  — per-block R-C override


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
    tsv_density: int = 900        # TSVs per chip (must be a perfect square)
    tsv_diameter: float = 10e-6   # 10 um TSV diameter
    tsv_keepout_zone: float = 5e-6  # 5 um keep-out zone radius from TSV edge

    # -- Thermal pillars (optional, span entire stack) ----------------------
    # Dummy metal pillars that provide additional vertical heat conduction.
    # When enabled, every layer gets heterogeneous pillar strip regions.
    use_thermal_pillars: bool = False
    num_thermal_pillars: int = 100          # must be a perfect square (e.g. 10x10)
    thermal_pillar_diameter: float = 50e-6  # 50 um
    thermal_pillar_keepout_zone: float = 10e-6  # 10 um from pillar edge
    num_thermal_pillar_strips: int = 2      # number of strip regions per layer
    # Material properties (copper by default — "made of metal")
    thermal_pillar_conductivity: float = 401.0   # W/(m-K)
    thermal_pillar_specific_heat: float = 3.42e6 # J/(m^3-K)

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
# Area fraction helpers
# ---------------------------------------------------------------------------

def _compute_tsv_area_fraction(cfg: StackConfig) -> float:
    """Total cross-sectional area fraction of TSVs on the chip."""
    tsv_radius = cfg.tsv_diameter / 2.0
    chip_area = cfg.chip_width * cfg.chip_height
    return cfg.tsv_density * math.pi * tsv_radius ** 2 / chip_area


def _compute_pillar_area_fraction(cfg: StackConfig) -> float:
    """Total cross-sectional area fraction of thermal pillars on the chip."""
    pillar_radius = cfg.thermal_pillar_diameter / 2.0
    chip_area = cfg.chip_width * cfg.chip_height
    return cfg.num_thermal_pillars * math.pi * pillar_radius ** 2 / chip_area


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


def validate_thermal_pillar_config(cfg: StackConfig) -> None:
    """Validate thermal pillar geometry parameters.

    Checks:
    1. ``num_thermal_pillars`` is a perfect square (uniform grid).
    2. Adjacent pillar keep-out zones do not overlap.
    3. Total strip area fractions do not exceed the chip area.
    """
    if not cfg.use_thermal_pillars or cfg.num_thermal_pillars == 0:
        return

    # --- perfect square ---
    n = int(round(math.isqrt(cfg.num_thermal_pillars)))
    if n * n != cfg.num_thermal_pillars:
        sys.exit(f"Error: --num-thermal-pillars ({cfg.num_thermal_pillars}) must "
                 f"be a perfect square (e.g. 4, 9, 16, 25, 100, ...)")

    # --- KOZ overlap ---
    pitch = cfg.chip_width / n
    min_pitch = cfg.thermal_pillar_diameter + 2.0 * cfg.thermal_pillar_keepout_zone

    if pitch < min_pitch:
        pitch_um = pitch * 1e6
        min_um = min_pitch * 1e6
        sys.exit(f"Error: Thermal pillar keep-out zones overlap. "
                 f"Pitch = {pitch_um:.1f} um (from {n}x{n} grid on "
                 f"{cfg.chip_width * 1e3:.1f} mm chip) but minimum "
                 f"non-overlapping pitch = {min_um:.1f} um "
                 f"(diameter {cfg.thermal_pillar_diameter * 1e6:.1f} + "
                 f"2 x KOZ {cfg.thermal_pillar_keepout_zone * 1e6:.1f} um). "
                 f"Reduce --num-thermal-pillars or --thermal-pillar-keepout-zone.")

    # --- total strip area fraction check ---
    pillar_frac = _compute_pillar_area_fraction(cfg)
    tsv_frac = _compute_tsv_area_fraction(cfg)
    # Worst case: TSV/bonding layer has both TSV and pillar strips
    max_frac = tsv_frac + pillar_frac
    if max_frac >= 1.0:
        sys.exit(f"Error: Combined TSV + thermal pillar area fraction "
                 f"({max_frac:.4f}) >= 1.0. No room for bulk material. "
                 f"Reduce --tsv-density or --num-thermal-pillars.")
    if max_frac > 0.5:
        print(f"Warning: Combined TSV + thermal pillar area fraction "
              f"({max_frac:.4f}) > 50%. Bulk regions will be narrow.",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Integration density
# ---------------------------------------------------------------------------

@dataclass
class IntegrationDensity:
    """Per-tier integration density metrics."""

    tier: int
    chip_area: float              # total chip area (m^2)
    num_tsvs: int                 # TSVs passing through this tier
    tsv_footprint: float          # total TSV cross-section area (m^2)
    tsv_keepout_area: float       # total TSV KOZ area (m^2), excludes TSV itself
    num_thermal_pillars: int      # thermal pillars passing through this tier
    pillar_footprint: float       # total pillar cross-section area (m^2)
    pillar_keepout_area: float    # total pillar KOZ area (m^2), excludes pillar
    excluded_area: float          # total excluded area (m^2)
    usable_area: float            # chip_area - excluded_area (m^2)
    density_pct: float            # usable_area / chip_area * 100


def compute_integration_density(cfg: StackConfig) -> list[IntegrationDensity]:
    """Compute per-tier integration density.

    TSVs are placed on a uniform sqrt(N) x sqrt(N) grid.  Each TSV
    occupies a circular cross-section of diameter ``tsv_diameter`` and
    imposes a keep-out zone of radius ``tsv_keepout_zone`` from the TSV
    edge where active devices cannot be placed.

    Thermal pillars (when enabled) similarly occupy area with their own
    keep-out zones but are present on ALL tiers, not just inter-tier layers.

    Tiers 0 .. N-2 each have TSVs; the top tier (N-1) has none.
    Thermal pillars are present on all tiers.

    Returns a list of :class:`IntegrationDensity`, one per tier.
    """
    chip_area = cfg.chip_width * cfg.chip_height

    # --- TSV exclusion ---
    num_tsvs = cfg.tsv_density
    tsv_radius = cfg.tsv_diameter / 2.0
    tsv_cross_section = math.pi * tsv_radius ** 2
    tsv_exclusion_radius = tsv_radius + cfg.tsv_keepout_zone
    tsv_exclusion_per = math.pi * tsv_exclusion_radius ** 2

    total_tsv_footprint = num_tsvs * tsv_cross_section
    total_tsv_exclusion = num_tsvs * tsv_exclusion_per
    total_tsv_koz = total_tsv_exclusion - total_tsv_footprint

    # --- Thermal pillar exclusion ---
    if cfg.use_thermal_pillars and cfg.num_thermal_pillars > 0:
        num_pillars = cfg.num_thermal_pillars
        p_radius = cfg.thermal_pillar_diameter / 2.0
        p_cross = math.pi * p_radius ** 2
        p_excl_radius = p_radius + cfg.thermal_pillar_keepout_zone
        p_excl_per = math.pi * p_excl_radius ** 2

        total_pillar_footprint = num_pillars * p_cross
        total_pillar_exclusion = num_pillars * p_excl_per
        total_pillar_koz = total_pillar_exclusion - total_pillar_footprint
    else:
        num_pillars = 0
        total_pillar_footprint = 0.0
        total_pillar_exclusion = 0.0
        total_pillar_koz = 0.0

    results: list[IntegrationDensity] = []
    for tier in range(cfg.num_layers):
        # TSVs: present in tiers 0..N-2
        if tier < cfg.num_layers - 1:
            t_tsvs = num_tsvs
            t_tsv_fp = total_tsv_footprint
            t_tsv_koz = total_tsv_koz
            t_tsv_excl = total_tsv_exclusion
        else:
            t_tsvs = 0
            t_tsv_fp = 0.0
            t_tsv_koz = 0.0
            t_tsv_excl = 0.0

        # Thermal pillars: present on ALL tiers
        excluded = t_tsv_excl + total_pillar_exclusion
        usable = chip_area - excluded

        results.append(IntegrationDensity(
            tier=tier,
            chip_area=chip_area,
            num_tsvs=t_tsvs,
            tsv_footprint=t_tsv_fp,
            tsv_keepout_area=t_tsv_koz,
            num_thermal_pillars=num_pillars,
            pillar_footprint=total_pillar_footprint,
            pillar_keepout_area=total_pillar_koz,
            excluded_area=excluded,
            usable_area=usable,
            density_pct=usable / chip_area * 100.0,
        ))

    return results


def print_integration_density(densities: list[IntegrationDensity]) -> None:
    """Print integration density report to stdout."""
    print("  Integration density (usable silicon area):")
    for d in densities:
        parts: list[str] = []
        if d.num_tsvs > 0:
            tsv_mm2 = d.tsv_footprint * 1e6
            koz_mm2 = d.tsv_keepout_area * 1e6
            parts.append(f"{d.num_tsvs} TSVs "
                         f"(footprint {tsv_mm2:.4f} + KOZ {koz_mm2:.4f} mm^2)")
        if d.num_thermal_pillars > 0:
            p_mm2 = d.pillar_footprint * 1e6
            pk_mm2 = d.pillar_keepout_area * 1e6
            parts.append(f"{d.num_thermal_pillars} pillars "
                         f"(footprint {p_mm2:.4f} + KOZ {pk_mm2:.4f} mm^2)")
        if parts:
            excl_mm2 = d.excluded_area * 1e6
            detail = ", ".join(parts)
            print(f"    Tier {d.tier}: {d.density_pct:6.2f}%  "
                  f"({detail}, total excluded {excl_mm2:.4f} mm^2)")
        else:
            print(f"    Tier {d.tier}: {d.density_pct:6.2f}%  (no exclusions)")

    avg = sum(d.density_pct for d in densities) / len(densities)
    print(f"    Average:  {avg:6.2f}%")


# ---------------------------------------------------------------------------
# Floorplan helpers
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


def _write_heterogeneous_flp(fh, chip_width: float, chip_height: float,
                             bulk_name_prefix: str,
                             strip_specs: list[StripSpec]) -> list[str]:
    """Write heterogeneous floorplan blocks with interleaved bulk and strips.

    Bulk blocks use the layer's default R-C properties (from the LCF).
    Strip blocks carry per-block R-C overrides for ``detailed_3D`` mode.

    Strips of different types are interleaved round-robin, with bulk
    blocks filling the space between them.

    Parameters
    ----------
    fh : file handle
        Already-opened file with header comments written.
    chip_width : float
        Full chip width (m).
    chip_height : float
        Full chip height (m).
    bulk_name_prefix : str
        Name prefix for bulk blocks, e.g. "bond_d_0" → "bond_d_0_0", ...
    strip_specs : list[StripSpec]
        Specifications for each strip type.

    Returns
    -------
    list[str]
        All block names in bottom-to-top order.
    """
    # Build the ordered list of strips via round-robin across types.
    strips_by_type: list[list[tuple[str, float, float, float]]] = []
    for spec in strip_specs:
        if spec.num_strips <= 0 or spec.area_fraction <= 0:
            continue
        total_height = spec.area_fraction * chip_height
        per_strip = total_height / spec.num_strips
        type_strips = []
        for i in range(spec.num_strips):
            type_strips.append((
                f"{spec.name_prefix}_{i}",
                per_strip,
                spec.specific_heat,
                spec.resistivity,
            ))
        strips_by_type.append(type_strips)

    # Round-robin interleave
    ordered_strips: list[tuple[str, float, float, float]] = []
    max_len = max((len(s) for s in strips_by_type), default=0)
    for i in range(max_len):
        for type_strips in strips_by_type:
            if i < len(type_strips):
                ordered_strips.append(type_strips[i])

    # Compute bulk block dimensions
    total_strip_height = sum(s[1] for s in ordered_strips)
    total_bulk_height = chip_height - total_strip_height
    num_bulk_blocks = len(ordered_strips) + 1
    bulk_height = total_bulk_height / num_bulk_blocks

    # Write blocks: bulk_0, strip_0, bulk_1, strip_1, ..., bulk_N
    names: list[str] = []
    y = 0.0
    bulk_idx = 0

    for strip_name, strip_h, sh, res in ordered_strips:
        # Bulk block
        name = f"{bulk_name_prefix}_{bulk_idx}"
        fh.write(_flp_line(name, chip_width, bulk_height, 0.0, y))
        names.append(name)
        y += bulk_height
        bulk_idx += 1

        # Strip block (with R-C overrides)
        fh.write(_flp_line(strip_name, chip_width, strip_h, 0.0, y,
                           specific_heat=sh, resistivity=res))
        names.append(strip_name)
        y += strip_h

    # Final bulk block (absorbs any FP rounding)
    remaining = chip_height - y
    name = f"{bulk_name_prefix}_{bulk_idx}"
    fh.write(_flp_line(name, chip_width, remaining, 0.0, y))
    names.append(name)

    return names


def _make_pillar_strip_spec(cfg: StackConfig,
                            name_prefix: str) -> StripSpec:
    """Create a StripSpec for thermal pillar strips."""
    return StripSpec(
        name_prefix=name_prefix,
        num_strips=cfg.num_thermal_pillar_strips,
        area_fraction=_compute_pillar_area_fraction(cfg),
        specific_heat=cfg.thermal_pillar_specific_heat,
        resistivity=1.0 / cfg.thermal_pillar_conductivity,
    )


# ---------------------------------------------------------------------------
# Floorplan generators
# ---------------------------------------------------------------------------

def generate_silicon_flp(cfg: StackConfig, tier: int,
                         path: str) -> list[str]:
    """Generate silicon die floorplan.

    Without thermal pillars: single monolithic block.
    With thermal pillars: heterogeneous with pillar strip R-C overrides.
    """
    if cfg.use_thermal_pillars:
        strips = [_make_pillar_strip_spec(cfg, f"tp_si_{tier}")]
        with open(path, "w") as fh:
            fh.write(f"# Silicon die – tier {tier}\n")
            fh.write(f"# Heterogeneous: silicon bulk + thermal pillar strips\n")
            return _write_heterogeneous_flp(fh, cfg.chip_width, cfg.chip_height,
                                            f"si_{tier}", strips)
    else:
        name = f"silicon_{tier}"
        with open(path, "w") as fh:
            fh.write(f"# Silicon die – tier {tier}\n")
            fh.write(_flp_line(name, cfg.chip_width, cfg.chip_height, 0.0, 0.0))
        return [name]


def generate_metal_dielectric_flp(cfg: StackConfig, tier: int,
                                  path: str) -> list[str]:
    """Generate metal/dielectric (BEOL) floorplan.

    Without thermal pillars: single monolithic block with EMA properties
    set at the layer level in the LCF.
    With thermal pillars: heterogeneous with pillar strip R-C overrides.
    """
    if cfg.use_thermal_pillars:
        strips = [_make_pillar_strip_spec(cfg, f"tp_md_{tier}")]
        with open(path, "w") as fh:
            fh.write(f"# Metal/dielectric (BEOL) – tier {tier}\n")
            fh.write(f"# Effective medium: metal fill fraction = "
                     f"{cfg.metal_fill_fraction}\n")
            fh.write(f"# Heterogeneous: EMA bulk + thermal pillar strips\n")
            return _write_heterogeneous_flp(fh, cfg.chip_width, cfg.chip_height,
                                            f"md_{tier}", strips)
    else:
        name = f"metal_dielectric_{tier}"
        with open(path, "w") as fh:
            fh.write(f"# Metal/dielectric (BEOL) – tier {tier}\n")
            fh.write(f"# Effective medium: metal fill fraction = "
                     f"{cfg.metal_fill_fraction}\n")
            fh.write(_flp_line(name, cfg.chip_width, cfg.chip_height, 0.0, 0.0))
        return [name]


def generate_tsv_bonding_flp(cfg: StackConfig, tier: int,
                             path: str) -> list[str]:
    """Generate TSV/bonding layer floorplan (heterogeneous).

    Always heterogeneous with TSV strips.  When thermal pillars are enabled,
    pillar strips are added alongside TSV strips.
    """
    tsv_area_fraction = _compute_tsv_area_fraction(cfg)
    tsv_resistivity = 1.0 / cfg.tsv_conductivity

    n_side = int(round(math.isqrt(cfg.tsv_density)))

    strips = [
        StripSpec(
            name_prefix=f"tsv_{tier}",
            num_strips=cfg.num_tsv_strips,
            area_fraction=tsv_area_fraction,
            specific_heat=cfg.tsv_specific_heat,
            resistivity=tsv_resistivity,
        ),
    ]

    if cfg.use_thermal_pillars:
        strips.append(_make_pillar_strip_spec(cfg, f"tp_bond_{tier}"))

    with open(path, "w") as fh:
        fh.write(f"# TSV/bonding layer – tier {tier}\n")
        fh.write(f"# Heterogeneous: dielectric bulk + TSV strips")
        if cfg.use_thermal_pillars:
            fh.write(" + thermal pillar strips")
        fh.write("\n")
        fh.write(f"# TSV density = {cfg.tsv_density} "
                 f"({n_side}x{n_side} grid), "
                 f"area fraction = {tsv_area_fraction:.6f}\n")
        return _write_heterogeneous_flp(fh, cfg.chip_width, cfg.chip_height,
                                        f"bond_d_{tier}", strips)


def generate_tim_top_flp(cfg: StackConfig, path: str) -> list[str]:
    """Generate top TIM layer floorplan.

    Without thermal pillars: single monolithic block.
    With thermal pillars: heterogeneous with pillar strip R-C overrides.
    """
    if cfg.use_thermal_pillars:
        strips = [_make_pillar_strip_spec(cfg, "tp_tim")]
        with open(path, "w") as fh:
            fh.write("# Top TIM layer (chip to heat sink)\n")
            fh.write("# Heterogeneous: TIM bulk + thermal pillar strips\n")
            return _write_heterogeneous_flp(fh, cfg.chip_width, cfg.chip_height,
                                            "tim", strips)
    else:
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

def generate_ptrace(cfg: StackConfig,
                    silicon_blocks: list[tuple[str, float]],
                    path: str) -> None:
    """Generate a power trace file.

    Parameters
    ----------
    cfg : StackConfig
    silicon_blocks : list of (name, power) tuples
        Every block in every power-dissipating (silicon) layer.
        Thermal pillar strips get 0.0 power; bulk silicon blocks share
        the tier's ``power_per_layer`` proportionally.
    path : str
        Output file path.
    """
    names = [b[0] for b in silicon_blocks]
    powers = [b[1] for b in silicon_blocks]

    with open(path, "w") as fh:
        fh.write("\t".join(names) + "\n")
        row = "\t".join(f"{p}" for p in powers)
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
    validate_thermal_pillar_config(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    flp_files: list[str] = []     # filenames in layer order (for LCF)
    silicon_blocks: list[tuple[str, float]] = []  # (name, power) for ptrace

    # Thermal pillar strip prefix for identifying non-power blocks
    tp_si_prefix = "tp_si_"

    for tier in range(cfg.num_layers):
        # --- silicon ---
        fname = f"silicon_{tier}.flp"
        names = generate_silicon_flp(
            cfg, tier, os.path.join(cfg.output_dir, fname))
        flp_files.append(fname)

        # Distribute power: bulk silicon blocks share power_per_layer;
        # thermal pillar strips get 0.
        if cfg.use_thermal_pillars:
            bulk_names = [n for n in names if not n.startswith(tp_si_prefix)]
            power_per_bulk = cfg.power_per_layer / len(bulk_names)
            for n in names:
                if n.startswith(tp_si_prefix):
                    silicon_blocks.append((n, 0.0))
                else:
                    silicon_blocks.append((n, power_per_bulk))
        else:
            for n in names:
                silicon_blocks.append((n, cfg.power_per_layer))

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
    generate_ptrace(cfg, silicon_blocks,
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
    n_side = int(round(math.isqrt(cfg.tsv_density)))
    pitch_um = (cfg.chip_width / n_side) * 1e6 if n_side > 0 else 0.0
    print()
    print(f"  TSV geometry (diameter = {cfg.tsv_diameter * 1e6:.1f} um, "
          f"KOZ = {cfg.tsv_keepout_zone * 1e6:.1f} um):")
    print(f"    Grid:  {n_side}x{n_side} = {cfg.tsv_density} TSVs  "
          f"(pitch = {pitch_um:.1f} um)")

    # Thermal pillar summary
    print()
    if cfg.use_thermal_pillars:
        p_side = int(round(math.isqrt(cfg.num_thermal_pillars)))
        p_pitch_um = (cfg.chip_width / p_side) * 1e6 if p_side > 0 else 0.0
        print(f"  Thermal pillars (diameter = "
              f"{cfg.thermal_pillar_diameter * 1e6:.1f} um, "
              f"KOZ = {cfg.thermal_pillar_keepout_zone * 1e6:.1f} um):")
        print(f"    Grid:  {p_side}x{p_side} = {cfg.num_thermal_pillars} "
              f"pillars  (pitch = {p_pitch_um:.1f} um)")
        print(f"    Material: copper "
              f"(k = {cfg.thermal_pillar_conductivity:.1f} W/(m-K))")
        print(f"    Spans all {total_lcf_layers} layers "
              f"({cfg.num_thermal_pillar_strips} strips/layer)")
    else:
        print("  Thermal pillars: disabled")

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
        help="Number of TSVs per chip (must be a perfect square, "
             "e.g. 900 = 30x30 grid)",
    )
    parser.add_argument(
        "--tsv-diameter", type=float, default=10e-6,
        help="TSV diameter in meters (default: 10 um)",
    )
    parser.add_argument(
        "--tsv-keepout-zone", type=float, default=5e-6,
        help="Keep-out zone radius from TSV edge in meters (default: 5 um)",
    )

    # Thermal pillar arguments
    parser.add_argument(
        "--use-thermal-pillars", action="store_true", default=False,
        help="Enable dummy thermal pillars spanning the entire stack",
    )
    parser.add_argument(
        "--num-thermal-pillars", type=int, default=100,
        help="Number of thermal pillars per chip (must be a perfect square, "
             "e.g. 100 = 10x10 grid)",
    )
    parser.add_argument(
        "--thermal-pillar-diameter", type=float, default=50e-6,
        help="Thermal pillar diameter in meters (default: 50 um)",
    )
    parser.add_argument(
        "--thermal-pillar-keepout-zone", type=float, default=10e-6,
        help="Keep-out zone radius from pillar edge in meters (default: 10 um)",
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
        use_thermal_pillars=args.use_thermal_pillars,
        num_thermal_pillars=args.num_thermal_pillars,
        thermal_pillar_diameter=args.thermal_pillar_diameter,
        thermal_pillar_keepout_zone=args.thermal_pillar_keepout_zone,
    )

    generate_3d_stack(cfg)


if __name__ == "__main__":
    main()
