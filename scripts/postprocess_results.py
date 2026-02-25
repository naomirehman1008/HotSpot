#!/usr/bin/env python3
"""
HotSpot 3D Thermal Simulation Post-Processor

Reads HotSpot output files from a simulation directory and produces:
  1. YAML file -- structured export of all results + metadata
  2. CSV file  -- per-tier summary table
  3. Terminal  -- formatted summary table

Usage:
    python postprocess_results.py generated_3d_stack/
    python postprocess_results.py generated_3d_stack/ -o results/
    python postprocess_results.py generated_3d_stack/ --no-grid
"""

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KELVIN_TO_CELSIUS = 273.15

# LCF comment patterns -- derived from generate_3d_stack.py lines 497-536
RE_SILICON = re.compile(r"Tier (\d+): Silicon die")
RE_BEOL = re.compile(
    r"Tier (\d+): (M\d+) \(EMA k_eff=([\d.]+) W/\(m-K\), "
    r"fill=([\d.]+), t=(\d+) nm\)"
)
RE_METAL_DIEL = re.compile(
    r"Tier (\d+): Metal/dielectric \(EMA k_eff=([\d.]+) W/\(m-K\), "
    r"fill=([\d.]+)\)"
)
RE_TSV = re.compile(r"Tier (\d+): TSV/bonding")
RE_TIM = re.compile(r"Top TIM")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Simulation metadata parsed from hotspot.config."""
    ambient_K: float
    grid_rows: int
    grid_cols: int
    sampling_interval_s: float
    t_chip: float
    k_chip: float
    init_temp_K: float


@dataclass
class LayerInfo:
    """Metadata for one LCF layer."""
    layer_num: int
    tier: int                          # -1 for TIM / package
    layer_type: str                    # silicon, beol, metal_dielectric, tsv_bonding, tim
    metal_name: Optional[str]          # e.g. "M1" -- only for beol type
    lateral_flow: bool
    power: bool
    specific_heat: float               # J/(m^3-K)
    resistivity: float                 # (m-K)/W
    conductivity: float                # W/(m-K), = 1/resistivity
    thickness_m: float
    floorplan_file: str
    ema_k_eff: Optional[float] = None  # parsed from comment
    ema_fill: Optional[float] = None   # parsed from comment


@dataclass
class GridLayerStats:
    """Statistics for one grid layer."""
    layer_num: int
    mean_K: float
    max_K: float
    min_K: float
    std_K: float


@dataclass
class BlockTemp:
    """One block from steady.out."""
    full_name: str
    layer_num: int
    block_name: str
    temp_K: float


@dataclass
class PackageNode:
    """Package-level node (hsp_, hsink_, inode_) from steady.out."""
    name: str
    temp_K: float


@dataclass
class TierSummary:
    """Aggregated per-tier summary."""
    tier: int
    silicon_mean_K: float
    silicon_max_K: float
    silicon_min_K: float
    silicon_delta_K: float   # max - ambient
    beol_mean_K: float
    beol_max_K: float
    tsv_bonding_mean_K: Optional[float]
    tsv_bonding_max_K: Optional[float]
    power_W: float


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_config(config_path: str) -> SimConfig:
    """Parse hotspot.config to extract simulation parameters.

    Format: tab-indented '-key  value' pairs; '#' lines are comments.
    """
    kv: dict[str, str] = {}
    with open(config_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("-"):
                kv[parts[0]] = parts[1]

    return SimConfig(
        ambient_K=float(kv.get("-ambient", "300")),
        grid_rows=int(kv.get("-grid_rows", "64")),
        grid_cols=int(kv.get("-grid_cols", "64")),
        sampling_interval_s=float(kv.get("-sampling_intvl", "1e-6")),
        t_chip=float(kv.get("-t_chip", "5e-5")),
        k_chip=float(kv.get("-k_chip", "130")),
        init_temp_K=float(kv.get("-init_temp", "300")),
    )


def parse_lcf(lcf_path: str) -> list[LayerInfo]:
    """Parse stack.lcf to extract per-layer metadata.

    Each layer is a block of lines: a comment describing the layer type,
    then 6 data lines (layer_num, lateral, power, specific_heat, resistivity,
    thickness, floorplan_file), separated by blank lines.

    The layer type and tier are determined from the comment via regex patterns
    derived from generate_3d_stack.py.
    """
    layers: list[LayerInfo] = []

    with open(lcf_path) as fh:
        lines = fh.readlines()

    # Collect non-blank lines, preserving order
    stripped: list[str] = []
    for line in lines:
        s = line.strip()
        if s:
            stripped.append(s)

    # Parse layer blocks: each starts with a comment matching a known pattern.
    # Skip any other comment lines (file header, format description, etc.).
    idx = 0
    while idx < len(stripped):
        line = stripped[idx]

        # Only process comment lines
        if not line.startswith("#"):
            idx += 1
            continue

        comment = line[2:].strip()  # strip "# "

        # Try to classify the layer from the comment
        tier = -1
        layer_type = "unknown"
        metal_name = None
        ema_k_eff = None
        ema_fill = None

        m = RE_SILICON.search(comment)
        if m:
            tier = int(m.group(1))
            layer_type = "silicon"
        else:
            m = RE_BEOL.search(comment)
            if m:
                tier = int(m.group(1))
                layer_type = "beol"
                metal_name = m.group(2)
                ema_k_eff = float(m.group(3))
                ema_fill = float(m.group(4))
            else:
                m = RE_METAL_DIEL.search(comment)
                if m:
                    tier = int(m.group(1))
                    layer_type = "metal_dielectric"
                    ema_k_eff = float(m.group(2))
                    ema_fill = float(m.group(3))
                else:
                    m = RE_TSV.search(comment)
                    if m:
                        tier = int(m.group(1))
                        layer_type = "tsv_bonding"
                    else:
                        m = RE_TIM.search(comment)
                        if m:
                            tier = -1
                            layer_type = "tim"

        # If the comment didn't match any known layer pattern, skip it
        if layer_type == "unknown":
            idx += 1
            continue

        # Read the 6 data lines following the layer comment
        idx += 1
        if idx + 6 > len(stripped):
            break

        layer_num = int(stripped[idx]); idx += 1
        lateral_flow = stripped[idx].upper() == "Y"; idx += 1
        power = stripped[idx].upper() == "Y"; idx += 1
        specific_heat = float(stripped[idx]); idx += 1
        resistivity = float(stripped[idx]); idx += 1
        thickness_m = float(stripped[idx]); idx += 1
        floorplan_file = stripped[idx]; idx += 1

        conductivity = 1.0 / resistivity if resistivity != 0 else 0.0

        layers.append(LayerInfo(
            layer_num=layer_num,
            tier=tier,
            layer_type=layer_type,
            metal_name=metal_name,
            lateral_flow=lateral_flow,
            power=power,
            specific_heat=specific_heat,
            resistivity=resistivity,
            conductivity=conductivity,
            thickness_m=thickness_m,
            floorplan_file=floorplan_file,
            ema_k_eff=ema_k_eff,
            ema_fill=ema_fill,
        ))

    return layers


def parse_ptrace(ptrace_path: str) -> dict[str, float]:
    """Parse power.ptrace to get power per silicon block.

    Returns dict mapping block_name -> power_W (from the first data row).
    """
    with open(ptrace_path) as fh:
        header = fh.readline().strip().split("\t")
        values = fh.readline().strip().split("\t")
    return {name: float(val) for name, val in zip(header, values)}


def parse_steady(steady_path: str) -> tuple[list[BlockTemp], list[PackageNode]]:
    """Parse steady.out to extract block-level steady-state temperatures.

    Names starting with 'layer_' are chip blocks; others are package nodes.
    """
    blocks: list[BlockTemp] = []
    package: list[PackageNode] = []

    with open(steady_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            name = parts[0]
            temp = float(parts[1])

            if name.startswith("layer_"):
                # Extract layer_num and block_name from "layer_N_blockname"
                # layer_num is the first integer after "layer_"
                rest = name[len("layer_"):]
                num_str, _, block_name = rest.partition("_")
                blocks.append(BlockTemp(
                    full_name=name,
                    layer_num=int(num_str),
                    block_name=block_name,
                    temp_K=temp,
                ))
            else:
                package.append(PackageNode(name=name, temp_K=temp))

    return blocks, package


def parse_grid_steady(grid_steady_path: str, grid_rows: int,
                      grid_cols: int) -> list[GridLayerStats]:
    """Parse grid_steady.out to extract per-layer grid temperature statistics.

    Format: 'Layer N:' headers followed by rows*cols lines of 'index\ttemp'.
    HotSpot appends 2 extra package grid layers beyond the LCF-defined layers.
    """
    stats: list[GridLayerStats] = []
    n_cells = grid_rows * grid_cols

    with open(grid_steady_path) as fh:
        layer_num = -1
        temps: list[float] = []

        for line in fh:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Layer "):
                # Flush previous layer
                if temps:
                    arr = np.array(temps)
                    stats.append(GridLayerStats(
                        layer_num=layer_num,
                        mean_K=float(np.mean(arr)),
                        max_K=float(np.max(arr)),
                        min_K=float(np.min(arr)),
                        std_K=float(np.std(arr)),
                    ))
                # Start new layer
                layer_num = int(line.split()[1].rstrip(":"))
                temps = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    temps.append(float(parts[1]))

        # Flush last layer
        if temps:
            arr = np.array(temps)
            stats.append(GridLayerStats(
                layer_num=layer_num,
                mean_K=float(np.mean(arr)),
                max_K=float(np.max(arr)),
                min_K=float(np.min(arr)),
                std_K=float(np.std(arr)),
            ))

    return stats


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_tiers(layers: list[LayerInfo],
                    grid_stats: list[GridLayerStats],
                    power_map: dict[str, float],
                    ambient_K: float) -> list[TierSummary]:
    """Aggregate per-layer data into per-tier summaries.

    Groups layers by tier, computes silicon/BEOL/TSV statistics using
    the grid-level data where available.
    """
    # Build lookup from layer_num -> grid stats
    grid_by_layer = {gs.layer_num: gs for gs in grid_stats}

    # Group layers by tier
    tier_nums = sorted(set(l.tier for l in layers if l.tier >= 0))

    summaries: list[TierSummary] = []
    for t in tier_nums:
        tier_layers = [l for l in layers if l.tier == t]

        # Silicon layer for this tier
        si_layers = [l for l in tier_layers if l.layer_type == "silicon"]
        si_gs = None
        if si_layers and si_layers[0].layer_num in grid_by_layer:
            si_gs = grid_by_layer[si_layers[0].layer_num]

        # BEOL layers (both old "metal_dielectric" and new "beol")
        beol_layers = [l for l in tier_layers
                       if l.layer_type in ("beol", "metal_dielectric")]
        beol_means: list[float] = []
        beol_maxes: list[float] = []
        for bl in beol_layers:
            if bl.layer_num in grid_by_layer:
                gs = grid_by_layer[bl.layer_num]
                beol_means.append(gs.mean_K)
                beol_maxes.append(gs.max_K)

        # TSV / bonding layer
        tsv_layers = [l for l in tier_layers if l.layer_type == "tsv_bonding"]
        tsv_mean = None
        tsv_max = None
        if tsv_layers and tsv_layers[0].layer_num in grid_by_layer:
            tsv_gs = grid_by_layer[tsv_layers[0].layer_num]
            tsv_mean = tsv_gs.mean_K
            tsv_max = tsv_gs.max_K

        # Power for this tier's silicon block
        power = 0.0
        if si_layers:
            # Block name in ptrace is the floorplan block name, e.g. "silicon_0"
            flp_name = os.path.splitext(si_layers[0].floorplan_file)[0]
            power = power_map.get(flp_name, 0.0)

        summaries.append(TierSummary(
            tier=t,
            silicon_mean_K=si_gs.mean_K if si_gs else 0.0,
            silicon_max_K=si_gs.max_K if si_gs else 0.0,
            silicon_min_K=si_gs.min_K if si_gs else 0.0,
            silicon_delta_K=(si_gs.max_K - ambient_K) if si_gs else 0.0,
            beol_mean_K=(float(np.mean(beol_means)) if beol_means else 0.0),
            beol_max_K=(float(np.max(beol_maxes)) if beol_maxes else 0.0),
            tsv_bonding_mean_K=tsv_mean,
            tsv_bonding_max_K=tsv_max,
            power_W=power,
        ))

    return summaries


# ---------------------------------------------------------------------------
# Output: YAML
# ---------------------------------------------------------------------------

def _k_to_c(k: float) -> float:
    return round(k - KELVIN_TO_CELSIUS, 2)


def build_yaml_dict(config: SimConfig,
                    layers: list[LayerInfo],
                    grid_stats: list[GridLayerStats],
                    block_temps: list[BlockTemp],
                    package_nodes: list[PackageNode],
                    tier_summaries: list[TierSummary],
                    power_map: dict[str, float]) -> dict:
    """Build the nested dict structure for YAML output."""
    grid_by_layer = {gs.layer_num: gs for gs in grid_stats}
    num_tiers = max((l.tier for l in layers if l.tier >= 0), default=0) + 1
    total_power = sum(power_map.values())

    result: dict = {}

    # Simulation metadata
    result["simulation"] = {
        "ambient_K": config.ambient_K,
        "ambient_C": _k_to_c(config.ambient_K),
        "grid_rows": config.grid_rows,
        "grid_cols": config.grid_cols,
        "sampling_interval_s": config.sampling_interval_s,
        "num_tiers": num_tiers,
        "total_power_W": total_power,
    }

    # Per-layer detail
    layer_list = []
    for l in layers:
        entry: dict = {
            "layer_num": l.layer_num,
            "type": l.layer_type,
            "tier": l.tier,
            "thickness_m": l.thickness_m,
            "conductivity_W_per_mK": round(l.conductivity, 2),
            "floorplan": l.floorplan_file,
        }
        if l.metal_name:
            entry["metal_level"] = l.metal_name
        if l.ema_k_eff is not None:
            entry["ema_k_eff"] = l.ema_k_eff
        if l.ema_fill is not None:
            entry["ema_fill"] = l.ema_fill

        # Grid stats if available
        gs = grid_by_layer.get(l.layer_num)
        if gs:
            entry["grid_stats"] = {
                "mean_K": round(gs.mean_K, 2),
                "max_K": round(gs.max_K, 2),
                "min_K": round(gs.min_K, 2),
                "std_K": round(gs.std_K, 4),
            }
        layer_list.append(entry)
    result["layers"] = layer_list

    # Per-tier summary
    tier_list = []
    for ts in tier_summaries:
        entry = {
            "tier": ts.tier,
            "power_W": ts.power_W,
            "silicon": {
                "mean_K": round(ts.silicon_mean_K, 2),
                "mean_C": _k_to_c(ts.silicon_mean_K),
                "max_K": round(ts.silicon_max_K, 2),
                "max_C": _k_to_c(ts.silicon_max_K),
                "min_K": round(ts.silicon_min_K, 2),
                "min_C": _k_to_c(ts.silicon_min_K),
                "delta_from_ambient_K": round(ts.silicon_delta_K, 2),
            },
            "beol": {
                "mean_K": round(ts.beol_mean_K, 2),
                "max_K": round(ts.beol_max_K, 2),
            },
        }
        if ts.tsv_bonding_mean_K is not None:
            entry["tsv_bonding"] = {
                "mean_K": round(ts.tsv_bonding_mean_K, 2),
                "max_K": round(ts.tsv_bonding_max_K, 2),
            }
        tier_list.append(entry)
    result["tiers"] = tier_list

    # Block-level temperatures
    result["block_temps"] = {bt.full_name: round(bt.temp_K, 2)
                             for bt in block_temps}

    # Package nodes
    result["package"] = [{"name": pn.name, "temp_K": round(pn.temp_K, 2)}
                         for pn in package_nodes]

    return result


def _simple_yaml_dump(data, indent: int = 0) -> str:
    """Minimal YAML serializer for nested dicts/lists of scalars.

    Fallback when PyYAML is not installed.
    """
    pad = "  " * indent
    lines: list[str] = []

    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_simple_yaml_dump(val, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_format_scalar(val)}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                first = True
                for key, val in item.items():
                    prefix = f"{pad}- " if first else f"{pad}  "
                    first = False
                    if isinstance(val, (dict, list)):
                        lines.append(f"{prefix}{key}:")
                        lines.append(_simple_yaml_dump(val, indent + 2))
                    else:
                        lines.append(f"{prefix}{key}: {_format_scalar(val)}")
            else:
                lines.append(f"{pad}- {_format_scalar(item)}")
    else:
        lines.append(f"{pad}{_format_scalar(data)}")

    return "\n".join(lines)


def _format_scalar(val) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        if abs(val) < 1e-4 and val != 0.0:
            return f"{val:.6e}"
        return f"{val}"
    return str(val)


def write_yaml(data: dict, output_path: str) -> None:
    """Write dict to YAML file. Uses PyYAML if available, else fallback."""
    with open(output_path, "w") as fh:
        if HAS_YAML:
            # Custom representer to avoid float precision artifacts and
            # the !!float tag that PyYAML emits for round numbers
            def float_representer(dumper, value):
                if abs(value) < 1e-4 and value != 0.0:
                    s = f"{value:.6e}"
                elif value == int(value) and abs(value) < 1e15:
                    s = f"{value:.1f}"
                else:
                    s = f"{value:g}"
                return dumper.represent_scalar("tag:yaml.org,2002:float", s)

            class CleanDumper(yaml.Dumper):
                pass
            CleanDumper.add_representer(float, float_representer)
            yaml.dump(data, fh, Dumper=CleanDumper, default_flow_style=False,
                      sort_keys=False)
        else:
            print("Warning: PyYAML not installed, using fallback writer",
                  file=sys.stderr)
            fh.write(_simple_yaml_dump(data))
            fh.write("\n")


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def write_csv(tier_summaries: list[TierSummary], config: SimConfig,
              output_path: str) -> None:
    """Write per-tier summary table as CSV."""
    header = [
        "Tier",
        "Si_Mean_K", "Si_Mean_C",
        "Si_Max_K", "Si_Max_C",
        "Si_Min_K", "Si_Min_C",
        "Delta_from_Ambient_K",
        "BEOL_Mean_K", "BEOL_Max_K",
        "Power_W",
    ]

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for ts in tier_summaries:
            writer.writerow([
                ts.tier,
                f"{ts.silicon_mean_K:.2f}", f"{_k_to_c(ts.silicon_mean_K):.2f}",
                f"{ts.silicon_max_K:.2f}", f"{_k_to_c(ts.silicon_max_K):.2f}",
                f"{ts.silicon_min_K:.2f}", f"{_k_to_c(ts.silicon_min_K):.2f}",
                f"{ts.silicon_delta_K:.2f}",
                f"{ts.beol_mean_K:.2f}", f"{ts.beol_max_K:.2f}",
                f"{ts.power_W:.2f}",
            ])


# ---------------------------------------------------------------------------
# Output: Terminal
# ---------------------------------------------------------------------------

def print_summary_table(tier_summaries: list[TierSummary],
                        config: SimConfig,
                        power_map: dict[str, float],
                        package_nodes: list[PackageNode]) -> None:
    """Print formatted summary table to terminal."""
    total_power = sum(power_map.values())

    print()
    print("=" * 72)
    print("  HotSpot 3D Thermal Simulation Results")
    print("=" * 72)
    print()
    print(f"  Ambient: {config.ambient_K:.2f} K ({_k_to_c(config.ambient_K):.2f} C)"
          f"    Grid: {config.grid_rows}x{config.grid_cols}"
          f"    Total Power: {total_power:.1f} W")
    print()

    # Header
    hdr = (f"  {'Tier':>4s}"
           f"  {'Si Mean(K)':>10s}"
           f"  {'Si Mean(C)':>10s}"
           f"  {'Si Max(K)':>10s}"
           f"  {'Si Max(C)':>10s}"
           f"  {'Si Min(K)':>10s}"
           f"  {'Delta(K)':>9s}"
           f"  {'BEOL Mean(K)':>12s}"
           f"  {'Power(W)':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for ts in tier_summaries:
        print(f"  {ts.tier:>4d}"
              f"  {ts.silicon_mean_K:>10.2f}"
              f"  {_k_to_c(ts.silicon_mean_K):>10.2f}"
              f"  {ts.silicon_max_K:>10.2f}"
              f"  {_k_to_c(ts.silicon_max_K):>10.2f}"
              f"  {ts.silicon_min_K:>10.2f}"
              f"  {ts.silicon_delta_K:>9.2f}"
              f"  {ts.beol_mean_K:>12.2f}"
              f"  {ts.power_W:>9.2f}")

    print()

    # Package nodes -- show key ones
    hsink = [p for p in package_nodes if p.name.startswith("hsink_")]
    hsp = [p for p in package_nodes if p.name.startswith("hsp_")]
    if hsp:
        print(f"  Heat spreader: {hsp[0].temp_K:.2f} K "
              f"({_k_to_c(hsp[0].temp_K):.2f} C)")
    if hsink:
        print(f"  Heat sink:     {hsink[0].temp_K:.2f} K "
              f"({_k_to_c(hsink[0].temp_K):.2f} C)")
    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def discover_files(sim_dir: str) -> dict[str, str]:
    """Discover input and output files in the simulation directory.

    Required: hotspot.config, stack.lcf, power.ptrace, outputs/steady.out.
    Optional: outputs/grid_steady.out, outputs/transient.ttrace,
              outputs/grid_transient.ttrace.
    """
    found: dict[str, str] = {}
    required = {
        "config": "hotspot.config",
        "lcf": "stack.lcf",
        "ptrace": "power.ptrace",
        "steady": os.path.join("outputs", "steady.out"),
    }
    optional = {
        "grid_steady": os.path.join("outputs", "grid_steady.out"),
        "transient": os.path.join("outputs", "transient.ttrace"),
        "grid_transient": os.path.join("outputs", "grid_transient.ttrace"),
    }

    for key, rel in required.items():
        path = os.path.join(sim_dir, rel)
        if not os.path.isfile(path):
            print(f"Error: required file not found: {path}", file=sys.stderr)
            sys.exit(1)
        found[key] = path

    for key, rel in optional.items():
        path = os.path.join(sim_dir, rel)
        if os.path.isfile(path):
            found[key] = path
        else:
            print(f"  Note: optional file not found: {rel}", file=sys.stderr)

    return found


def postprocess(sim_dir: str, output_dir: Optional[str] = None,
                skip_grid: bool = False) -> None:
    """Main orchestration: parse, aggregate, and output results."""
    files = discover_files(sim_dir)

    if output_dir is None:
        output_dir = os.path.join(sim_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # --- Parse inputs ---
    config = parse_config(files["config"])
    layers = parse_lcf(files["lcf"])
    power_map = parse_ptrace(files["ptrace"])
    block_temps, package_nodes = parse_steady(files["steady"])

    # --- Parse grid (optional) ---
    grid_stats: list[GridLayerStats] = []
    if not skip_grid and "grid_steady" in files:
        grid_stats = parse_grid_steady(
            files["grid_steady"], config.grid_rows, config.grid_cols)

    # --- Aggregate ---
    tier_summaries = aggregate_tiers(layers, grid_stats, power_map,
                                     config.ambient_K)

    # --- Output ---
    yaml_path = os.path.join(output_dir, "results.yaml")
    csv_path = os.path.join(output_dir, "summary.csv")

    yaml_data = build_yaml_dict(config, layers, grid_stats, block_temps,
                                package_nodes, tier_summaries, power_map)
    write_yaml(yaml_data, yaml_path)
    print(f"  Wrote: {yaml_path}")

    write_csv(tier_summaries, config, csv_path)
    print(f"  Wrote: {csv_path}")

    print_summary_table(tier_summaries, config, power_map, package_nodes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process HotSpot 3D thermal simulation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sim_dir",
        help="Path to the simulation directory (e.g., generated_3d_stack/)",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory for YAML/CSV (default: <sim_dir>/outputs/)",
    )
    parser.add_argument(
        "--no-grid", action="store_true",
        help="Skip grid file parsing (faster, block-level stats only)",
    )

    args = parser.parse_args()
    postprocess(args.sim_dir, args.output_dir, args.no_grid)


if __name__ == "__main__":
    main()
