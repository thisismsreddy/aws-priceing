#!/usr/bin/env python3
"""openstack_aws_pricing_tool.py

A single‑file CLI utility that:
1. Authenticates to OpenStack using the profile defined in clouds.yaml.
2. Fetches the VM inventory for either one project or *all* projects.
3. Shows a live progress bar while collecting data.
4. Loads a pre‑downloaded AWS EC2 price list CSV (On‑Demand + 3‑year No‑Upfront Reserved).
5. Maps each OpenStack VM to the cheapest matching AWS instance type and calculates costs.
6. Prints a compact tabular report (or CSV via --output) with OpenStack vs AWS pricing.

Dependencies:
  pip install openstacksdk pandas tqdm tabulate

Usage examples:
  python openstack_aws_pricing_tool.py --cloud mycloud --project demo \
         --aws-csv ./prices/AmazonEC2-pricing-20250504.csv
  python openstack_aws_pricing_tool.py --cloud mycloud --all-projects \
         --aws-csv ./prices/latest.csv --output report.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from openstack import connection
from tqdm import tqdm
from tabulate import tabulate

# ────────────────────────────────────────────────────────────────────────────────
# Helpers for AWS price look‑ups
# ────────────────────────────────────────────────────────────────────────────────

def _load_aws_prices(csv_path: Path):
    """Return two dicts keyed by instance type → hourly price.

    * on_demand[type]  ➜ float
    * ri_3yr_no_upfront[type] ➜ float
    """
    df = pd.read_csv(csv_path)

    # Normalise column names (strip leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Pre‑filter by region, unit and currency once for speed
    df = df[
        (df["Region Code"] == "us-east-1")
        & (df["Unit"] == "Hrs")
        & (df["Currency"] == "USD")
        & (df["Tenancy"].fillna("Shared") == "Shared")  # Ignore Dedicated/Host
        & (df["Operating System"].fillna("Linux") == "Linux")
    ]

    on_demand = (
        df[df["TermType"] == "OnDemand"]
        .groupby("Instance Type")["PricePerUnit"]
        .min()
        .astype(float)
        .to_dict()
    )

    ri_filter = (
        (df["TermType"] == "Reserved")
        & (df["LeaseContractLength"] == "3yr")
        & (df["PurchaseOption"] == "No Upfront")
    )
    ri_3yr = (
        df[ri_filter]
        .groupby("Instance Type")["PricePerUnit"]
        .min()
        .astype(float)
        .to_dict()
    )

    return on_demand, ri_3yr


# ────────────────────────────────────────────────────────────────────────────────
# OpenStack → AWS mapping logic (very naive vCPU/RAM fit)
# ────────────────────────────────────────────────────────────────────────────────

def _choose_instance(vcpus: int, ram_mb: int, aws_od: dict):
    """Return the cheapest instance type that fits vCPU & RAM (GiB)."""
    ram_gib = (ram_mb + 1023) // 1024  # round‑up MiB → GiB

    candidates = [
        (itype, price)
        for itype, price in aws_od.items()
        if _AWS_SPECS.get(itype, (None, None))[0] >= vcpus
        and _AWS_SPECS.get(itype, (None, None))[1] >= ram_gib
    ]

    if not candidates:
        return None, None  # no match

    return min(candidates, key=lambda x: x[1])  # (itype, price)


# NOTE: A tiny subset to keep the script self‑contained.
# In production, load this from the same CSV (Instance Type, vCPU, Memory) once.
_AWS_SPECS = {
    "t3.micro": (2, 1),
    "t3.small": (2, 2),
    "t3.medium": (2, 4),
    "t3.large": (2, 8),
    "t3.xlarge": (4, 16),
    "t3.2xlarge": (8, 32),
    "m6a.large": (2, 8),
    "m6a.xlarge": (4, 16),
    "m6a.2xlarge": (8, 32),
    "m6a.4xlarge": (16, 64),
    "m6a.8xlarge": (32, 128),
    "m6a.16xlarge": (64, 256),
    "m6a.24xlarge": (96, 384),
    "c5d.xlarge": (4, 8),
    "c5d.2xlarge": (8, 16),
}


# ────────────────────────────────────────────────────────────────────────────────
# Main inventory / cost routine
# ────────────────────────────────────────────────────────────────────────────────

def build_report(conn, project_id: str | None, aws_prices_csv: Path):
    aws_od, aws_ri = _load_aws_prices(aws_prices_csv)

    rows = []
    missing_map = set()

    srv_iter = conn.compute.servers(details=True, all_projects=project_id is None)
    srv_iter = list(srv_iter)  # materialise once for length

    with tqdm(total=len(srv_iter), desc="Fetching VMs", unit="vm") as bar:
        for srv in srv_iter:
            if project_id and srv.project_id != project_id:
                bar.update(1)
                continue

            flav = conn.compute.get_flavor(srv.flavor["id"])
            vcpus, ram, root_gb = flav.vcpus, flav.ram, flav.disk

            # Extra volumes
            total_disk = root_gb
            for att in conn.compute.volume_attachments(server=srv):
                vol = conn.block_storage.get_volume(att.id)
                total_disk += vol.size

            itype, od_price = _choose_instance(vcpus, ram, aws_od)
            ri_price = aws_ri.get(itype)
            if not itype:
                missing_map.add((vcpus, ram))

            rows.append(
                {
                    "project": srv.project_id,
                    "server": srv.name,
                    "vcpus": vcpus,
                    "ram_gib": (ram + 1023) // 1024,
                    "disk_gb": total_disk,
                    "aws_type": itype or "N/A",
                    "cost_od_hr": od_price,
                    "cost_ri3y_hr": ri_price,
                }
            )
            bar.update(1)

    return rows, missing_map


# ────────────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ────────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compare OpenStack VM costs against AWS")
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--project", help="OpenStack project name or ID")
    scope.add_argument("--all-projects", action="store_true", help="Include all projects")
    p.add_argument("--cloud", default="openstack", help="clouds.yaml profile name")
    p.add_argument("--aws-csv", required=True, type=Path, help="EC2 price list CSV path")
    p.add_argument("--output", type=Path, help="Optional CSV output path")
    args = p.parse_args()

    conn = connection.from_config(cloud=args.cloud)

    rows, missing = build_report(conn, None if args.all_projects else args.project, args.aws_csv)

    if missing:
        print("[WARN] No AWS match for VM shapes:", sorted(missing), file=sys.stderr)

    if args.output:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Report written to {args.output}")
    else:
        print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
