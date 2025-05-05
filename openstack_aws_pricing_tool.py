#!/usr/bin/env python3
"""openstack_aws_pricing_tool.py  (rev‑3)

Adds **monthly, yearly** cost columns and a grand‑total summary row.

USAGE remains the same:
  python openstack_aws_pricing_tool.py --cloud <cloud> --project <proj> \
         --aws-csv ./prices/AmazonEC2-pricing-20250505.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from openstack import connection, exceptions as os_exc
from tqdm import tqdm
from tabulate import tabulate

HOURS_IN_MONTH = 730   # avg 365/12*24
HOURS_IN_YEAR  = 8760

# ────────────────────────────────────────────────────────────────────────────────
# AWS price loading (unchanged)
# ────────────────────────────────────────────────────────────────────────────────

def _load_aws_prices(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        for idx, line in enumerate(f):
            if line.lstrip().startswith(("SKU", "\"SKU\"")):
                header = idx
                break
        else:
            raise ValueError("CSV header not found (no line starting with 'SKU')")

    df = pd.read_csv(csv_path, skiprows=header, quotechar='"', dtype=str)
    df.columns = df.columns.str.strip()
    df["PricePerUnit"] = pd.to_numeric(df["PricePerUnit"], errors="coerce")

    base = df[
        (df["Region Code"] == "us-east-1") & (df["Unit"] == "Hrs") & (df["Currency"] == "USD") &
        (df["Tenancy"].fillna("Shared") == "Shared") & (df["Operating System"].fillna("Linux") == "Linux") &
        (df["PricePerUnit"].notna()) & (df["PricePerUnit"] > 0)
    ]

    od = (base[base["TermType"] == "OnDemand"].groupby("Instance Type")["PricePerUnit"].min().to_dict())
    ri = (base[(base["TermType"] == "Reserved") & (base["LeaseContractLength"] == "3yr") &
               (base["PurchaseOption"] == "No Upfront")]
           .groupby("Instance Type")["PricePerUnit"].min().to_dict())
    return od, ri

# ────────────────────────────────────────────────────────────────────────────────
# Minimal spec map (demo only)
# ────────────────────────────────────────────────────────────────────────────────
_AWS_SPECS: Dict[str, Tuple[int, int]] = {
    "t3.small": (2, 2), "t3.medium": (2, 4), "t3.large": (2, 8), "t3.xlarge": (4, 16),
    "m6a.large": (2, 8), "m6a.xlarge": (4, 16), "m6a.2xlarge": (8, 32), "m6a.4xlarge": (16, 64),
    "c5d.xlarge": (4, 8), "c5d.2xlarge": (8, 16),
}


def _choose_instance(vcpu: int, ram_mb: int, aws_od: Dict[str, float]):
    ram_gib = (ram_mb + 1023) // 1024
    fits = [
        (itype, price) for itype, price in aws_od.items()
        if price and price > 0 and _AWS_SPECS.get(itype, (0, 0))[0] >= vcpu and _AWS_SPECS[itype][1] >= ram_gib
    ]
    return min(fits, key=lambda x: x[1]) if fits else (None, None)

# ────────────────────────────────────────────────────────────────────────────────
# Report builder
# ────────────────────────────────────────────────────────────────────────────────

def build_report(conn, project_id: Optional[str], csv: Path):
    aws_od, aws_ri = _load_aws_prices(csv)

    # flavour cache
    flav_map: Dict[str, object] = {}
    for f in conn.compute.flavors():
        flav_map[f.id] = f
        flav_map[f.name] = f

    rows: List[Dict[str, object]] = []
    missing_shapes: Set[Tuple[int, int]] = set()

    servers = list(conn.compute.servers(details=True, all_projects=True))
    with tqdm(total=len(servers), desc="Fetching VMs", unit="vm") as bar:
        for srv in servers:
            bar.update(1)
            if project_id and srv.project_id != project_id:
                continue

            flav_ref = srv.flavor.get("id") or srv.flavor.get("original_name")
            flav = flav_map.get(flav_ref)
            if not flav:
                try:
                    flav = conn.compute.get_flavor(flav_ref)
                except os_exc.ResourceNotFound:
                    flav = conn.compute.find_flavor(flav_ref, ignore_missing=True)
            if not flav:
                print(f"[WARN] missing flavor {flav_ref} for {srv.name}", file=sys.stderr)
                continue

            vcpus, ram_mb, root_gb = flav.vcpus, flav.ram, flav.disk
            total_disk = root_gb + sum(conn.block_storage.get_volume(att.id).size for att in conn.compute.volume_attachments(server=srv))

            itype, od_hr = _choose_instance(vcpus, ram_mb, aws_od)
            ri_hr = aws_ri.get(itype) if itype else None
            if not itype:
                missing_shapes.add((vcpus, (ram_mb + 1023)//1024))

            rows.append({
                "project": srv.project_id,
                "server": srv.name,
                "vcpus": vcpus,
                "ram_GiB": (ram_mb + 1023)//1024,
                "disk_GB": total_disk,
                "aws_type": itype or "N/A",
                "od$/hr": od_hr,
                "ri3yr$/hr": ri_hr,
                "od$/mo": None if od_hr is None else od_hr * HOURS_IN_MONTH,
                "ri3yr$/mo": None if ri_hr is None else ri_hr * HOURS_IN_MONTH,
                "od$/yr": None if od_hr is None else od_hr * HOURS_IN_YEAR,
                "ri3yr$/yr": None if ri_hr is None else ri_hr * HOURS_IN_YEAR,
            })

    return rows, missing_shapes

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compare OpenStack VM costs to AWS")
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--project", help="OpenStack project name or ID")
    scope.add_argument("--all-projects", action="store_true")
    p.add_argument("--cloud", default="openstack")
    p.add_argument("--aws-csv", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    conn = connection.from_config(cloud=args.cloud)
    proj_id = None if args.all_projects else (conn.identity.find_project(args.project, ignore_missing=True) or args.project).id if not args.all_projects else None

    rows, missing = build_report(conn, proj_id, args.aws_csv)
    df = pd.DataFrame(rows)

    # Add grand‑total row for numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    total_row = {c: df[c].sum() if c in numeric_cols else ("TOTAL" if c == "server" else "") for c in df.columns}
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Report saved → {args.output}")
    else:
        print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))

    if missing:
        print("[WARN] AWS spec map missing shapes:", sorted(missing), file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
