#!/usr/bin/env python3
"""openstack_aws_pricing_tool.py  (rev‑2)

CLI utility that compares the cost of running OpenStack VMs against AWS EC2.
Now robust against flavours referenced **by name** (e.g. `gp.small`) and fixes
minor syntax typos from the previous iteration.

 USAGE (unchanged):
   python openstack_aws_pricing_tool.py \
       --cloud dc1-cdl-osp4 \
       --project osp-c1npgt04-Infr-Arch \
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

# ────────────────────────────────────────────────────────────────────────────────
# AWS price loading (same as before, header‑skip intact)
# ────────────────────────────────────────────────────────────────────────────────

def _load_aws_prices(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        for idx, line in enumerate(f):
            if line.lstrip().startswith("SKU") or line.lstrip().startswith("\"SKU\""):
                header = idx
                break
        else:
            raise ValueError("Could not locate CSV header in price file")

    df = pd.read_csv(csv_path, skiprows=header, quotechar='"', dtype=str)
    df.columns = df.columns.str.strip()
    df["PricePerUnit"] = pd.to_numeric(df["PricePerUnit"], errors="coerce")

    base = df[
        (df["Region Code"] == "us-east-1")
        & (df["Unit"] == "Hrs") & (df["Currency"] == "USD")
        & (df["Tenancy"].fillna("Shared") == "Shared")
        & (df["Operating System"].fillna("Linux") == "Linux")
    ]

    on_demand = (base[base["TermType"] == "OnDemand"]
                 .groupby("Instance Type")["PricePerUnit"].min().to_dict())
    ri_3yr = (base[(base["TermType"] == "Reserved")
                   & (base["LeaseContractLength"] == "3yr")
                   & (base["PurchaseOption"] == "No Upfront")]
              .groupby("Instance Type")["PricePerUnit"].min().to_dict())
    return on_demand, ri_3yr

# ────────────────────────────────────────────────────────────────────────────────
# AWS instance specs – still minimal demo list
# ────────────────────────────────────────────────────────────────────────────────
_AWS_SPECS: Dict[str, Tuple[int, int]] = {
    "t3.small": (2, 2),
    "t3.medium": (2, 4),
    "t3.large": (2, 8),
    "t3.xlarge": (4, 16),
    "m6a.large": (2, 8),
    "m6a.xlarge": (4, 16),
    "m6a.2xlarge": (8, 32),
    "m6a.4xlarge": (16, 64),
    "c5d.xlarge": (4, 8),
    "c5d.2xlarge": (8, 16),
}


def _choose_instance(vcpu: int, ram_mb: int, aws_od: Dict[str, float]):
    ram_gib = (ram_mb + 1023) // 1024
    fits = [(itype, price) for itype, price in aws_od.items()
            if _AWS_SPECS.get(itype, (0, 0))[0] >= vcpu
            and _AWS_SPECS.get(itype, (0, 0))[1] >= ram_gib]
    return min(fits, key=lambda x: x[1]) if fits else (None, None)

# ────────────────────────────────────────────────────────────────────────────────
# Report builder
# ────────────────────────────────────────────────────────────────────────────────

def build_report(conn, project_id: Optional[str], csv_path: Path):
    aws_od, aws_ri = _load_aws_prices(csv_path)

    # Cache *all* flavors once: lookup by both ID *and* name
    flavors: Dict[str, object] = {}
    for flav in conn.compute.flavors():
        flavors[flav.id] = flav
        flavors[flav.name] = flav

    rows: List[Dict[str, object]] = []
    missing_shapes: Set[Tuple[int, int]] = set()

    servers = list(conn.compute.servers(details=True, all_projects=True))
    with tqdm(total=len(servers), desc="Fetching VMs", unit="vm") as bar:
        for srv in servers:
            bar.update(1)
            if project_id and srv.project_id != project_id:
                continue

            flav_ref = srv.flavor.get("id") or srv.flavor.get("original_name")
            flav = flavors.get(flav_ref)
            if not flav:
                try:
                    flav = conn.compute.get_flavor(flav_ref)
                except os_exc.ResourceNotFound:
                    flav = conn.compute.find_flavor(flav_ref, ignore_missing=True)
            if not flav:
                print(f"[WARN] flavor not found for server {srv.name}: {flav_ref}", file=sys.stderr)
                continue

            vcpus, ram_mb, root_gb = flav.vcpus, flav.ram, flav.disk
            total_disk = root_gb
            for att in conn.compute.volume_attachments(server=srv):
                vol = conn.block_storage.get_volume(att.id)
                total_disk += vol.size

            itype, od = _choose_instance(vcpus, ram_mb, aws_od)
            ri = aws_ri.get(itype) if itype else None
            if not itype:
                missing_shapes.add((vcpus, (ram_mb + 1023)//1024))

            rows.append({
                "project": srv.project_id,
                "server": srv.name,
                "vcpus": vcpus,
                "ram_GiB": (ram_mb + 1023)//1024,
                "disk_GB": total_disk,
                "aws_type": itype or "N/A",
                "on_demand$/hr": od,
                "ri3yr$/hr": ri,
            })
    return rows, missing_shapes

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compare OpenStack VM costs to AWS")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--project", help="OpenStack project name or ID")
    grp.add_argument("--all-projects", action="store_true")
    ap.add_argument("--cloud", default="openstack")
    ap.add_argument("--aws-csv", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    conn = connection.from_config(cloud=args.cloud)

    proj_id = None
    if not args.all_projects:
        found = conn.identity.find_project(args.project, ignore_missing=True)
        proj_id = found.id if found else args.project

    rows, missing = build_report(conn, proj_id, args.aws_csv)
    if missing:
        print("[WARN] AWS spec map missing shapes:", sorted(missing), file=sys.stderr)

    df = pd.DataFrame(rows)
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Report saved → {args.output}")
    else:
        print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
