#!/usr/bin/env python3
"""openstack_aws_pricing_tool.py  —  *real‑pricing edition*

Compares the cost of running OpenStack VMs to AWS **with:**
• **Complete EC2 instance catalog** (auto‑parsed from the provided price CSV)
• **EBS gp3 storage pricing** per attached volume (deduplicated across multi‑attach)
• Hourly, monthly (730 h) and yearly (8 760 h) totals for **On‑Demand** and **3‑yr No‑Upfront RI**
• Grand totals at the bottom

Dependencies
------------
```bash
pip install openstacksdk pandas tqdm tabulate
```

Example
-------
```bash
python openstack_aws_pricing_tool.py \
  --cloud dc1-cdl-osp4 \
  --project osp-c1npgt04-Infr-Arch \
  --aws-csv ./prices/AmazonEC2-pricing-20250505.csv
```
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

HOURS_IN_MONTH = 730  # avg
HOURS_IN_YEAR = 8_760

# ────────────────────────────────────────────────────────────────────────────────
# 1. Parse AWS price list CSV
# ────────────────────────────────────────────────────────────────────────────────

def _read_price_csv(csv_path: Path) -> pd.DataFrame:
    """Return DataFrame starting at header row (auto‑detect)."""
    with csv_path.open("r", newline="") as f:
        for idx, line in enumerate(f):
            if line.lstrip().startswith(("SKU", "\"SKU\"")):
                header_idx = idx
                break
        else:
            raise ValueError("Could not find header row (starts with SKU)")
    return pd.read_csv(csv_path, skiprows=header_idx, quotechar='"', dtype=str)


def _extract_instance_prices(df: pd.DataFrame):
    """Return (on_demand, ri_3yr, spec_map).

    * on_demand / ri_3yr: {instance_type: hourly USD}
    * spec_map: {instance_type: (vCPU, memory_GiB)}
    """
    df.columns = df.columns.str.strip()
    df["PricePerUnit"] = pd.to_numeric(df["PricePerUnit"], errors="coerce")

    inst = df[
        (df["Product Family"] == "Compute Instance") &
        (df["Region Code"] == "us-east-1") &
        (df["Unit"] == "Hrs") &
        (df["Currency"] == "USD") &
        (df["Tenancy"].fillna("Shared") == "Shared") &
        (df["Operating System"].fillna("Linux") == "Linux") &
        (df["PricePerUnit"].notna()) & (df["PricePerUnit"] > 0)
    ].copy()

    # Build spec map (vCPU, Mem GiB)
    spec_map: Dict[str, Tuple[int, int]] = {}
    for _, row in inst.groupby("Instance Type").first().iterrows():
        spec_map[row["Instance Type"]] = (int(row["vCPU"]), int(float(row["Memory"].split()[0])))

    od = (
        inst[inst["TermType"] == "OnDemand"]
        .groupby("Instance Type")["PricePerUnit"].min().to_dict()
    )

    ri = (
        inst[(inst["TermType"] == "Reserved") &
             (inst["LeaseContractLength"] == "3yr") &
             (inst["PurchaseOption"] == "No Upfront")]
            .groupby("Instance Type")["PricePerUnit"].min().to_dict()
    )
    return od, ri, spec_map


def _extract_gp3_price(df: pd.DataFrame) -> float:
    """Return hourly USD per‑GB for gp3 (On‑Demand)."""
    df.columns = df.columns.str.strip()
    gp3 = df[
        (df["Product Family"] == "Storage") &
        (df["Volume API Name"] == "gp3") &
        (df["Region Code"] == "us-east-1") &
        (df["Unit"].str.contains("GB")) &
        (df["TermType"] == "OnDemand") &
        (df["PricePerUnit"].notna()) & (pd.to_numeric(df["PricePerUnit"]) > 0)
    ]
    if gp3.empty:
        raise ValueError("gp3 pricing not found in CSV")
    price_gb_month = gp3.iloc[0]["PricePerUnit"].astype(float)
    return price_gb_month / HOURS_IN_MONTH  # convert to hourly

# ────────────────────────────────────────────────────────────────────────────────
# 2. Choose cheapest instance meeting vCPU/RAM
# ────────────────────────────────────────────────────────────────────────────────

def _choose_instance(vcpu: int, ram_mb: int, od_prices: Dict[str, float], spec: Dict[str, Tuple[int, int]]):
    ram_gib = (ram_mb + 1023) // 1024
    fits = [
        (itype, price) for itype, price in od_prices.items()
        if spec.get(itype, (0, 0))[0] >= vcpu and spec[itype][1] >= ram_gib
    ]
    return min(fits, key=lambda x: x[1]) if fits else (None, None)

# ────────────────────────────────────────────────────────────────────────────────
# 3. Build cost report
# ────────────────────────────────────────────────────────────────────────────────

def build_report(conn, project_id: Optional[str], price_csv: Path):
    df_price = _read_price_csv(price_csv)
    od_price, ri_price, spec_map = _extract_instance_prices(df_price)
    gp3_hr = _extract_gp3_price(df_price)

    # flavour cache
    flav_cache: Dict[str, object] = {f.id: f for f in conn.compute.flavors()}
    flav_cache.update({f.name: f for f in flav_cache.values()})

    rows: List[Dict[str, object]] = []
    missing_shapes: Set[Tuple[int, int]] = set()

    seen_vols: Dict[str, int] = {}  # volume_id → size (dedup across multi‑attach)

    servers = list(conn.compute.servers(details=True, all_projects=True))
    with tqdm(total=len(servers), desc="Fetching VMs", unit="vm") as bar:
        for srv in servers:
            bar.update(1)
            if project_id and srv.project_id != project_id:
                continue

            flav_ref = srv.flavor.get("id") or srv.flavor.get("original_name")
            flav = flav_cache.get(flav_ref) or conn.compute.find_flavor(flav_ref, ignore_missing=True)
            if not flav:
                print(f"[WARN] missing flavor {flav_ref} for {srv.name}", file=sys.stderr)
                continue

            vcpus, ram_mb, root_gb = flav.vcpus, flav.ram, flav.disk

            # Attached volumes (dedup)
            total_disk = root_gb
            for att in conn.compute.volume_attachments(server=srv):
                if att.id not in seen_vols:
                    vol = conn.block_storage.get_volume(att.id)
                    seen_vols[att.id] = vol.size
                total_disk += seen_vols[att.id]

            itype, od_hr = _choose_instance(vcpus, ram_mb, od_price, spec_map)
            ri_hr = ri_price.get(itype) if itype else None
            if not itype:
                missing_shapes.add((vcpus, (ram_mb + 1023)//1024))

            storage_hr = total_disk * gp3_hr

            rows.append({
                "project": srv.project_id,
                "server": srv.name,
                "vcpus": vcpus,
                "ram_GiB": (ram_mb + 1023)//1024,
                "disk_GB": total_disk,
                "aws_type": itype or "N/A",
                "od$/hr": None if od_hr is None else od_hr + storage_hr,
                "ri3yr$/hr": None if ri_hr is None else ri_hr + storage_hr,
            })

    df = pd.DataFrame(rows)
    # expand monthly/yearly columns
    for col in ["od$/hr", "ri3yr$/hr"]:
        df[col.replace("/hr", "/mo")] = df[col] * HOURS_IN_MONTH
        df[col.replace("/hr", "/yr")] = df[col] * HOURS_IN_YEAR

    # grand total row
    numeric_cols = df.select_dtypes("number").columns
    total = {c: df[c].sum() if c in numeric_cols else ("TOTAL" if c == "server" else "") for c in df.columns}
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

    return df, missing_shapes

# ────────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compare OpenStack VM costs to AWS (real pricing)")
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
        proj = conn.identity.find_project(args.project, ignore_missing=True)
        proj_id = proj.id if proj else args.project  # accept raw UUID fallback

    df, missing = build_report(conn, proj_id, args.aws_csv)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Report saved → {args.output}")
    else:
        print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))

    if missing:
        print("[WARN] no AWS shape match for:", sorted(missing), file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
