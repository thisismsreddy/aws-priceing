#!/usr/bin/env python3
"""openstack_aws_pricing_tool.py  –  robust real‑pricing edition (bug‑fixed)

* Parses AWS price CSV with flexible column names
* Builds full EC2 spec map and per‑GB gp3 cost
* Adds storage cost to instance hourly price
* Outputs hourly, monthly, yearly totals + grand total row
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openstack import connection, exceptions as os_exc
from tabulate import tabulate
from tqdm import tqdm

HOURS_IN_MONTH = 730
HOURS_IN_YEAR = 8_760
_MULTI_SPC = re.compile(r"\s+")

# ──────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    with path.open("r", newline="") as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith(("SKU", "\"SKU\"")):
                header = i
                break
        else:
            raise ValueError("Header row not found (SKU)")
    return pd.read_csv(path, skiprows=header, quotechar='"', dtype=str)


def _norm_cols(df: pd.DataFrame) -> None:
    df.columns = (_MULTI_SPC.sub(" ", c.strip()).title() for c in df.columns)
    alias = {
        'Instancetype': 'Instance Type',
        'Vpcu': 'Vcpu',
        'Priceperunit': 'Price Per Unit',
        'Termtype': 'Term Type',
        'Leasecontractlength': 'Lease Contract Length',
        'Purchaseoption': 'Purchase Option',
    }
    df.rename(columns={k: v for k, v in alias.items() if k in df.columns}, inplace=True)


REQ = {
    'Instance Type', 'Vcpu', 'Memory', 'Product Family', 'Region Code', 'Unit',
    'Currency', 'Tenancy', 'Operating System', 'Price Per Unit', 'Term Type'
}


def _extract_pricing(df: pd.DataFrame):
    """Return (on_demand, ri_3yr_no_upfront, spec_map, gp3_hourly_per_gb)"""
    _norm_cols(df)
    missing = REQ - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')

    inst = df[
        (df['Product Family'].str.lower() == 'compute instance') &
        (df['Region Code'].str.lower() == 'us-east-1') &
        (df['Unit'] == 'Hrs') & (df['Currency'] == 'USD') &
        (df['Tenancy'].fillna('Shared') == 'Shared') &
        (df['Operating System'].fillna('Linux') == 'Linux') &
        (df['Price Per Unit'] > 0)
    ]

    # spec map
    spec: Dict[str, Tuple[int, int]] = {}
    for itype, grp in inst.groupby('Instance Type'):
        row = grp.iloc[0]
        mem = int(float(str(row['Memory']).split()[0]))
        spec[itype] = (int(row['Vcpu']), mem)

    od = (inst[inst['Term Type'] == 'OnDemand']
          .groupby('Instance Type')['Price Per Unit']
          .min().to_dict())

    ri = (inst[(inst['Term Type'] == 'Reserved') &
               (inst['Lease Contract Length'].str.startswith('3')) &
               (inst['Purchase Option'] == 'No Upfront')]
           .groupby('Instance Type')['Price Per Unit']
           .min().to_dict())

    # gp3 storage cost (/hr per GB)
    storage = df.copy()
    _norm_cols(storage)
    gp3_row = storage[(storage['Product Family'].str.lower() == 'storage') &
                      (storage['Volume Api Name'] == 'gp3') &
                      (storage['Region Code'].str.lower() == 'us-east-1') &
                      (storage['Unit'].str.contains('GB')) &
                      (storage['Term Type'] == 'OnDemand')]
    gp3_hr = pd.to_numeric(gp3_row['Price Per Unit']).min() / HOURS_IN_MONTH

    return od, ri, spec, gp3_hr

# ──────────────────────────────────────────────────────────────
# Instance picker
# ──────────────────────────────────────────────────────────────

def _pick_shape(vcpu: int, ram_mb: int, prices: Dict[str, float], spec: Dict[str, Tuple[int, int]]):
    ram_gib = (ram_mb + 1023) // 1024
    fits = [(t, p) for t, p in prices.items() if spec[t][0] >= vcpu and spec[t][1] >= ram_gib]
    return min(fits, key=lambda x: x[1]) if fits else (None, None)

# ──────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────

def build_report(conn, pid: Optional[str], csv: Path):
    od, ri, spec, gp3 = _extract_pricing(_load_csv(csv))

    flavors = {f.id: f for f in conn.compute.flavors()}
    flavors.update({f.name: f for f in flavors.values()})

    seen_vol: Dict[str, int] = {}
    rows: List[Dict[str, object]] = []

    servers = list(conn.compute.servers(details=True, all_projects=True))
    with tqdm(total=len(servers), desc='VMs', unit='vm') as bar:
        for s in servers:
            bar.update(1)
            if pid and s.project_id != pid:
                continue

            f_ref = s.flavor.get('id') or s.flavor.get('original_name')
            flv = flavors.get(f_ref) or conn.compute.find_flavor(f_ref, ignore_missing=True)
            if not flv:
                print('[WARN] missing flavor', f_ref, file=sys.stderr)
                continue

            vcpu, ram_mb, root = flv.vcpus, flv.ram, flv.disk
            disk = root
            for att in conn.compute.volume_attachments(server=s):
                if att.id not in seen_vol:
                    seen_vol[att.id] = conn.block_storage.get_volume(att.id).size
                disk += seen_vol[att.id]

            itype, inst_od = _pick_shape(vcpu, ram_mb, od, spec)
            inst_ri = ri.get(itype) if itype else None
            storage_hr = disk * gp3

            rows.append({
                'project': s.project_id,
                'server': s.name,
                'vcpus': vcpu,
                'ram_GiB': (ram_mb + 1023)//1024,
                'disk_GB': disk,
                'aws_type': itype or 'N/A',
                'od$/hr': None if inst_od is None else inst_od + storage_hr,
                'ri3yr$/hr': None if inst_ri is None else inst_ri + storage_hr,
            })

    df = pd.DataFrame(rows)
    for col in ['od$/hr', 'ri3yr$/hr']:
        df[col.replace('/hr', '/mo')] = df[col] * HOURS_IN_MONTH
        df[col.replace('/hr', '/yr')] = df[col] * HOURS_IN_YEAR

    total = {c: df[c].sum() if df[c].dtype.kind in 'f' else ('TOTAL' if c == 'server' else '') for c in df.columns}
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    return df

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='OpenStack → AWS cost comparison')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--project')
    g.add_argument('--all-projects', action='store_true')
    ap.add_argument('--cloud', default='openstack')
    ap.add_argument('--aws-csv', required=True, type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()

    conn = connection.from_config(cloud=args.cloud)
    pid = None
    if not args.all_projects:
        proj = conn.identity.find_project(args.project, ignore_missing=True)
        pid = proj.id if proj else args.project

    df = build_report(conn, pid, args.aws_csv)
    if args.output:
        df.to_csv(args.output, index=False)
        print('saved →', args.output)
    else:
        print(tabulate(df, headers='keys', tablefmt='github', floatfmt='.4f'))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
