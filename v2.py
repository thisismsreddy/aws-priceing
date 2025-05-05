#!/usr/bin/env python3
"""openstack_aws_pricing_tool.py – clean build (syntactically valid)

• Friendly column names (OnDemand_*/RI3yr_*)
• Strict CSV filters (current‑gen, canonical OD & RI rows)
• Includes gp3 storage cost (deduped)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openstack import connection
from tabulate import tabulate
from tqdm import tqdm

HOURS_IN_MONTH = 730
HOURS_IN_YEAR = 8_760
_SPACE_RE = re.compile(r"\s+")

# ───────────────────────────────────────── CSV helpers ──────

def _load_csv(path: Path) -> pd.DataFrame:
    """Load AWS price CSV starting at header row."""
    with path.open() as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith(("SKU", "\"SKU\"")):
                header = i
                break
        else:
            raise ValueError("Header row not found in price file")
    return pd.read_csv(path, skiprows=header, quotechar='"', dtype=str)


def _norm_cols(df: pd.DataFrame) -> None:
    """Normalise column names in‑place and add aliases."""
    df.columns = (_SPACE_RE.sub(" ", c.strip()).title() for c in df.columns)
    alias = {
        'Instancetype': 'Instance Type',
        'Vpcu': 'Vcpu',
        'Priceperunit': 'Price Per Unit',
        'Pricedescription': 'Price Description',
        'Termtype': 'Term Type',
        'Leasecontractlength': 'Lease Contract Length',
        'Purchaseoption': 'Purchase Option',
    }
    df.rename(columns={k: v for k, v in alias.items() if k in df.columns}, inplace=True)

# ─────────────────────────────────── pricing extraction ─────

def _extract_pricing(df: pd.DataFrame):
    _norm_cols(df)
    df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')

    base = df[
        (df['Product Family'].str.lower() == 'compute instance') &
        (df['Region Code'].str.lower() == 'us-east-1') &
        (df['Unit'] == 'Hrs') & (df['Currency'] == 'USD') &
        (df['Tenancy'].fillna('Shared') == 'Shared') &
        (df['Operating System'].fillna('Linux') == 'Linux') &
        (df['Current Generation'].fillna('Yes') == 'Yes') &
        (df['Price Per Unit'] > 0)
    ].copy()

    # Spec map
    spec: Dict[str, Tuple[int, int]] = {}
    for itype, grp in base.groupby('Instance Type'):
        first = grp.iloc[0]
        mem_gib = int(float(str(first['Memory']).split()[0]))
        spec[itype] = (int(first['Vcpu']), mem_gib)

    od_rows = base[(base['Term Type'] == 'OnDemand') &
                   (base.get('Price Description', '').str.contains('per On Demand', case=False, na=False))]
    od = od_rows.groupby('Instance Type')['Price Per Unit'].min().to_dict()

    ri_rows = base[(base['Term Type'] == 'Reserved') &
                   (base['Lease Contract Length'].str.startswith('3')) &
                   (base['Purchase Option'] == 'No Upfront') &
                   (~base.get('Price Description', '').str.contains('Upfront Fee', case=False, na=False))]
    ri = ri_rows.groupby('Instance Type')['Price Per Unit'].min().to_dict()

    # gp3 storage $/hr/GB
    _norm_cols(df)
    gp3_row = df[(df['Product Family'].str.lower() == 'storage') &
                 (df['Volume Api Name'] == 'gp3') &
                 (df['Region Code'].str.lower() == 'us-east-1') &
                 (df['Unit'].str.contains('GB')) &
                 (df['Term Type'] == 'OnDemand') &
                 (df['Price Per Unit'] > 0)]
    gp3_hr = pd.to_numeric(gp3_row['Price Per Unit']).min() / HOURS_IN_MONTH

    return od, ri, spec, gp3_hr

# ───────────────────────────── instance chooser ─────────────

def _pick_shape(vcpu: int, ram_mb: int, prices: Dict[str, float], spec: Dict[str, Tuple[int, int]]):
    ram_gib = (ram_mb + 1023) // 1024
    choices = [(t, p) for t, p in prices.items() if spec[t][0] >= vcpu and spec[t][1] >= ram_gib]
    return min(choices, key=lambda x: x[1]) if choices else (None, None)

# ─────────────────────────── report builder ────────────────

def build_report(conn, project_id: Optional[str], csv_path: Path):
    od, ri, spec, gp3 = _extract_pricing(_load_csv(csv_path))

    flavors = {f.id: f for f in conn.compute.flavors()}
    flavors.update({f.name: f for f in flavors.values()})

    seen_vol: Dict[str, int] = {}
    rows: List[Dict[str, object]] = []

    servers = list(conn.compute.servers(details=True, all_projects=True))
    with tqdm(total=len(servers), desc='VMs', unit='vm') as bar:
        for srv in servers:
            bar.update(1)
            if project_id and srv.project_id != project_id:
                continue

            f_ref = srv.flavor.get('id') or srv.flavor.get('original_name')
            flav = flavors.get(f_ref) or conn.compute.find_flavor(f_ref, ignore_missing=True)
            if not flav:
                continue

            vcpus, ram_mb, root = flav.vcpus, flav.ram, flav.disk
            disk = root
            for att in conn.compute.volume_attachments(server=srv):
                if att.id not in seen_vol:
                    seen_vol[att.id] = conn.block_storage.get_volume(att.id).size
                disk += seen_vol[att.id]

            itype, od_hr_inst = _pick_shape(vcpus, ram_mb, od, spec)
            ri_hr_inst = ri.get(itype) if itype else None
            storage_hr = disk * gp3

            rows.append({
                'Project': srv.project_id,
                'Server': srv.name,
                'vCPU': vcpus,
                'RAM_GiB': (ram_mb + 1023)//1024,
                'Disk_GB': disk,
                'AWS_Type': itype or 'N/A',
                'OnDemand_Hourly': None if od_hr_inst is None else od_hr_inst + storage_hr,
                'RI3yr_Hourly': None if ri_hr_inst is None else ri_hr_inst + storage_hr,
            })

    df = pd.DataFrame(rows)
    for col in ['OnDemand_Hourly', 'RI3yr_Hourly']:
        df[col.replace('Hourly', 'Monthly')] = df[col] * HOURS_IN_MONTH
        df[col.replace('Hourly', 'Yearly')] = df[col] * HOURS_IN_YEAR

    total = {c: df[c].sum() if df[c].dtype.kind in 'f' else ('TOTAL' if c == 'Server' else '') for c in df.columns}
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)

# ───────────────────────────── CLI ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description='OpenStack → AWS cost comparison')
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--project')
    grp.add_argument('--all-projects', action='store_true')
    ap.add_argument('--cloud', default='openstack')
    ap.add_argument('--aws-csv', required=True, type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()

    conn = connection.from_config(cloud=args.cloud)
    proj_id = None
    if not args.all_projects:
        pr = conn.identity.find_project(args.project, ignore_missing=True)
        proj_id = pr.id if pr else args.project

    df = build_report(conn, proj_id, args.aws_csv)

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
