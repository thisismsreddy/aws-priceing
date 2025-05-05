#!/usr/bin/env python3
"""openstack_aws_pricing_tool.py  —  robust real‑pricing edition

* Auto‑parses *all* EC2 shapes from a price CSV (column‑name agnostic).
* Adds gp3 storage cost.
* Calculates hourly / monthly / yearly totals (On‑Demand & 3‑yr RI).
* Handles header variations (e.g., "InstanceType" vs "Instance Type").
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

# ──────────────────────────────────────────────────────────────
# Helpers for AWS CSV
# ──────────────────────────────────────────────────────────────

_RE_MULTI_SPACE = re.compile(r"\s+")


def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV starting at the header (auto‑detect)."""
    with csv_path.open('r', newline='') as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith(('SKU', '"SKU"')):
                header_row = i
                break
        else:
            raise ValueError('Header row not found in AWS price file')
    return pd.read_csv(csv_path, skiprows=header_row, quotechar='"', dtype=str)


def _normalise_cols(df: pd.DataFrame) -> None:
    """Normalise header names in‑place: strip, collapse spaces, title‑case."""
    df.columns = (_RE_MULTI_SPACE.sub(' ', c.strip()).title() for c in df.columns)

    # Canonical aliases
    alias = {
        'Instancetype': 'Instance Type',
        'Vpcu': 'Vcpu',
        'Priceperunit': 'Price Per Unit',
        'Termtype': 'Term Type',
        'Leasecontractlength': 'Lease Contract Length',
        'Purchaseoption': 'Purchase Option',
    }
    df.rename(columns={k: v for k, v in alias.items() if k in df.columns}, inplace=True)


# required logical columns we rely on
REQ = {
    'Instance Type', 'Vcpu', 'Memory', 'Product Family', 'Region Code',
    'Unit', 'Currency', 'Tenancy', 'Operating System', 'Price Per Unit',
    'Term Type'
}


def _instance_and_spec(df: pd.DataFrame):
    """Return (on_demand, ri_3yr, spec_map) dicts."""
    _normalise_cols(df)
    missing = REQ - set(df.columns)
    if missing:
        raise ValueError(f'Price CSV missing columns: {missing}')

    df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')

    inst = df[
        (df['Product Family'].str.lower() == 'compute instance') &
        (df['Region Code'].str.lower() == 'us-east-1') &
        (df['Unit'] == 'Hrs') &
        (df['Currency'] == 'USD') &
        (df['Tenancy'].fillna('Shared') == 'Shared') &
        (df['Operating System'].fillna('Linux') == 'Linux') &
        (df['Price Per Unit'] > 0)
    ].copy()

    # spec map
    spec_map: Dict[str, Tuple[int, int]] = {}
    for _, row in inst.groupby('Instance Type').first().iterrows():
        mem_gib = int(float(str(row['Memory']).split()[0]))
        spec_map[row['Instance Type']] = (int(row['Vcpu']), mem_gib)

    od = (inst[inst['Term Type'] == 'OnDemand']
           .groupby('Instance Type')['Price Per Unit']
           .min().to_dict())

    ri = (inst[(inst['Term Type'] == 'Reserved') &
               (inst['Lease Contract Length'].str.startswith('3')) &
               (inst['Purchase Option'] == 'No Upfront')]
            .groupby('Instance Type')['Price Per Unit']
            .min().to_dict())
    return od, ri, spec_map


def _gp3_price_hr(df: pd.DataFrame) -> float:
    _normalise_cols(df)
    storage = df[
        (df['Product Family'].str.lower() == 'storage') &
        (df['Volume Api Name'] == 'gp3') &
        (df['Region Code'].str.lower() == 'us-east-1') &
        (df['Unit'].str.contains('GB')) &
        (df['Term Type'] == 'OnDemand')
    ]
    price_gb_month = pd.to_numeric(storage['Price Per Unit'].dropna()).min()
    if pd.isna(price_gb_month):
        raise ValueError('gp3 price not found')
    return price_gb_month / HOURS_IN_MONTH


# ──────────────────────────────────────────────────────────────
# OpenStack → AWS mapping
# ──────────────────────────────────────────────────────────────

def _choose_shape(vcpu: int, ram_mb: int, od: Dict[str, float], spec: Dict[str, Tuple[int, int]]):
    ram_gib = (ram_mb + 1023) // 1024
    fits = [(t, p) for t, p in od.items() if spec[t][0] >= vcpu and spec[t][1] >= ram_gib]
    return min(fits, key=lambda x: x[1]) if fits else (None, None)


# ──────────────────────────────────────────────────────────────
# Build report
# ──────────────────────────────────────────────────────────────

def build_report(conn, project_id: Optional[str], price_csv: Path):
    price_df = _load_csv(price_csv)
    od_price, ri_price, spec_map = _instance_and_spec(price_df)
    gp3_hr = _gp3_price_hr(price_df)

    # flavour cache
    flav = {f.id: f for f in conn.compute.flavors()}
    flav.update({f.name: f for f in flav.values()})

    rows: List[Dict[str, object]] = []
    seen_vol: Dict[str, int] = {}

    servers = list(conn.compute.servers(details=True, all_projects=True))
    with tqdm(total=len(servers), desc='VMs', unit='vm') as bar:
        for s in servers:
            bar.update(1)
            if project_id and s.project_id != project_id:
                continue

            f_ref = s.flavor.get('id') or s.flavor.get('original_name')
            f = flav.get(f_ref) or conn.compute.find_flavor(f_ref, ignore_missing=True)
            if not f:
                print(f'[WARN] flavor {f_ref} not found', file=sys.stderr)
                continue

            vcpus, ram_mb, root = f.vcpus, f.ram, f.disk
            disk = root
            for att in conn.compute.volume_attachments(server=s):
                if att.id not in seen_vol:
                    v = conn.block_storage.get_volume(att.id)
                    seen_vol[att.id] = v.size
                disk += seen_vol[att.id]

            itype, inst_od = _choose_shape(vcpus, ram_mb, od_price, spec_map)
            inst_ri = ri_price.get(itype) if itype else None
            storage_hr = disk * gp3_hr

            rows.append({
                'project': s.project_id,
                'server': s.name,
                'vcpus': vcpus,
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
    ap = argparse.ArgumentParser(description='OpenStack→AWS cost comparison')
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
        p = conn.identity.find_project(args.project, ignore_missing=True)
        pid = p.id if p else args.project

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
