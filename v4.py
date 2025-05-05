#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web-based OpenStack → AWS pricing comparison
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, request, render_template_string
from openstack import connection

# Constants
HOURS_IN_MONTH = 730
HOURS_IN_YEAR = 8_760
_SPACE_RE = re.compile(r"\s+")
CSV_PATH = Path('/path/to/aws_price_list.csv')  # <-- set your CSV path here
DEFAULT_CLOUD = 'openstack'  # default cloud name from clouds.yaml

# --- CSV helpers ---
def _load_csv(path: Path) -> pd.DataFrame:
    """Load AWS price CSV starting at header row."""
    with path.open() as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith(("SKU", '"SKU"')):
                header = i
                break
        else:
            raise ValueError("Header row not found in price file")
    return pd.read_csv(path, skiprows=header, quotechar='"', dtype=str)

def _norm_cols(df: pd.DataFrame) -> None:
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

# --- Pricing extraction ---
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

    # gp3 storage
    _norm_cols(df)
    gp3_row = df[(df['Product Family'].str.lower() == 'storage') &
                 (df['Volume Api Name'] == 'gp3') &
                 (df['Region Code'].str.lower() == 'us-east-1') &
                 (df['Unit'].str.contains('GB')) &
                 (df['Term Type'] == 'OnDemand') &
                 (df['Price Per Unit'] > 0)]
    gp3_hr = pd.to_numeric(gp3_row['Price Per Unit']).min() / HOURS_IN_MONTH

    return od, ri, spec, gp3_hr

# --- Instance chooser ---
def _pick_shape(vcpu: int, ram_mb: int, prices: Dict[str, float], spec: Dict[str, Tuple[int, int]]):
    ram_gib = (ram_mb + 1023) // 1024
    choices = [(t, p) for t, p in prices.items() if spec[t][0] >= vcpu and spec[t][1] >= ram_gib]
    return min(choices, key=lambda x: x[1]) if choices else (None, None)

# --- Report builder ---
def build_report(conn, project_id: Optional[str], csv_path: Path) -> pd.DataFrame:
    od, ri, spec, gp3 = _extract_pricing(_load_csv(csv_path))
    # ensure columns even if no rows
    columns = ['Project','Server','vCPU','RAM_GiB','Disk_GB','AWS_Type','OnDemand_Hourly','RI3yr_Hourly']
    rows: List[Dict[str, object]] = []
    seen_vol: Dict[str, int] = {}
    flavors = {f.id: f for f in conn.compute.flavors()}
    flavors.update({f.name: f for f in flavors.values()})
    servers = list(conn.compute.servers(details=True, all_projects=True))
    for srv in servers:
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
        itype, od_hr = _pick_shape(vcpus, ram_mb, od, spec)
        ri_hr = ri.get(itype) if itype else None
        storage_hr = disk * gp3
        rows.append({
            'Project': srv.project_id,
            'Server': srv.name,
            'vCPU': vcpus,
            'RAM_GiB': (ram_mb + 1023)//1024,
            'Disk_GB': disk,
            'AWS_Type': itype or 'N/A',
            'OnDemand_Hourly': None if od_hr is None else od_hr + storage_hr,
            'RI3yr_Hourly': None if ri_hr is None else ri_hr + storage_hr,
        })
    # build dataframe with fixed columns
    df = pd.DataFrame(rows, columns=columns)
    # compute monthly/yearly
    if 'OnDemand_Hourly' in df.columns:
        df['OnDemand_Monthly'] = df['OnDemand_Hourly'].fillna(0) * HOURS_IN_MONTH
        df['OnDemand_Yearly'] = df['OnDemand_Hourly'].fillna(0) * HOURS_IN_YEAR
    if 'RI3yr_Hourly' in df.columns:
        df['RI3yr_Monthly'] = df['RI3yr_Hourly'].fillna(0) * HOURS_IN_MONTH
        df['RI3yr_Yearly'] = df['RI3yr_Hourly'].fillna(0) * HOURS_IN_YEAR
    # add total row
    total = {}
    for c in df.columns:
        if df[c].dtype.kind in ('i','f'):
            total[c] = df[c].sum()
        elif c == 'Server':
            total[c] = 'TOTAL'
        else:
            total[c] = ''
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)([df, pd.DataFrame([total])], ignore_index=True)

# --- Flask web app ---
app = Flask(__name__)

FORM_HTML = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AWS Pricing Comparison</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="p-4">
  <div class="container">
    <h1>AWS Pricing Comparison</h1>
    <form method="post">
      <div class="mb-3">
        <label for="cloud" class="form-label">Cloud Name</label>
        <input type="text" class="form-control" id="cloud" name="cloud" value="{{ default_cloud }}" required>
      </div>
      <div class="mb-3 form-check">
        <input type="checkbox" class="form-check-input" id="all_projects" name="all_projects">
        <label class="form-check-label" for="all_projects">All Projects</label>
      </div>
      <div class="mb-3">
        <label for="project" class="form-label">Project ID (if not all)</label>
        <input type="text" class="form-control" id="project" name="project">
      </div>
      <button type="submit" class="btn btn-primary">Run</button>
    </form>
  </div>
</body>
</html>'''

RESULT_HTML = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Results</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="p-4">
  <div class="container">
    <h1>Comparison Results</h1>
    <div>{{ table|safe }}</div>
    <a href="/" class="btn btn-secondary mt-3">Back</a>
  </div>
</body>
</html>'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        cloud = request.form.get('cloud', DEFAULT_CLOUD)
        proj = None if request.form.get('all_projects') else request.form.get('project')
        conn = connection.from_config(cloud=cloud)
        df = build_report(conn, proj, CSV_PATH)
        table = df.to_html(classes='table table-striped table-bordered', index=False, float_format='{:,.4f}'.format)
        return render_template_string(RESULT_HTML, table=table)
    return render_template_string(FORM_HTML, default_cloud=DEFAULT_CLOUD)

if __name__ == '__main__':
    app.run(debug=True)
