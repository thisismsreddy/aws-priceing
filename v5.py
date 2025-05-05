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
        (df['Unit'] == 'Hrs') &
        (df['Currency'] == 'USD') &
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
    od = base[(base['Term Type'] == 'OnDemand') & base['Price Description'].str.contains('per On Demand', case=False, na=False)]
    od = od.groupby('Instance Type')['Price Per Unit'].min().to_dict()
    ri = base[(base['Term Type'] == 'Reserved') &
              base['Lease Contract Length'].str.startswith('3') &
              (base['Purchase Option'] == 'No Upfront') &
              (~base['Price Description'].str.contains('Upfront Fee', case=False, na=False))]
    ri = ri.groupby('Instance Type')['Price Per Unit'].min().to_dict()
    _norm_cols(df)
    gp3_row = df[
        (df['Product Family'].str.lower() == 'storage') &
        (df['Volume Api Name'] == 'gp3') &
        (df['Region Code'].str.lower() == 'us-east-1') &
        (df['Unit'].str.contains('GB')) &
        (df['Term Type'] == 'OnDemand') &
        (df['Price Per Unit'] > 0)
    ]
    gp3_hr = pd.to_numeric(gp3_row['Price Per Unit']).min() / HOURS_IN_MONTH
    return od, ri, spec, gp3_hr

# --- Instance chooser ---
def _pick_shape(vcpu: int, ram_mb: int, prices: Dict[str, float], spec: Dict[str, Tuple[int, int]]):
    ram_gib = (ram_mb + 1023) // 1024
    candidates = [(t, p) for t, p in prices.items() if spec.get(t, (0, 0))[0] >= vcpu and spec.get(t, (0, 0))[1] >= ram_gib]
    return min(candidates, key=lambda x: x[1]) if candidates else (None, None)

# --- Report builder ---
def build_report(conn, project_id: Optional[str], csv_path: Path) -> pd.DataFrame:
    od, ri, spec, gp3 = _extract_pricing(_load_csv(csv_path))
    columns = [
        'Project', 'Server', 'vCPU', 'RAM_GiB', 'Disk_GB',
        'AWS_Type', 'OnDemand_Hourly', 'RI3yr_Hourly',
        'OnDemand_Monthly', 'OnDemand_Yearly',
        'RI3yr_Monthly', 'RI3yr_Yearly'
    ]
    rows: List[Dict[str, object]] = []
    seen_vol: Dict[str, int] = {}
    flavors = {f.id: f for f in conn.compute.flavors()}
    flavors.update({f.name: f for f in flavors.values()})
    for srv in conn.compute.servers(details=True, all_projects=True):
        if project_id and srv.project_id != project_id:
            continue
        f_ref = srv.flavor.get('id') or srv.flavor.get('original_name')
        flav = flavors.get(f_ref) or conn.compute.find_flavor(f_ref, ignore_missing=True)
        if not flav:
            continue
        disk = flav.disk
        for att in conn.compute.volume_attachments(server=srv):
            if att.id not in seen_vol:
                seen_vol[att.id] = conn.block_storage.get_volume(att.id).size
            disk += seen_vol[att.id]
        itype, od_hr = _pick_shape(flav.vcpus, flav.ram, od, spec)
        ri_hr = ri.get(itype, 0)
        storage_hr = disk * gp3
        rows.append({
            'Project': srv.project_id,
            'Server': srv.name,
            'vCPU': flav.vcpus,
            'RAM_GiB': (flav.ram + 1023) // 1024,
            'Disk_GB': disk,
            'AWS_Type': itype or 'N/A',
            'OnDemand_Hourly': (od_hr or 0) + storage_hr,
            'RI3yr_Hourly': ri_hr + storage_hr,
            'OnDemand_Monthly': ((od_hr or 0) + storage_hr) * HOURS_IN_MONTH,
            'OnDemand_Yearly': ((od_hr or 0) + storage_hr) * HOURS_IN_YEAR,
            'RI3yr_Monthly': (ri_hr + storage_hr) * HOURS_IN_MONTH,
            'RI3yr_Yearly': (ri_hr + storage_hr) * HOURS_IN_YEAR,
        })
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        totals = {c: (df[c].sum() if df[c].dtype.kind in ('i','f') else '') for c in df.columns}
        totals['Server'] = 'TOTAL'
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    return df

# --- Flask web app ---
app = Flask(__name__)

BASE_CSS = '''
<style>
  body { background-color: #f8f9fa; }
  .card { margin-top: 20px; }
  #loading {
    display: none;
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(255,255,255,0.8);
    z-index: 1050;
  }
</style>
'''

FORM_HTML = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AWS Pricing Comparison</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
  ''' + BASE_CSS + '''
</head>
<body>
  <div id="loading">
    <div class="d-flex align-items-center justify-content-center h-100">
      <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMmpwdDBzNzN6djV3cWM3NGtzMzE1ZGxjeDRuZWp2YWZpdHdsZmt5ZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/cGQihe7X3yehxr4BsX/giphy.gif" alt="Loading..." />
    </div>
  </div>
    </div>
  </div>
  <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
    <div class="container-fluid">
      <a class="navbar-brand" href="#">Pricing Tool</a>
    </div>
  </nav>
  <div class="container">
    <div class="card">
      <div class="card-body">
        <h5 class="card-title">AWS Pricing Comparison</h5>
        <form method="post" onsubmit="document.getElementById('submit-btn').disabled=true; document.getElementById('loading').style.display='block';">
          <div class="row mb-3">
            <div class="col-md-6">
              <label for="cloud" class="form-label">Cloud Name</label>
              <input type="text" class="form-control" id="cloud" name="cloud" value="{{ default_cloud }}" required>
            </div>
            <div class="col-md-6 form-check d-flex align-items-end">
              <input type="checkbox" class="form-check-input me-2" id="all_projects" name="all_projects">
              <label class="form-check-label" for="all_projects">All Projects</label>
            </div>
          </div>
          <div class="mb-3">
            <label for="project" class="form-label">Project Name or ID</label>
            <input type="text" class="form-control" id="project" name="project">
          </div>
          <button type="submit" id="submit-btn" class="btn btn-primary">Run</button>
        </form>
      </div>
    </div>
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
  ''' + BASE_CSS + '''
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
    <div class="container-fluid">
      <a class="navbar-brand" href="#">Pricing Tool</a>
    </div>
  </nav>
  <div class="container">
    <div class="card mt-4">
      <div class="card-body">
        <h5 class="card-title">Comparison Results</h5>
        {% if df.empty %}
          <div class="alert alert-warning">No VMs found for the given project/cloud.</div>
        {% else %}
          <div class="table-responsive">
            {{ df.to_html(classes='table table-hover table-striped', index=False, float_format='{:,.4f}'.format) | safe }}
          </div>
        {% endif %}
        <a href="/" class="btn btn-secondary mt-3">Back</a>
      </div>
    </div>
  </div>
</body>
</html>'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        cloud = request.form.get('cloud', DEFAULT_CLOUD)
        all_projects = True if request.form.get('all_projects') else False
        raw_proj = request.form.get('project', '').strip()
        conn = connection.from_config(cloud=cloud)
        if all_projects:
            project_id = None
        else:
            if raw_proj:
                pr = conn.identity.find_project(raw_proj, ignore_missing=True)
                project_id = pr.id if pr else raw_proj
            else:
                project_id = None
        df = build_report(conn, project_id, CSV_PATH)
        return render_template_string(RESULT_HTML, default_cloud=DEFAULT_CLOUD, df=df)
    return render_template_string(FORM_HTML, default_cloud=DEFAULT_CLOUD)

if __name__ == '__main__':
    app.run(debug=True)
