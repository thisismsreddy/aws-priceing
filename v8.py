#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web-based OpenStack → AWS pricing comparison
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, request, render_template_string, Response, abort
from openstack import connection

# Constants
HOURS_IN_MONTH = 730
HOURS_IN_YEAR = 8_760
_SPACE_RE = re.compile(r"\s+")
CSV_PATH = Path('/home/srini/aws-priceing/AmazonEC2-pricing-20250505.csv')  # <-- set your CSV path here
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
    """Normalize column names and apply common aliases."""
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

    # Build spec map
    spec: Dict[str, Tuple[int, int]] = {}
    for itype, grp in base.groupby('Instance Type'):
        first = grp.iloc[0]
        mem_gib = int(float(str(first['Memory']).split()[0]))
        spec[itype] = (int(first['Vcpu']), mem_gib)

    # OnDemand pricing
    od_rows = base[
        (base['Term Type'] == 'OnDemand') &
        base['Price Description'].str.contains('per On Demand', case=False, na=False)
    ]
    od = od_rows.groupby('Instance Type')['Price Per Unit'].min().to_dict()

    # Reserved 3yr No Upfront
    ri_rows = base[
        (base['Term Type'] == 'Reserved') &
        (base['Lease Contract Length'].str.startswith('3')) &
        (base['Purchase Option'] == 'No Upfront') &
        (~base['Price Description'].str.contains('Upfront Fee', case=False, na=False))
    ]
    ri = ri_rows.groupby('Instance Type')['Price Per Unit'].min().to_dict()

    # gp3 storage $/hr/GB
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
    candidates = [
        (t, p) for t, p in prices.items()
        if spec.get(t, (0, 0))[0] >= vcpu and spec.get(t, (0, 0))[1] >= ram_gib
    ]
    return min(candidates, key=lambda x: x[1]) if candidates else (None, None)

# --- Report builder (with prefetches for speed) ---
def build_report(conn, project_id: Optional[str], csv_path: Path) -> pd.DataFrame:
    # 1) Prefetch all volumes (and optionally filter by project)
    all_vols = list(conn.block_storage.volumes(details=True, all_projects=True))
    if project_id:
        all_vols = [v for v in all_vols if getattr(v, 'project_id', None) == project_id]
    vol_size: Dict[str, int] = {v.id: v.size for v in all_vols}

    # 2) Build server→volumes map from attachments metadata
    vol_attach_map: Dict[str, List[str]] = {}
    for v in all_vols:
        for att in getattr(v, 'attachments', []) or []:
            sid = att.get('server_id') or att.get('server-id')
            if sid:
                vol_attach_map.setdefault(sid, []).append(v.id)

    # 3) Prefetch all flavors
    fls = list(conn.compute.flavors())
    flavor_map: Dict[str, object] = {}
    for fl in fls:
        flavor_map[fl.id] = fl
        flavor_map[fl.name] = fl

    # 4) Extract AWS pricing
    od, ri, spec, gp3 = _extract_pricing(_load_csv(csv_path))

    # 5) Iterate servers (single API call)
    rows: List[Dict[str, object]] = []
    for srv in conn.compute.servers(details=True, all_projects=True):
        if project_id and srv.project_id != project_id:
            continue

        # lookup flavor locally
        f_ref = srv.flavor.get('id') or srv.flavor.get('original_name')
        flav = flavor_map.get(f_ref)
        if not flav:
            continue

        # compute total disk: root + attached
        disk = getattr(flav, 'disk', 0) or 0
        for vid in vol_attach_map.get(srv.id, []):
            disk += vol_size.get(vid, 0)

        # pick AWS instance and compute costs
        itype, od_hr = _pick_shape(flav.vcpus, flav.ram, od, spec)
        ri_hr = ri.get(itype, 0)
        storage_hr = disk * gp3

        rows.append({
            'Project':    srv.project_id,
            'Server':     srv.name,
            'vCPU':       flav.vcpus,
            'RAM_GiB':    (flav.ram + 1023) // 1024,
            'Disk_GB':    disk,
            'AWS_Type':   itype or 'N/A',
            'OnDemand_Hourly':   (od_hr or 0) + storage_hr,
            'RI3yr_Hourly':      ri_hr + storage_hr,
            'OnDemand_Monthly':  ((od_hr or 0) + storage_hr) * HOURS_IN_MONTH,
            'OnDemand_Yearly':   ((od_hr or 0) + storage_hr) * HOURS_IN_YEAR,
            'RI3yr_Monthly':     (ri_hr + storage_hr) * HOURS_IN_MONTH,
            'RI3yr_Yearly':      (ri_hr + storage_hr) * HOURS_IN_YEAR,
        })

    df = pd.DataFrame(rows, columns=[
        'Project','Server','vCPU','RAM_GiB','Disk_GB',
        'AWS_Type','OnDemand_Hourly','RI3yr_Hourly',
        'OnDemand_Monthly','OnDemand_Yearly',
        'RI3yr_Monthly','RI3yr_Yearly'
    ])

    # append totals row
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
  #loading { display: none; position: fixed; top:0; left:0; width:100%; height:100%; background: rgba(255,255,255,0.8); z-index:1050; }
</style>
'''

FORM_HTML = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AWS Pricing Comparison</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
  ''' + BASE_CSS + '''
</head><body>
  <div id="loading">
    <div class="d-flex align-items-center justify-content-center h-100">
      <img src="{{ url_for('static','loading.gif') }}" alt="Loading..." />
    </div>
  </div>
  <nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
    <div class="container-fluid"><a class="navbar-brand" href="#">Pricing Tool</a></div>
  </nav>
  <div class="container">
    <div class="card"><div class="card-body">
      {% if error %}<div class="alert alert-danger">{{ error }}</div>{% endif %}
      <h5 class="card-title">AWS Pricing Comparison</h5>
      <div class="alert alert-info">
        Note: we are only considering EBS volumes here. Newer clouds
        (dc1-cdl-osp4, dc1-osp5, dc2-osp5) use NVMe storage, so take these
        results with a grain of salt.
      </div>
      <form method="post" onsubmit="
         document.getElementById('submit-btn').disabled=true;
         document.getElementById('loading').style.display='block';">
        <div class="row mb-3">
          <div class="col-md-6">
            <label for="cloud" class="form-label">Cloud Name</label>
            <input type="text" class="form-control" id="cloud" name="cloud"
                   value="{{ default_cloud }}" required>
          </div>
          <div class="col-md-6 form-check d-flex align-items-end">
            <input type="checkbox" class="form-check-input me-2"
                   id="all_projects" name="all_projects">
            <label class="form-check-label" for="all_projects">All Projects</label>
          </div>
        </div>
        <div class="mb-3">
          <label for="project" class="form-label">Project Name or ID</label>
          <input type="text" class="form-control" id="project" name="project">
        </div>
        <button type="submit" id="submit-btn" class="btn btn-primary">Run</button>
      </form>
    </div></div>
  </div>
</body></html>'''

RESULT_HTML = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Results</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css">
  ''' + BASE_CSS + '''
</head><body>
  <nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
    <div class="container-fluid"><a class="navbar-brand" href="/">Pricing Tool</a></div>
  </nav>
  <div class="container">
    <div class="card mb-4"><div class="card-body">
      <h5 class="card-title">Quick Summary</h5>
      <ul class="list-inline mb-0">
        <li class="list-inline-item"><strong>Servers:</strong> {{ summary.servers }}</li>
        <li class="list-inline-item"><strong>Total vCPU:</strong> {{ summary.vcpu }}</li>
        <li class="list-inline-item"><strong>Total RAM (GiB):</strong> {{ summary.ram }}</li>
        <li class="list-inline-item"><strong>Total Disk (GB):</strong> {{ summary.disk }}</li>
        <li class="list-inline-item"><strong>Monthly Cost:</strong> ${{ summary.monthly_cost | round(2) }}</li>
      </ul>
      <div class="mt-2 text-end">
        <a href="{{ url_for('download',
                            cloud=cloud,
                            project=project,
                            all_projects=('true' if all_projects else 'false')) }}"
           class="btn btn-success btn-sm">Download CSV</a>
      </div>
    </div></div>

    <div class="card"><div class="card-body">
      {% if df.empty %}
        <div class="alert alert-warning">No VMs found.</div>
      {% else %}
        <div class="table-responsive">
          {{ df.to_html(
               classes='table table-striped table-hover table-bordered table-sm',
               table_id='results',
               index=False,
               float_format='${:,.2f}'.format
          ) | safe }}
        </div>
      {% endif %}
      <a href="/" class="btn btn-secondary mt-3">Back</a>
    </div></div>
  </div>

  <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>
  <script>
    $(document).ready(function() {
      $('#results').DataTable({
        paging: true,
        searching: true,
        info: true,
        order: [[0, 'asc']],
        lengthChange: false,
        pageLength: 10
      });
    });
  </script>
</body></html>'''

@app.route('/', methods=['GET', 'POST'])
def index():
    error: Optional[str] = None
    if request.method == 'POST':
        cloud       = request.form.get('cloud', DEFAULT_CLOUD)
        raw_proj    = request.form.get('project', '').strip()
        all_projects= bool(request.form.get('all_projects'))

        # validate cloud
        try:
            conn = connection.from_config(cloud=cloud)
        except Exception:
            error = f"Cloud '{cloud}' not found."
            return render_template_string(FORM_HTML,
                                          default_cloud=DEFAULT_CLOUD,
                                          error=error)

        # validate project
        if not all_projects and raw_proj:
            pr = conn.identity.find_project(raw_proj, ignore_missing=True)
            if not pr:
                error = f"Project '{raw_proj}' not found."
                return render_template_string(FORM_HTML,
                                              default_cloud=DEFAULT_CLOUD,
                                              error=error)
            project_id = pr.id
        else:
            project_id = None

        # build full report
        df = build_report(conn, project_id, CSV_PATH)

        # summary metrics
        summary = {
            'servers':      len(df),
            'vcpu':         int(df['vCPU'].sum()),
            'ram':          int(df['RAM_GiB'].sum()),
            'disk':         int(df['Disk_GB'].sum()),
            'monthly_cost': float(df['OnDemand_Monthly'].sum()),
        }

        return render_template_string(
            RESULT_HTML,
            df=df,
            summary=summary,
            cloud=cloud,
            project=raw_proj,
            all_projects=all_projects
        )

    return render_template_string(FORM_HTML,
                                  default_cloud=DEFAULT_CLOUD,
                                  error=error)

@app.route('/download')
def download():
    cloud       = request.args.get('cloud', DEFAULT_CLOUD)
    raw_proj    = request.args.get('project', '').strip()
    all_projects= request.args.get('all_projects') == 'true'

    # same validation as index()
    try:
        conn = connection.from_config(cloud=cloud)
    except Exception:
        abort(404, f"Cloud '{cloud}' not found")

    if not all_projects and raw_proj:
        pr = conn.identity.find_project(raw_proj, ignore_missing=True)
        if not pr:
            abort(404, f"Project '{raw_proj}' not found")
        project_id = pr.id
    else:
        project_id = None

    df = build_report(conn, project_id, CSV_PATH)
    csv_str = df.to_csv(index=False)

    return Response(
        csv_str,
        mimetype='text/csv',
        headers={'Content-Disposition':'attachment;filename=pricing-report.csv'}
    )

if __name__ == '__main__':
    app.run(debug=True)
