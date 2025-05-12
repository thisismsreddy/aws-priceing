#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web-based OpenStack → AWS pricing comparison with Caching
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time # For cache TTL

import pandas as pd
from flask import Flask, request, render_template_string, Response, abort
from openstack import connection

# --- Cache Configuration ---
_OPENSTACK_DATA_CACHE: Dict[Tuple[str, Optional[str]], Tuple[float, List[Dict]]] = {}
CACHE_TTL_SECONDS = 3600  # Cache OpenStack data for 1 hour (3600 seconds)

# --- Constants ---
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
    _norm_cols(df) # Re-normalize if needed, as we are operating on the original df
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

# --- Helper function to fetch and process OpenStack server data (for caching) ---
def _get_processed_openstack_servers(conn: connection.Connection, project_id_filter: Optional[str]) -> List[Dict]:
    """
    Fetches and processes server, flavor, and volume information from OpenStack.
    This is the data that will be cached.
    """
    print(f"Fetching OpenStack data for cloud connection, project_filter: {project_id_filter or 'ALL'}")
    # 1) Prefetch all volumes (and optionally filter by project_id_filter)
    # Note: conn.block_storage.volumes(all_projects=True) is important if project_id_filter is None
    # If project_id_filter is set, we filter *after* fetching all_projects=True for volumes
    # because volumes don't always have project_id in their direct list attributes for filtering.
    all_vols_from_api = list(conn.block_storage.volumes(details=True, all_projects=True))
    
    # Create a mapping of server_id to project_id from compute if needed for volume filtering later
    # This is a bit indirect but ensures we can associate volumes to projects if project_id_filter is active
    server_project_map = {}
    if project_id_filter: # Only build this map if we are filtering by a specific project
        servers_for_map = list(conn.compute.servers(details=True, project_id=project_id_filter))
        for srv_map in servers_for_map:
            server_project_map[srv_map.id] = srv_map.project_id

    filtered_vols: List[object] = []
    if project_id_filter:
        for v_check in all_vols_from_api:
            # A volume belongs to the project if it's directly associated OR if it's attached to a server in that project
            if getattr(v_check, 'os-vol-tenant-attr:tenant_id', None) == project_id_filter:
                 filtered_vols.append(v_check)
            elif getattr(v_check, 'project_id', None) == project_id_filter: # Newer SDK attribute
                 filtered_vols.append(v_check)
            else:
                for att in getattr(v_check, 'attachments', []) or []:
                    sid = att.get('server_id') or att.get('server-id')
                    if sid and server_project_map.get(sid) == project_id_filter:
                        filtered_vols.append(v_check)
                        break # Added to this project's list
        all_vols = filtered_vols
    else:
        all_vols = all_vols_from_api # Use all volumes if no project filter

    vol_size: Dict[str, int] = {v.id: v.size for v in all_vols}

    # 2) Build server→volumes map from attachments metadata for relevant volumes
    vol_attach_map: Dict[str, List[str]] = {}
    for v in all_vols: # Use the potentially filtered list of volumes
        for att in getattr(v, 'attachments', []) or []:
            sid = att.get('server_id') or att.get('server-id')
            if sid:
                vol_attach_map.setdefault(sid, []).append(v.id)

    # 3) Prefetch all flavors
    fls = list(conn.compute.flavors(details=True)) # details=True is good practice
    flavor_map: Dict[str, object] = {}
    for fl in fls:
        flavor_map[fl.id] = fl
        flavor_map[fl.name] = fl # Also map by name for robustness

    # 4) Iterate servers (single API call, filtered by project_id_filter if provided)
    processed_servers: List[Dict[str, object]] = []
    server_list_args = {'details': True, 'all_projects': project_id_filter is None}
    if project_id_filter:
        server_list_args['project_id'] = project_id_filter

    for srv in conn.compute.servers(**server_list_args):
        # Double check project_id if project_id_filter was applied during server fetch
        # This is mostly redundant if server_list_args['project_id'] worked, but good for safety.
        if project_id_filter and srv.project_id != project_id_filter:
            continue

        f_ref = srv.flavor.get('id') or srv.flavor.get('original_name')
        flav = flavor_map.get(f_ref)
        if not flav:
            print(f"Warning: Flavor '{f_ref}' for server '{srv.name}' ({srv.id}) not found. Skipping server.")
            continue

        disk = getattr(flav, 'disk', 0) or 0
        for vid in vol_attach_map.get(srv.id, []): # Uses vol_attach_map derived from (filtered) all_vols
            disk += vol_size.get(vid, 0)

        processed_servers.append({
            'os_project_id': srv.project_id,
            'os_server_name': srv.name,
            'os_vcpus': flav.vcpus,
            'os_ram_mb': flav.ram,
            'os_disk_gb': disk,
        })
    return processed_servers

# --- Report builder (with OpenStack data caching) ---
def build_report(conn: connection.Connection, cloud_name_for_cache: str, project_id_for_filtering: Optional[str], csv_path: Path) -> pd.DataFrame:
    # --- Cache Check for OpenStack Data ---
    # The project_id_for_filtering (which can be None for 'all projects') is part of the cache key.
    cache_key = (cloud_name_for_cache, project_id_for_filtering)
    current_time = time.time()
    openstack_servers_data: List[Dict]

    if cache_key in _OPENSTACK_DATA_CACHE:
        timestamp, cached_data = _OPENSTACK_DATA_CACHE[cache_key]
        if (current_time - timestamp) < CACHE_TTL_SECONDS:
            print(f"CACHE HIT for OpenStack data: {cache_key}")
            openstack_servers_data = cached_data
        else:
            print(f"CACHE EXPIRED for OpenStack data: {cache_key}")
            openstack_servers_data = _get_processed_openstack_servers(conn, project_id_for_filtering)
            _OPENSTACK_DATA_CACHE[cache_key] = (current_time, openstack_servers_data)
    else:
        print(f"CACHE MISS for OpenStack data: {cache_key}")
        openstack_servers_data = _get_processed_openstack_servers(conn, project_id_for_filtering)
        _OPENSTACK_DATA_CACHE[cache_key] = (current_time, openstack_servers_data)
    # --- End of Cache Logic ---

    # AWS pricing extraction (this part is not cached with OS data as it's from CSV)
    od, ri, spec, gp3 = _extract_pricing(_load_csv(csv_path))

    rows: List[Dict[str, object]] = []
    # Iterate over the (potentially cached) processed OpenStack server data
    for server_spec in openstack_servers_data:
        itype, od_hr = _pick_shape(server_spec['os_vcpus'], server_spec['os_ram_mb'], od, spec)
        ri_hr = ri.get(itype, 0.0) # Default to 0.0 if not found
        storage_hr = server_spec['os_disk_gb'] * gp3

        rows.append({
            'Project':    server_spec['os_project_id'],
            'Server':     server_spec['os_server_name'],
            'vCPU':       server_spec['os_vcpus'],
            'RAM_GiB':    (server_spec['os_ram_mb'] + 1023) // 1024,
            'Disk_GB':    server_spec['os_disk_gb'],
            'AWS_Type':   itype or 'N/A',
            'OnDemand_Hourly':   (od_hr or 0.0) + storage_hr,
            'RI3yr_Hourly':      ri_hr + storage_hr, # ri_hr is already a float
            'OnDemand_Monthly':  ((od_hr or 0.0) + storage_hr) * HOURS_IN_MONTH,
            'OnDemand_Yearly':   ((od_hr or 0.0) + storage_hr) * HOURS_IN_YEAR,
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
        # Ensure all cost columns are numeric before sum, handle potential NaNs from 'N/A' AWS_Type
        cost_columns = ['OnDemand_Hourly', 'RI3yr_Hourly', 'OnDemand_Monthly', 'OnDemand_Yearly', 'RI3yr_Monthly', 'RI3yr_Yearly']
        for col in cost_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        totals = {c: (df[c].sum() if df[c].dtype.kind in ('i','f') else '') for c in df.columns}
        totals['Server'] = 'TOTAL'
        # df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True) # Original
        df = pd.concat([df, pd.DataFrame([totals], columns=df.columns)], ignore_index=True) # Ensure columns match

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
      <img src="{{ url_for('static', filename='loading.gif') }}" alt="Loading..." />
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
        results with a grain of salt. (Cache TTL: ''' + str(CACHE_TTL_SECONDS // 60) + ''' minutes)
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
          <label for="project" class="form-label">Project Name or ID (leave blank for 'All Projects' if checked)</label>
          <input type="text" class="form-control" id="project" name="project">
        </div>
        <button type="submit" id="submit-btn" class="btn btn-primary">Run</button>
      </form>
    </div></div>
  </div>
  <script>
    // Optional: Disable project input if 'All Projects' is checked
    const allProjectsCheckbox = document.getElementById('all_projects');
    const projectInput = document.getElementById('project');
    if (allProjectsCheckbox && projectInput) {
      allProjectsCheckbox.addEventListener('change', function() {
        projectInput.disabled = this.checked;
        if (this.checked) {
          projectInput.value = ''; // Clear value if disabled
        }
      });
      // Initial state
      projectInput.disabled = allProjectsCheckbox.checked;
       if (allProjectsCheckbox.checked) {
          projectInput.value = '';
        }
    }
  </script>
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
      <h5 class="card-title">Quick Summary for {{ cloud }}{% if project %} (Project: {{ project }}){% elif all_projects %} (All Projects){% endif %}</h5>
      <ul class="list-inline mb-0">
        <li class="list-inline-item"><strong>Servers:</strong> {{ summary.servers }}</li>
        <li class="list-inline-item"><strong>Total vCPU:</strong> {{ summary.vcpu }}</li>
        <li class="list-inline-item"><strong>Total RAM (GiB):</strong> {{ summary.ram }}</li>
        <li class="list-inline-item"><strong>Total Disk (GB):</strong> {{ summary.disk }}</li>
        <li class="list-inline-item"><strong>Est. Monthly OnDemand Cost (AWS):</strong> ${{ summary.monthly_cost | round(2) }}</li>
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
        <div class="alert alert-warning">No VMs found matching criteria.</div>
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
        order: [[0, 'asc']], // Default sort by Project
        lengthChange: true, // Allow changing page length
        pageLength: 25 // Default page length
      });
    });
  </script>
</body></html>'''

@app.route('/', methods=['GET', 'POST'])
def index():
    error: Optional[str] = None
    # For GET requests, set default cloud for the template
    cloud_for_template = request.form.get('cloud', DEFAULT_CLOUD) if request.method == 'POST' else DEFAULT_CLOUD
    
    if request.method == 'POST':
        cloud_name = request.form.get('cloud', DEFAULT_CLOUD).strip()
        raw_proj_input = request.form.get('project', '').strip()
        fetch_all_projects = bool(request.form.get('all_projects'))

        # Update cloud_for_template for re-rendering form if error
        cloud_for_template = cloud_name

        # Validate cloud connection
        try:
            conn = connection.from_config(cloud=cloud_name)
            # Perform a simple API call to check connectivity
            conn.identity.get_project_details(conn.current_project_id)
        except Exception as e:
            print(f"Error connecting to cloud '{cloud_name}': {e}")
            error = f"Cloud '{cloud_name}' not found or connection error: {str(e)[:100]}"
            return render_template_string(FORM_HTML,
                                          default_cloud=cloud_for_template, # Use potentially entered cloud
                                          error=error)

        # Determine project_id for OpenStack SDK and cache key
        project_id_to_filter: Optional[str] = None
        project_name_for_display = raw_proj_input # For display on results page

        if not fetch_all_projects and raw_proj_input:
            pr = conn.identity.find_project(raw_proj_input, ignore_missing=False) # Fails if not found
            if not pr: # Should be caught by ignore_missing=False, but for safety
                error = f"Project '{raw_proj_input}' not found in cloud '{cloud_name}'."
                return render_template_string(FORM_HTML,
                                              default_cloud=cloud_for_template,
                                              error=error)
            project_id_to_filter = pr.id
        elif fetch_all_projects:
            project_name_for_display = "" # Clear project name if all_projects is selected
            # project_id_to_filter remains None, indicating all projects
        # If not fetch_all_projects and raw_proj_input is empty, project_id_to_filter also remains None
        # which effectively means all projects for the default project of the cloud connection.
        # To be more explicit, if not fetch_all_projects and raw_proj_input is empty, it might be better
        # to default to current project or show an error. For now, it implies "all projects accessible
        # by the connection's default scope" if not specific project ID given.
        # Let's refine: if not all_projects and no project is given, use current project context
        if not fetch_all_projects and not raw_proj_input:
             # Use current project if no specific project and not 'all projects'
            project_id_to_filter = conn.current_project_id
            try:
                proj_details = conn.identity.get_project(project_id_to_filter)
                project_name_for_display = proj_details.name
            except Exception:
                project_name_for_display = project_id_to_filter # fallback to ID

        # Build full report
        try:
            df = build_report(conn, cloud_name, project_id_to_filter, CSV_PATH)
        except Exception as e:
            print(f"Error building report for cloud '{cloud_name}', project '{project_id_to_filter}': {e}")
            error = f"Error generating report: {str(e)}. Check logs."
            return render_template_string(FORM_HTML,
                                          default_cloud=cloud_for_template,
                                          error=error)
        # Summary metrics
        summary = {
            'servers': 0, 'vcpu': 0, 'ram': 0, 'disk': 0, 'monthly_cost': 0.0,
        }
        if not df.empty and len(df) > 0: # Check if DataFrame has rows before trying to access df.iloc[-1]
            # Exclude TOTAL row for summary calculation if it exists
            summary_df = df.iloc[:-1] if df.iloc[-1]['Server'] == 'TOTAL' and len(df) > 1 else df
            
            if not summary_df.empty:
                summary['servers'] = len(summary_df)
                summary['vcpu']    = int(pd.to_numeric(summary_df['vCPU'], errors='coerce').sum())
                summary['ram']     = int(pd.to_numeric(summary_df['RAM_GiB'], errors='coerce').sum())
                summary['disk']    = int(pd.to_numeric(summary_df['Disk_GB'], errors='coerce').sum())
                summary['monthly_cost'] = float(pd.to_numeric(summary_df['OnDemand_Monthly'], errors='coerce').sum())

        return render_template_string(
            RESULT_HTML,
            df=df,
            summary=summary,
            cloud=cloud_name,
            project=project_name_for_display,
            all_projects=fetch_all_projects
        )

    return render_template_string(FORM_HTML,
                                  default_cloud=cloud_for_template,
                                  error=error)

@app.route('/download')
def download():
    cloud_name = request.args.get('cloud', DEFAULT_CLOUD).strip()
    raw_proj_input = request.args.get('project', '').strip()
    fetch_all_projects = request.args.get('all_projects') == 'true'

    try:
        conn = connection.from_config(cloud=cloud_name)
        conn.identity.get_project_details(conn.current_project_id) # Test connection
    except Exception as e:
        print(f"Error connecting to cloud '{cloud_name}' for download: {e}")
        abort(404, f"Cloud '{cloud_name}' not found or connection error.")

    project_id_to_filter: Optional[str] = None
    if not fetch_all_projects and raw_proj_input:
        pr = conn.identity.find_project(raw_proj_input, ignore_missing=True)
        if not pr:
            abort(404, f"Project '{raw_proj_input}' not found in cloud '{cloud_name}'.")
        project_id_to_filter = pr.id
    elif not fetch_all_projects and not raw_proj_input: # Default to current project
        project_id_to_filter = conn.current_project_id
    # If fetch_all_projects is true, project_id_to_filter remains None

    try:
        df = build_report(conn, cloud_name, project_id_to_filter, CSV_PATH)
    except Exception as e:
        print(f"Error building report for download: {e}")
        abort(500, "Error generating report for download.")

    csv_str = df.to_csv(index=False)

    return Response(
        csv_str,
        mimetype='text/csv',
        headers={'Content-Disposition':f'attachment;filename=pricing-report-{cloud_name}-{raw_proj_input or "all"}.csv'}
    )

# --- Static file for loading GIF (Create a 'static' folder next to your script) ---
# Example: Create a folder named 'static' and put a 'loading.gif' inside it.
# If you don't have one, you can comment out the loading GIF part in FORM_HTML.
# For simplicity, I'm not adding the actual GIF file handling here, but Flask serves
# from a 'static' directory by default.

if __name__ == '__main__':
    # For development:
    # You might need to create a 'static' folder in the same directory as this script
    # and place a 'loading.gif' image there for the loading indicator to work.
    # Example:
    # Path("static").mkdir(exist_ok=True)
    # with open("static/loading.gif", "wb") as f:
    #     # (Optional) You could download a small public domain gif here
    #     # import requests
    #     # r = requests.get("https://some-public-domain-gif-url/loading.gif")
    #     # f.write(r.content)
    #     f.write(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;') # Tiny placeholder GIF
    
    app.run(debug=True, host='0.0.0.0', port=5000)
