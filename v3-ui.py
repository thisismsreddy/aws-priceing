#!/usr/bin/env python3
"""app.py – self‑contained Flask UI (fixed Jinja inheritance)

* Uses embedded CSS/JS (no CDN).
* AWS CSV path hard‑coded via AWS_CSV_DEFAULT.
* Renders pages by injecting `body` into a base template – avoids TemplateNotFound.
"""
from __future__ import annotations
import re, json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd
from flask import Flask, request, render_template_string
from openstack import connection

AWS_CSV_DEFAULT = Path("./prices/AmazonEC2-pricing-20250505.csv")
HOURS_IN_MONTH = 730
HOURS_IN_YEAR = 8_760
SPACE_RE = re.compile(r"\s+")
app = Flask(__name__)
app.secret_key = "stack‑aws‑ui"

# ─────────────────────────── CSV helpers ───────────────

def _load_csv(path: Path) -> pd.DataFrame:
    with path.open() as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith(("SKU", "\"SKU\"")):
                header = i
                break
        else:
            raise ValueError("CSV header not found")
    return pd.read_csv(path, skiprows=header, quotechar='"', dtype=str)

def _norm(df: pd.DataFrame) -> None:
    df.columns = (SPACE_RE.sub(" ", c.strip()).title() for c in df.columns)
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

# ───────────────────── pricing extraction ───────────────

def _extract_pricing(df: pd.DataFrame):
    _norm(df)
    df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')
    base = df[(df['Product Family'].str.lower() == 'compute instance') &
              (df['Region Code'].str.lower() == 'us-east-1') &
              (df['Unit'] == 'Hrs') & (df['Currency'] == 'USD') &
              (df['Tenancy'].fillna('Shared') == 'Shared') &
              (df['Operating System'].fillna('Linux') == 'Linux') &
              (df['Current Generation'].fillna('Yes') == 'Yes') &
              (df['Price Per Unit'] > 0)]
    spec = {t: (int(g.iloc[0]['Vcpu']), int(float(str(g.iloc[0]['Memory']).split()[0]))) for t, g in base.groupby('Instance Type')}
    od = base[(base['Term Type'] == 'OnDemand') &
              (base.get('Price Description', '').str.contains('per On Demand', case=False, na=False))]
    od = od.groupby('Instance Type')['Price Per Unit'].min().to_dict()
    ri = base[(base['Term Type'] == 'Reserved') &
              (base['Lease Contract Length'].str.startswith('3')) &
              (base['Purchase Option'] == 'No Upfront') &
              (~base.get('Price Description', '').str.contains('Upfront Fee', case=False, na=False))]
    ri = ri.groupby('Instance Type')['Price Per Unit'].min().to_dict()
    gp3 = df[(df['Product Family'].str.lower() == 'storage') &
             (df['Volume Api Name'] == 'gp3') &
             (df['Region Code'].str.lower() == 'us-east-1') &
             (df['Unit'].str.contains('GB')) &
             (df['Term Type'] == 'OnDemand') &
             (df['Price Per Unit'] > 0)]['Price Per Unit'].astype(float).min() / HOURS_IN_MONTH
    return od, ri, spec, gp3

# ───────────────────────── instance picker ───────────────

def _pick(vcpu: int, ram_mb: int, prices: Dict[str, float], spec: Dict[str, Tuple[int, int]]):
    ram = (ram_mb + 1023) // 1024
    fits = [(t, p) for t, p in prices.items() if spec[t][0] >= vcpu and spec[t][1] >= ram]
    return min(fits, key=lambda x: x[1]) if fits else (None, None)

# ─────────────────────── report builder ─────────────────

def build_report(cloud: str, project: Optional[str]):
    conn = connection.from_config(cloud=cloud)
    pid = None if project in (None, 'ALL') else (conn.identity.find_project(project, ignore_missing=True) or type('o',(object,),{'id':project})).id
    od, ri, spec, gp3 = _extract_pricing(_load_csv(AWS_CSV_DEFAULT))
    flavors = {f.id: f for f in conn.compute.flavors()}
    flavors.update({f.name: f for f in flavors.values()})
    seen: Dict[str, int] = {}
    rows: List[Dict[str, object]] = []
    for s in conn.compute.servers(details=True, all_projects=True):
        if pid and s.project_id != pid:
            continue
        flv = flavors.get(s.flavor.get('id') or s.flavor.get('original_name')) or conn.compute.find_flavor(s.flavor.get('id'), ignore_missing=True)
        if not flv:
            continue
        vcpu, ram, root = flv.vcpus, flv.ram, flv.disk
        disk = root
        for a in conn.compute.volume_attachments(server=s):
            if a.id not in seen:
                seen[a.id] = conn.block_storage.get_volume(a.id).size
            disk += seen[a.id]
        itype, od_hr = _pick(vcpu, ram, od, spec)
        ri_hr = ri.get(itype) if itype else None
        storage = disk * gp3
        rows.append({
            'Project': s.project_id,
            'Server': s.name,
            'vCPU': vcpu,
            'RAM_GiB': (ram + 1023)//1024,
            'Disk_GB': disk,
            'AWS_Type': itype or 'N/A',
            'OnDemand_Hourly': None if od_hr is None else od_hr + storage,
            'RI3yr_Hourly': None if ri_hr is None else ri_hr + storage,
        })
    df = pd.DataFrame(rows)
    for c in ['OnDemand_Hourly', 'RI3yr_Hourly']:
        df[c.replace('Hourly', 'Monthly')] = df[c] * HOURS_IN_MONTH
        df[c.replace('Hourly', 'Yearly')] = df[c] * HOURS_IN_YEAR
    num_cols = df.select_dtypes('number').columns
    total = {c: df[c].sum() if c in num_cols else ('TOTAL' if c == 'Server' else '') for c in df.columns}
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)

# ───────────────────── HTML templates ───────────────────
CSS = """
body{font-family:system-ui;background:#0d1117;color:#c9d1d9;padding-top:60px}
.navbar{background:#161b22!important}
.table{color:#c9d1d9}
th{cursor:pointer}
"""
JS_SORT = """
function sortTable(tbl,col){const dir=tbl.dataset.sortDir==='asc'?'desc':'asc';tbl.dataset.sortDir=dir;const val=v=>parseFloat(v)||v;[...tbl.tBodies[0].rows].sort((a,b)=>{const A=val(a.cells[col].innerText),B=val(b.cells[col].innerText);return dir==='asc'?(A>B?1:-1):(A<B?1:-1)}).forEach(r=>tbl.tBodies[0].appendChild(r));}
"""
BASE = """
<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<style>{{ css }}</style><script>{{ js }}</script><title>Cost Tool</title></head>
<body><nav class='navbar fixed-top'><div class='container-fluid'><span class='navbar-brand mb-0 h4'>OpenStack ➜ AWS Pricing</span></div></nav>
<div class='container-fluid p-4'>{{ body|safe }}</div></body></html>"""

FORM_BODY = """
<h1 class='mb-4'>OpenStack → AWS Pricing</h1>
<form method='post' action='/report' class='row gy-2'>
  <div class='col-md-4'><label class='form-label'>Cloud profile</label><input class='form-control' name='cloud' value='openstack' required></div>
  <div class='col-md-4'><label class='form-label'>Project (or ALL)</label><input class
