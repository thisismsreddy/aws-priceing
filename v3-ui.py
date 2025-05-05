#!/usr/bin/env python3
"""app.py — Single‑file Flask UI (self‑contained assets)

* CSV path is configured via AWS_CSV_DEFAULT at top of file.
* No external CDN — Bootstrap‑like styles are embedded in <style>.
* Table uses a lightweight vanilla‑JS sorter (tiny function).
"""
from __future__ import annotations

import re, sys, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from flask import Flask, render_template_string, request
from openstack import connection

AWS_CSV_DEFAULT = Path("./prices/AmazonEC2-pricing-20250505.csv")
HOURS_IN_MONTH = 730
HOURS_IN_YEAR = 8_760
_SPACE = re.compile(r"\s+")
app = Flask(__name__)
app.secret_key = 'stack‑aws‑ui'

# ── CSV helpers ─────────────────────────────────────────────

def _load_csv(p: Path):
    with p.open() as f:
        for i,l in enumerate(f):
            if l.lstrip().startswith(('SKU','"SKU"')):
                hdr=i;break
        else:
            raise ValueError('header not found')
    return pd.read_csv(p, skiprows=hdr, quotechar='"', dtype=str)

def _norm(df):
    df.columns = (_SPACE.sub(' ', c.strip()).title() for c in df.columns)
    alias={'Instancetype':'Instance Type','Vpcu':'Vcpu','Priceperunit':'Price Per Unit','Pricedescription':'Price Description','Termtype':'Term Type','Leasecontractlength':'Lease Contract Length','Purchaseoption':'Purchase Option'}
    df.rename(columns={k:v for k,v in alias.items() if k in df.columns}, inplace=True)

# ── pricing extraction ─────────────────────────────────────

def _pricing(df: pd.DataFrame):
    _norm(df)
    df['Price Per Unit']=pd.to_numeric(df['Price Per Unit'],errors='coerce')
    base=df[(df['Product Family'].str.lower()=='compute instance')&(df['Region Code'].str.lower()=='us-east-1')&(df['Unit']=='Hrs')&(df['Currency']=='USD')&(df['Tenancy'].fillna('Shared')=='Shared')&(df['Operating System'].fillna('Linux')=='Linux')&(df['Current Generation'].fillna('Yes')=='Yes')&(df['Price Per Unit']>0)]
    spec={t:(int(g.iloc[0]['Vcpu']),int(float(str(g.iloc[0]['Memory']).split()[0]))) for t,g in base.groupby('Instance Type')}
    od=base[(base['Term Type']=='OnDemand')&(base.get('Price Description','').str.contains('per On Demand',case=False,na=False))].groupby('Instance Type')['Price Per Unit'].min().to_dict()
    ri=base[(base['Term Type']=='Reserved')&(base['Lease Contract Length'].str.startswith('3'))&(base['Purchase Option']=='No Upfront')&(~base.get('Price Description','').str.contains('Upfront Fee',case=False,na=False))].groupby('Instance Type')['Price Per Unit'].min().to_dict()
    gp3=df[(df['Product Family'].str.lower()=='storage')&(df['Volume Api Name']=='gp3')&(df['Region Code'].str.lower()=='us-east-1')&(df['Unit'].str.contains('GB'))&(df['Term Type']=='OnDemand')&(df['Price Per Unit']>0)]['Price Per Unit'].astype(float).min()/HOURS_IN_MONTH
    return od,ri,spec,gp3

# ── chooser ────────────────────────────────────────────────

def _pick(v: int, ram_mb: int, prices: Dict[str,float], spec):
    ram=(ram_mb+1023)//1024
    opts=[(t,p) for t,p in prices.items() if spec[t][0]>=v and spec[t][1]>=ram]
    return min(opts,key=lambda x:x[1]) if opts else (None,None)

# ── report builder ─────────────────────────────────────────

def build_report(cloud:str, project:str|None):
    conn=connection.from_config(cloud=cloud)
    pid=None if project in (None,'ALL') else (conn.identity.find_project(project,ignore_missing=True) or type('obj',(object,),{'id':project})).id
    od,ri,spec,gp3=_pricing(_load_csv(AWS_CSV_DEFAULT))
    flav={f.id:f for f in conn.compute.flavors()};flav.update({f.name:f for f in flav.values()})
    seen:Dict[str,int]={}
    rows=[]
    for s in conn.compute.servers(details=True,all_projects=True):
        if pid and s.project_id!=pid:continue
        f=flav.get(s.flavor.get('id') or s.flavor.get('original_name')) or conn.compute.find_flavor(s.flavor.get('id'),ignore_missing=True)
        if not f:continue
        vcpu,ram,root=f.vcpus,f.ram,f.disk
        disk=root
        for a in conn.compute.volume_attachments(server=s):
            if a.id not in seen:seen[a.id]=conn.block_storage.get_volume(a.id).size
            disk+=seen[a.id]
        itype,od_hr=_pick(vcpu,ram,od,spec)
        ri_hr=ri.get(itype) if itype else None
        stor=disk*gp3
        rows.append({'Project':s.project_id,'Server':s.name,'vCPU':vcpu,'RAM_GiB':(ram+1023)//1024,'Disk_GB':disk,'AWS_Type':itype or 'N/A','OnDemand_Hourly':None if od_hr is None else od_hr+stor,'RI3yr_Hourly':None if ri_hr is None else ri_hr+stor})
    df=pd.DataFrame(rows)
    for c in ['OnDemand_Hourly','RI3yr_Hourly']:
        df[c.replace('Hourly','Monthly')]=df[c]*HOURS_IN_MONTH
        df[c.replace('Hourly','Yearly')]=df[c]*HOURS_IN_YEAR
    nums=df.select_dtypes('number').columns
    total={c:df[c].sum() if c in nums else ('TOTAL' if c=='Server' else '') for c in df.columns}
    return pd.concat([df,pd.DataFrame([total])],ignore_index=True)

# ── HTML templates ─────────────────────────────────────────
CSS="""
body{font-family:system-ui;background:#0d1117;color:#c9d1d9;padding-top:60px}
.navbar{background:#161b22!important}
.table{color:#c9d1d9}
th{cursor:pointer}
"""
SORT_JS="""
function sortTable(tbl,col){const dir=tbl.dataset.sortDir==='asc'?'desc':'asc';tbl.dataset.sortDir=dir;const ints=v=>parseFloat(v)||v;[...tbl.tBodies[0].rows].sort((a,b)=>{const A=ints(a.cells[col].innerText),B=ints(b.cells[col].innerText);return dir==='asc'?(A>B?1:-1):(A<B?1:-1)}).forEach(r=>tbl.tBodies[0].appendChild(r));}
"""

FORM_PAGE="""
{% extends 'base' %}{% block body %}
<h1 class="mb-4">OpenStack → AWS Pricing</h1>
<form method="post" action="{{ url_for('report') }}" class="row gy-2">
  <div class="col-md-4"><label class="form-label">Cloud profile</label><input class="form-control" name="cloud" value="openstack" required></div>
  <div class="col-md-4"><label class="form-label">Project (or ALL)</label><input class="form-control" name="project" value="ALL"></div>
  <div class="col-md-4 align-self-end"><button class="btn btn-primary w-100" type="submit">Generate</button></div>
</form>
{% endblock %}
"""

RESULT_PAGE="""
{% extends 'base' %}{% block body %}
<a href="{{ url_for('home') }}" class="btn btn-secondary mb-3">← New report</a>
<table id="tbl" class="table table-striped table-hover small">
<thead><tr>{% for c in cols %}<th onclick="sortTable(document.getElementById('tbl'),{{ loop.index0 }})">{{ c }}</th>{% endfor %}</tr></thead>
<tbody>{% for row in data %}<tr>{% for c in cols %}<td>{{ row[c] }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table>
{% endblock %}
"""

BASE_PAGE="""
{% block dummy %}{% endblock %}
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{{css}}</style><script>{{js}}</script><title>Cost Tool</title></head><body><nav class="navbar fixed-top"><div class="container-fluid"><span class="navbar-brand mb-0 h4">OpenStack ➜ AWS Pricing</span></div></nav><div class="container-fluid p-4">{% block body %}{% endblock %}</div></body></html>
"""

@app.route('/')
def home():
    return render_template_string(FORM_PAGE, css=CSS, js=SORT_JS, template_folder='.', globals={'base':BASE_PAGE})

@app.route('/report', methods=['POST'])
def report():
    cloud=request.form['cloud']
    project=request.form['project']
    df=build_report(cloud,project)
    cols=list(df.columns)
    data=json.loads(df.to_json(orient='records'))
    return render_template_string(RESULT_PAGE, css=CSS, js=SORT_JS, cols=cols, data=data, template_folder='.', globals={'base':BASE_PAGE})

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
