#!/usr/bin/env python3
"""openstack_aws_pricing_tool.py – friendly column names

Outputs:
  • OnDemand_Hourly / Monthly / Yearly
  • RI3yr_Hourly / Monthly / Yearly
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from openstack import connection
from tabulate import tabulate
from tqdm import tqdm

HOURS_IN_MONTH = 730
HOURS_IN_YEAR = 8_760
_SP = re.compile(r"\s+")

def _load_csv(path: Path) -> pd.DataFrame:
    with path.open() as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith(('SKU', '"SKU"')):
                header=i;break
        else:
            raise ValueError('header not found')
    return pd.read_csv(path, skiprows=header, quotechar='"', dtype=str)

def _norm(df):
    df.columns = (_SP.sub(' ', c.strip()).title() for c in df.columns)
    df.rename(columns={'Instancetype':'Instance Type','Vpcu':'Vcpu','Priceperunit':'Price Per Unit','Termtype':'Term Type','Leasecontractlength':'Lease Contract Length','Purchaseoption':'Purchase Option'},inplace=True)

def _pricing(df: pd.DataFrame):
    """Extract pricing dicts and gp3 hourly rate with stricter filters.

    * Only **current‑generation** instance rows
    * On‑Demand rows must have a PriceDescription that contains "per On Demand"
    * Reserved‑Instance rows must be the recurring hourly part (ignore "Upfront Fee")
    """
    _norm(df)
    df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')

    inst = df[
        (df['Product Family'].str.lower() == 'compute instance') &
        (df['Region Code'].str.lower() == 'us-east-1') &
        (df['Unit'] == 'Hrs') & (df['Currency'] == 'USD') &
        (df['Tenancy'].fillna('Shared') == 'Shared') &
        (df['Operating System'].fillna('Linux') == 'Linux') &
        (df['Current Generation'].fillna('Yes') == 'Yes') &
        (df['Price Per Unit'] > 0)
    ].copy()

    # Build spec map (vCPU, Memory GiB) from current‑gen rows
    spec: Dict[str, Tuple[int, int]] = {}
    for itype, grp in inst.groupby('Instance Type'):
        row = grp.iloc[0]
        mem_gib = int(float(str(row['Memory']).split()[0]))
        spec[itype] = (int(row['Vcpu']), mem_gib)

    od_df = inst[(inst['Term Type'] == 'OnDemand') &
                 (inst['Price Description'].str.contains('per On Demand', case=False, na=False))]
    od = od_df.groupby('Instance Type')['Price Per Unit'].min().to_dict()

    ri_df = inst[(inst['Term Type'] == 'Reserved') &
                 (inst['Lease Contract Length'].str.startswith('3')) &
                 (inst['Purchase Option'] == 'No Upfront') &
                 (~inst['Price Description'].str.contains('Upfront Fee', case=False, na=False))]
    ri = ri_df.groupby('Instance Type')['Price Per Unit'].min().to_dict()

    # gp3 hourly per‑GB
    _norm(df)  # ensure storage columns normalized too
    gp3_row = df[(df['Product Family'].str.lower() == 'storage') &
                 (df['Volume Api Name'] == 'gp3') &
                 (df['Region Code'].str.lower() == 'us-east-1') &
                 (df['Unit'].str.contains('GB')) &
                 (df['Term Type'] == 'OnDemand') &
                 (df['Price Per Unit'] > 0)]
    gp3_hr = pd.to_numeric(gp3_row['Price Per Unit']).min() / HOURS_IN_MONTH

    return od, ri, spec, gp3

def _pick(vcpu:int,ram:int,prices,spec):
    gib=(ram+1023)//1024
    opts=[(t,p) for t,p in prices.items() if spec[t][0]>=vcpu and spec[t][1]>=gib]
    return min(opts,key=lambda x:x[1]) if opts else (None,None)

def build(conn, pid, csv):
    od,ri,spec,gp3=_pricing(_load_csv(csv))
    flav={f.id:f for f in conn.compute.flavors()};flav.update({f.name:f for f in flav.values()})
    seen={}
    rows=[]
    servers=list(conn.compute.servers(details=True,all_projects=True))
    with tqdm(total=len(servers),desc='VMs',unit='vm') as bar:
        for s in servers:
            bar.update(1)
            if pid and s.project_id!=pid:continue
            f=flav.get(s.flavor.get('id') or s.flavor.get('original_name')) or conn.compute.find_flavor(s.flavor.get('id'),ignore_missing=True)
            if not f:continue
            v,r,root=f.vcpus,f.ram,f.disk
            disk=root
            for a in conn.compute.volume_attachments(server=s):
                if a.id not in seen:seen[a.id]=conn.block_storage.get_volume(a.id).size
                disk+=seen[a.id]
            t,od_hr=_pick(v,r,od,spec)
            ri_hr=ri.get(t) if t else None
            stor=disk*gp3
            rows.append({'Project':s.project_id,'Server':s.name,'vCPU':v,'RAM_GiB':(r+1023)//1024,'Disk_GB':disk,'AWS_Type':t or 'N/A','OnDemand_Hourly':None if od_hr is None else od_hr+stor,'RI3yr_Hourly':None if ri_hr is None else ri_hr+stor})
    df=pd.DataFrame(rows)
    for col in ['OnDemand_Hourly','RI3yr_Hourly']:
        df[col.replace('Hourly','Monthly')]=df[col]*HOURS_IN_MONTH
        df[col.replace('Hourly','Yearly')]=df[col]*HOURS_IN_YEAR
    total={c:df[c].sum() if df[c].dtype.kind in 'f' else ('TOTAL' if c=='Server' else '') for c in df.columns}
    return pd.concat([df,pd.DataFrame([total])],ignore_index=True)

def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--project');g.add_argument('--all-projects',action='store_true')
    p.add_argument('--cloud',default='openstack');p.add_argument('--aws-csv',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args()
    conn=connection.from_config(cloud=a.cloud)
    pid=None
    if not a.all_projects:
        pr=conn.identity.find_project(a.project,ignore_missing=True);pid=pr.id if pr else a.project
    df=build(conn,pid,a.aws_csv)
    if a.output:
        df.to_csv(a.output,index=False);print('saved →',a.output)
    else:
        print(tabulate(df,headers='keys',tablefmt='github',floatfmt='.4f'))

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:sys.exit(130)
