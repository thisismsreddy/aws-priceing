#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file Flask web application for OpenStack -> AWS cost comparison.
(Simplified UI version for debugging TemplateSyntaxError)
"""
from __future__ import annotations

import io
import logging
import os
import re
import sys
import traceback
import json # Needed for pretty-printing JSON in the simplified UI
from pathlib import Path # Still needed for Path type hint, but not file loading
from typing import Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from openstack import connection
from openstack.exceptions import OpenStackCloudException

# --- Constants ---
HOURS_IN_MONTH = 730
HOURS_IN_YEAR = 8_760
_SPACE_RE = re.compile(r"\s+")
ALLOWED_EXTENSIONS = {'csv'}

# --- Flask App Setup ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions (Pricing Logic - Kept Intact) ---
# _allowed_file, _load_csv, _norm_cols, _extract_pricing, _pick_shape, build_report
# remain EXACTLY the same as in the previous version.
# --- SNIPPED FOR BREVITY - Assume they are correctly copied from the previous response ---
# --- Make sure you copy them from the previous version into this spot! ---

# START COPYING FROM PREVIOUS SCRIPT HERE

def _allowed_file(filename):
    return '.' in filename and \           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _load_csv(file_stream: io.BytesIO) -> pd.DataFrame:
    """Load AWS price CSV starting at header row from a file stream."""
    try:
        # Read into memory first to find header
        content = file_stream.read().decode('utf-8', errors='ignore')
        lines = content.splitlines()
        header = -1
        for i, line in enumerate(lines):
            # Handle potential quotes around SKU
            clean_line = line.lstrip().strip('"')
            if clean_line.startswith("SKU"):
                header = i
                break
        else:
            # Try a looser match if exact 'SKU' not found (e.g., BOM characters)
             for i, line in enumerate(lines):
                if 'SKU' in line[:20]: # Check near the start
                     logging.warning(f"Found 'SKU' header heuristically on line {i+1}. CSV might have leading issues.")
                     header = i
                     break
             else:
                raise ValueError("Header row containing 'SKU' not found in price file")

        # Go back to the beginning of the stream simulation
        csv_data = io.StringIO(content)
        # Be explicit about quote character and handle bad lines
        return pd.read_csv(csv_data, skiprows=header, quotechar='"', dtype=str, low_memory=False, on_bad_lines='warn')
    except Exception as e:
        logging.error(f"Error reading or parsing CSV stream: {e}")
        raise ValueError(f"Failed to process CSV file: {e}")


def _norm_cols(df: pd.DataFrame) -> None:
    """Normalise column names in‑place and add aliases."""
    # Store original columns to check existence later
    original_columns = {c.strip().title() for c in df.columns}

    df.columns = (_SPACE_RE.sub(" ", c.strip()).title() for c in df.columns)

    # Define aliases, check if the *original* (case-insensitive normalized) column existed
    alias = {
        'Instancetype': 'Instance Type',
        'Vpcu': 'Vcpu',
        'Priceperunit': 'Price Per Unit',
        'Pricedescription': 'Price Description',
        'Termtype': 'Term Type',
        'Leasecontractlength': 'Lease Contract Length',
        'Purchaseoption': 'Purchase Option',
        'Region Code': 'Region', # Changed from Region Code to Region
        'Product Family': 'Product Family',
        'Memory': 'Memory',
        'Unit': 'Unit',
        'Currency': 'Currency',
        'Tenancy': 'Tenancy',
        'Operating System': 'Operating System',
        'Current Generation': 'Current Generation',
        'Volume Api Name': 'Volume Api Name',
    }
    # Apply renaming only for columns that exist *after* title casing
    rename_map = {k: v for k, v in alias.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)
    logging.debug(f"Normalized columns: {list(df.columns)}")


def _extract_pricing(df: pd.DataFrame):
    """Extracts OD, RI, Spec, and gp3 pricing from the DataFrame."""
    _norm_cols(df) # Normalize once at the beginning

    # --- Coerce Price to Numeric ---
    if 'Price Per Unit' not in df.columns:
        # Try common variations if exact match failed
        potential_price_cols = [c for c in df.columns if 'price' in c.lower() and ('unit' in c.lower() or c.lower().endswith('price'))]
        if potential_price_cols:
             price_col_found = potential_price_cols[0]
             logging.warning(f"'Price Per Unit' not found, using heuristic match: '{price_col_found}'")
             df.rename(columns={price_col_found: 'Price Per Unit'}, inplace=True)
        else:
             raise ValueError("Could not find a suitable 'Price Per Unit' column after normalization.")

    df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')
    df.dropna(subset=['Price Per Unit'], inplace=True) # Drop rows where conversion failed

    # --- Base Compute Filter ---
    required_cols = ['Product Family', 'Region', 'Unit', 'Currency']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for filtering: {', '.join(missing_cols)}")

    # Ensure columns used in filtering exist before using .str accessor
    df['Product Family Lower'] = df['Product Family'].str.lower()
    df['Region Lower'] = df['Region'].str.lower()
    df['Tenancy Filled'] = df.get('Tenancy', pd.Series(dtype=str)).fillna('Shared')
    df['OS Filled'] = df.get('Operating System', pd.Series(dtype=str)).fillna('Linux')
    df['Current Gen Filled'] = df.get('Current Generation', pd.Series(dtype=str)).fillna('Yes')


    base = df[
        (df['Product Family Lower'] == 'compute instance') &
        (df['Region Lower'] == 'us-east-1') &
        (df['Unit'] == 'Hrs') &
        (df['Currency'] == 'USD') &
        (df['Tenancy Filled'] == 'Shared') &
        (df['OS Filled'] == 'Linux') &
        (df['Current Gen Filled'] == 'Yes') &
        (df['Price Per Unit'] > 0)
    ].copy()

    if base.empty:
        logging.warning("Base compute filter (us-east-1, Linux, Shared, Hrs, USD, Current Gen) resulted in an empty DataFrame. Check CSV content and filters.")
        return {}, {}, {}, 0.0

    # --- Spec Map ---
    spec: Dict[str, Tuple[int, int]] = {}
    required_spec_cols = ['Instance Type', 'Vcpu', 'Memory']
    if not all(col in base.columns for col in required_spec_cols):
         raise ValueError(f"Missing required columns for spec mapping: {', '.join([c for c in required_spec_cols if c not in base.columns])}")

    for itype, grp in base.groupby('Instance Type'):
        if grp.empty: continue
        first = grp.iloc[0]
        try:
            mem_str = str(first.get('Memory', '0 GiB')).split()[0]
            # Handle potential locale-specific floats (e.g., "1,024.5" or "1.024,5") - normalize to use '.'
            mem_str_norm = mem_str.replace(',', '.')
            mem_gib = int(float(mem_str_norm))
            vcpu_val_str = str(first['Vcpu']).replace(',', '.')
            vcpu_val = int(float(vcpu_val_str)) # Handle Vcpu potentially being float like "2.0"
            spec[str(itype)] = (vcpu_val, mem_gib)
        except (ValueError, TypeError, IndexError) as e:
            logging.warning(f"Could not parse spec for '{itype}': {e} - Memory='{first.get('Memory', 'N/A')}', Vcpu='{first.get('Vcpu', 'N/A')}'. Skipping this type.")
            continue

    # --- OnDemand Pricing ---
    od_rows = base[(base.get('Term Type', pd.Series(dtype=str)) == 'OnDemand') &
                   (base.get('Price Description', pd.Series(dtype=str)).str.contains('linux on demand', case=False, na=False))] # More specific OD match
    if 'Instance Type' not in od_rows.columns or od_rows.empty:
         od = {}
         logging.warning("No OnDemand rows found matching criteria. OD prices will be empty.")
    else:
         # Prefer minimum price, then maybe check description further if needed
         od = od_rows.loc[od_rows.groupby('Instance Type')['Price Per Unit'].idxmin()]
         od = od.set_index('Instance Type')['Price Per Unit'].to_dict()


    # --- Reserved Instance Pricing ---
    ri_rows = base[(base.get('Term Type', pd.Series(dtype=str)) == 'Reserved') &
                   (base.get('Lease Contract Length', pd.Series(dtype=str)).str.strip().str.startswith('3', na=False)) & # Handle whitespace
                   (base.get('Purchase Option', pd.Series(dtype=str)) == 'No Upfront') &
                   (base.get('Price Description', pd.Series(dtype=str)).str.contains('linux reserved instance', case=False, na=False)) & # More specific RI match
                   (~base.get('Price Description', pd.Series(dtype=str)).str.contains('upfront fee', case=False, na=False))]

    if 'Instance Type' not in ri_rows.columns or ri_rows.empty:
        ri = {}
        logging.warning("No 3yr No Upfront RI rows found matching criteria. RI prices will be empty.")
    else:
        # Group by instance type and find the minimum price for that type
        ri = ri_rows.loc[ri_rows.groupby('Instance Type')['Price Per Unit'].idxmin()]
        ri = ri.set_index('Instance Type')['Price Per Unit'].to_dict()

    # --- GP3 Storage Pricing ---
    # Use the original df for storage pricing (after normalization)
    # Re-check required columns
    storage_required_cols = ['Product Family', 'Volume Api Name', 'Region', 'Unit', 'Term Type', 'Price Per Unit']
    missing_storage_cols = [c for c in storage_required_cols if c not in df.columns]
    if missing_storage_cols:
        logging.warning(f"Missing columns needed for GP3 Storage pricing: {', '.join(missing_storage_cols)}. Storage cost will be 0.")
        gp3_hr = 0.0
    else:
        # Ensure relevant columns have expected types before filtering
        df['Volume Api Name Lower'] = df['Volume Api Name'].str.lower()
        df['Unit Lower'] = df['Unit'].str.lower()
        df['Term Type Lower'] = df['Term Type'].str.lower()

        gp3_rows = df[
            (df['Product Family Lower'] == 'storage') &
            (df['Volume Api Name Lower'] == 'gp3') &
            (df['Region Lower'] == 'us-east-1') &
            (df['Unit Lower'].str.contains('gb-mo', na=False)) & # Match 'gb-mo' specifically
            (df['Term Type Lower'] == 'ondemand') &
            (df['Price Per Unit'] > 0)
        ]

        gp3_hr = 0.0
        if not gp3_rows.empty:
            gp3_monthly_per_gb = pd.to_numeric(gp3_rows['Price Per Unit'], errors='coerce').min()
            if pd.notna(gp3_monthly_per_gb) and gp3_monthly_per_gb > 0:
                gp3_hr = gp3_monthly_per_gb / HOURS_IN_MONTH
                logging.info(f"Found gp3 price: ${gp3_monthly_per_gb:.4f}/GB-Mo -> ${gp3_hr:.8f}/GB-Hr")
            else:
                logging.warning("Found gp3 rows, but could not determine a valid minimum monthly price > 0.")
        else:
            logging.warning("No matching gp3 storage rows found (us-east-1, OnDemand, GB-Mo). Storage cost will be 0.")

    logging.info(f"Pricing extracted: {len(od)} OD types, {len(ri)} RI types, {len(spec)} Specs, gp3/hr={gp3_hr:.8f}")
    # Clean up temporary columns
    df.drop(columns=[col for col in ['Product Family Lower', 'Region Lower', 'Tenancy Filled', 'OS Filled', 'Current Gen Filled', 'Volume Api Name Lower', 'Unit Lower', 'Term Type Lower'] if col in df.columns], inplace=True, errors='ignore')

    return od, ri, spec, gp3_hr


def _pick_shape(vcpu: int, ram_mb: int, prices: Dict[str, float], spec: Dict[str, Tuple[int, int]]):
    """Picks the cheapest AWS instance type meeting vCPU/RAM requirements."""
    if not prices or not spec: return (None, None)
    if vcpu <= 0 or ram_mb <= 0: return (None, None) # Cannot match invalid requirements

    ram_gib = (ram_mb + 1023) // 1024
    choices = []
    for t, p in prices.items():
        instance_spec = spec.get(t)
        # Ensure price is valid number
        if instance_spec and isinstance(p, (int, float)) and p > 0:
            instance_vcpu, instance_ram_gib = instance_spec
            if instance_vcpu >= vcpu and instance_ram_gib >= ram_gib:
                choices.append((t, p, instance_vcpu, instance_ram_gib)) # Include specs for sorting

    if not choices:
        return (None, None)

    # Sort by price (primary), then vCPU (secondary, prefer smaller fit), then RAM (tertiary)
    choices.sort(key=lambda x: (x[1], x[2], x[3]))

    # Return only the type and price of the best choice
    return choices[0][0], choices[0][1]


def build_report(conn, project_id: Optional[str], csv_stream: io.BytesIO):
    """Builds the cost comparison report."""
    try:
        df_aws = _load_csv(csv_stream)
        if df_aws.empty:
             raise ValueError("Loaded AWS CSV is empty after finding header.")
        od, ri, spec, gp3 = _extract_pricing(df_aws)
    except ValueError as e:
        logging.error(f"Error processing CSV: {e}")
        raise # Re-raise to be caught by the API handler
    except Exception as e:
        logging.error(f"Unexpected error during pricing extraction: {e}\n{traceback.format_exc()}")
        raise ValueError(f"Failed to extract pricing data: {e}")

    if not spec:
         logging.warning("AWS Spec dictionary is empty. Cannot map instance types.")
         # Depending on requirements, maybe raise error or return empty report
         # raise ValueError("Could not extract any valid instance specifications from the AWS CSV.")


    try:
        flavors_list = list(conn.compute.flavors())
        flavors = {f.id: f for f in flavors_list}
        # Add name mapping as fallback - careful with duplicate names
        flavors_by_name = {f.name: f for f in flavors_list}
        flavors.update(flavors_by_name) # ID mapping takes precedence if collision
        logging.info(f"Fetched {len(flavors_list)} OpenStack flavors ({len(flavors)} unique refs).")
        if not flavors:
            raise ConnectionError("No flavors found in the OpenStack cloud.")
    except OpenStackCloudException as e:
        logging.error(f"Failed to fetch OpenStack flavors: {e}")
        raise ConnectionError(f"Could not fetch flavors from OpenStack: {e}")
    except Exception as e:
        logging.error(f"Unexpected error fetching flavors: {e}")
        raise ConnectionError(f"Unexpected error fetching flavors: {e}")


    seen_vol: Dict[str, int] = {}
    rows: List[Dict[str, object]] = []
    servers = []

    try:
        project_display_name = "ALL" # Default for logging
        # Determine if we need all projects or a specific one
        if project_id is None: # Indicates 'all projects' was selected
            servers_generator = conn.compute.servers(details=True, all_projects=True)
            logging.info("Fetching servers for ALL projects...")
        else:
            # Project ID should already be validated in the API route
            try:
                project = conn.identity.get_project(project_id)
                project_display_name = f"{project.name} ({project_id})"
                logging.info(f"Fetching servers for project: {project_display_name}")
                servers_generator = conn.compute.servers(details=True, project_id=project_id)
            except OpenStackCloudException as e:
                 logging.error(f"Error fetching project details {project_id} (should have been verified earlier): {e}")
                 raise ValueError(f"Could not access project '{project_id}'. Check permissions.")


        server_count = 0
        processed_count = 0
        skipped_no_flavor = 0
        skipped_no_match = 0

        for srv in servers_generator:
             server_count += 1
             if server_count % 100 == 0:
                 logging.info(f"Checking server {server_count}...")

             # Filter again just in case `all_projects=True` still returns others (unlikely)
             if project_id and srv.project_id != project_id:
                 continue

             f_ref = srv.flavor.get('id') # Prefer ID
             flav = flavors.get(f_ref)

             if not flav:
                  f_ref_name = srv.flavor.get('original_name') or srv.flavor.get('name')
                  if f_ref_name:
                      flav = flavors.get(f_ref_name)
                      if flav:
                          logging.debug(f"Server '{srv.name}' ({srv.id}): Found flavor by name '{f_ref_name}' after ID '{f_ref}' failed.")
                      else:
                          logging.warning(f"Server '{srv.name}' ({srv.id}): Flavor ID '{f_ref}' and name '{f_ref_name}' not found in fetched list. Skipping.")
                          skipped_no_flavor += 1
                          continue
                  else:
                       logging.warning(f"Server '{srv.name}' ({srv.id}): Flavor reference missing or not found (ID: '{f_ref}', Name: None). Skipping.")
                       skipped_no_flavor += 1
                       continue

             # Ensure flavor has necessary attributes
             if not all(hasattr(flav, attr) for attr in ['vcpus', 'ram', 'disk']):
                  logging.warning(f"Server '{srv.name}' ({srv.id}): Flavor '{f_ref}' is missing required attributes (vcpus, ram, disk). Skipping.")
                  skipped_no_flavor += 1
                  continue

             vcpus, ram_mb, root_disk = flav.vcpus, flav.ram, flav.disk

             # Handle missing flavor specs gracefully
             if vcpus is None or ram_mb is None:
                 logging.warning(f"Server '{srv.name}' ({srv.id}): Flavor '{f_ref}' has missing vCPU ({vcpus}) or RAM ({ram_mb}). Skipping.")
                 skipped_no_flavor += 1
                 continue

             disk = root_disk if root_disk is not None else 0

             # Get attached volumes
             try:
                 # Convert generator to list to avoid potential issues if iterated multiple times or connection drops
                 attachments = list(conn.compute.volume_attachments(server=srv))
                 for att in attachments:
                     vol_id = att.volume_id # Correct attribute
                     if not vol_id: continue
                     if vol_id not in seen_vol:
                         try:
                             # Fetch volume details only once
                             volume = conn.block_storage.get_volume(vol_id)
                             seen_vol[vol_id] = volume.size if volume and volume.size is not None else 0
                             logging.debug(f"Fetched volume {vol_id} size: {seen_vol[vol_id]} GB")
                         except OpenStackCloudException as e:
                             logging.warning(f"Could not get volume details for {vol_id} attached to {srv.name}: {e}. Assuming size 0.")
                             seen_vol[vol_id] = 0 # Cache failure as 0 size
                     disk += seen_vol.get(vol_id, 0) # Add cached size
             except OpenStackCloudException as e:
                logging.warning(f"Could not list volume attachments for server {srv.name}: {e}. Disk size might be missing attached volumes.")


             # Find matching AWS instance type using extracted pricing data
             itype, od_hr_inst = _pick_shape(vcpus, ram_mb, od, spec)

             if itype is None:
                 logging.debug(f"Server '{srv.name}' ({srv.id}) - vCPU={vcpus}, RAM={ram_mb}MB: No matching AWS OnDemand instance found in price list.")
                 # Still try RI mapping? If OD mapping failed, RI likely will too if based on same spec.
                 ri_hr_inst = None
                 skipped_no_match += 1
                 # Option: Still include row with N/A for costs, or skip entirely? Include for now.
             else:
                  # RI price lookup is direct using the matched OD type 'itype'
                  ri_hr_inst = ri.get(itype) if ri else None # Ensure ri dict exists
                  if not ri_hr_inst:
                      logging.debug(f"Server '{srv.name}' ({srv.id}): Found OD match '{itype}', but no corresponding RI price found.")


             # Calculate storage cost safely
             storage_hr = disk * gp3 if disk is not None and gp3 is not None and gp3 > 0 else 0

             # Calculate final costs, handling None values from mapping/lookups
             od_final = (od_hr_inst + storage_hr) if od_hr_inst is not None else None
             ri_final = (ri_hr_inst + storage_hr) if ri_hr_inst is not None else None

             rows.append({
                 'Project': srv.project_id,
                 'Server': srv.name,
                 'vCPU': vcpus,
                 'RAM_GiB': (ram_mb + 1023) // 1024 if ram_mb else 0,
                 'Disk_GB': disk,
                 'AWS_Type': itype or 'N/A', # Use 'N/A' if no match found
                 'OnDemand_Hourly': od_final,
                 'RI3yr_Hourly': ri_final,
             })
             processed_count += 1

        logging.info(f"Finished processing servers for project: {project_display_name}.")
        logging.info(f"Total servers checked: {server_count}. Added to report: {processed_count}. Skipped (no flavor/spec): {skipped_no_flavor}. Skipped (no AWS match): {skipped_no_match}.")

    except OpenStackCloudException as e:
        logging.error(f"Error during OpenStack server/volume processing: {e}")
        raise ConnectionError(f"Error communicating with OpenStack while processing VMs: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing servers: {e}\n{traceback.format_exc()}")
        raise RuntimeError(f"An unexpected error occurred while processing VMs: {e}")


    if not rows:
        logging.warning("No servers found or processed for the given criteria.")
        # Return an empty DataFrame with expected columns for consistency
        return pd.DataFrame(columns=[
            'Project', 'Server', 'vCPU', 'RAM_GiB', 'Disk_GB', 'AWS_Type',
            'OnDemand_Hourly', 'RI3yr_Hourly', 'OnDemand_Monthly', 'OnDemand_Yearly',
            'RI3yr_Monthly', 'RI3yr_Yearly'
        ])

    df = pd.DataFrame(rows)

    # Calculate Monthly/Yearly costs
    for col in ['OnDemand_Hourly', 'RI3yr_Hourly']:
        if col in df.columns:
            # Convert column to numeric, coercing errors (like None) to NaN
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            df[col.replace('Hourly', 'Monthly')] = numeric_col * HOURS_IN_MONTH
            df[col.replace('Hourly', 'Yearly')] = numeric_col * HOURS_IN_YEAR
        else:
             # This case should ideally not happen if rows are appended correctly
             logging.error(f"Expected column '{col}' not found in DataFrame during final calculation.")
             df[col.replace('Hourly', 'Monthly')] = None
             df[col.replace('Hourly', 'Yearly')] = None

    # Add TOTAL row
    total_row = {}
    numeric_cols_for_sum = df.select_dtypes(include='number').columns
    for c in df.columns:
        if c in numeric_cols_for_sum:
            total_row[c] = df[c].sum(skipna=True) # Ensure NaNs are skipped
        elif c == 'Server':
            total_row[c] = 'TOTAL'
        else:
            total_row[c] = '' # Keep other string columns blank

    total_df = pd.DataFrame([total_row])
    # Ensure column order matches before concat
    total_df = total_df[df.columns]

    final_df = pd.concat([df, total_df], ignore_index=True)

    # Final check for data types before returning
    logging.debug(f"Final DataFrame info:\n{final_df.info()}")

    return final_df

# END COPYING FROM PREVIOUS SCRIPT HERE


# --- Simplified HTML Template, CSS, and JavaScript ---

SIMPLE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simplified Pricing Tool</title>
    <style>
        body {{
            font-family: sans-serif; margin: 20px; line-height: 1.6;
            background-color: #f4f4f4;
        }}
        .container {{
             max-width: 800px; margin: auto; background: #fff;
             padding: 20px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{ text-align: center; color: #333; }}
        form {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; background: #f9f9f9; }}
        .form-group {{ margin-bottom: 15px; }}
        label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
        input[type="text"], input[type="file"] {{
            width: 98%; padding: 8px; border: 1px solid #ccc; border-radius: 3px;
        }}
        button {{
            padding: 10px 15px; background-color: #5cb85c; color: white;
            border: none; border-radius: 3px; cursor: pointer;
        }}
        button:disabled {{ background-color: #aaa; cursor: not-allowed; }}
        button:hover:enabled {{ background-color: #4cae4c; }}
        #loading {{ display: none; text-align: center; margin: 15px 0; color: #555; }}
        #error-message {{
            display: none; color: #d9534f; border: 1px solid #d9534f;
            padding: 10px; margin-top: 15px; border-radius: 3px; background-color: #f2dede;
        }}
        #results {{ margin-top: 20px; border: 1px solid #eee; padding: 10px; background: #fff; }}
        #results pre {{
             white-space: pre-wrap; /* Wrap long lines */
             word-wrap: break-word; /* Break words if necessary */
             max-height: 500px; /* Limit height */
             overflow-y: auto; /* Add scrollbar if needed */
             background: #f8f8f8;
             padding: 10px;
             border: 1px solid #ddd;
        }}
        .checkbox-group label {{ display: inline; font-weight: normal; margin-left: 5px; }}
        input[type="checkbox"] {{ vertical-align: middle; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Simplified OpenStack vs AWS Pricing</h1>

        <form id="pricing-form">
            <div class="form-group">
                <label for="cloud">OpenStack Cloud:</label>
                <input type="text" id="cloud" name="cloud" required>
            </div>
            <div class="form-group">
                <label for="project">Project Name or ID:</label>
                <input type="text" id="project" name="project">
            </div>
            <div class="form-group checkbox-group">
                 <input type="checkbox" id="all_projects" name="all_projects">
                 <label for="all_projects">Process All Projects</label>
            </div>
            <div class="form-group">
                <label for="aws_csv">AWS Pricing CSV:</label>
                <input type="file" id="aws_csv" name="aws_csv" accept=".csv" required>
            </div>
            <button type="submit" id="submit-button">Calculate</button>
        </form>

        <div id="loading">Loading...</div>
        <div id="error-message"></div>
        <div id="results">
            <h2>Raw Results (JSON)</h2>
            <pre id="results-data">Submit the form to see results here.</pre>
        </div>
    </div>

    <!-- jQuery CDN -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>

    <script>
        $(document).ready(function() {
            // Hide results initially
            $('#results').hide();

            // Handle checkbox interaction
            $('#all_projects').on('change', function() {
                const isChecked = $(this).is(':checked');
                $('#project').prop('disabled', isChecked);
                if (isChecked) {
                    $('#project').val('');
                }
            });
            $('#all_projects').trigger('change'); // Set initial state

            // Handle form submission
            $('#pricing-form').on('submit', function(event) {
                event.preventDefault(); // Stop default form submission

                const formData = new FormData(this);
                const $submitButton = $('#submit-button');
                const $loading = $('#loading');
                const $errorMsg = $('#error-message');
                const $resultsDiv = $('#results');
                const $resultsData = $('#results-data');

                // Basic validation
                const cloud = formData.get('cloud').trim();
                const project = formData.get('project').trim();
                const allProjects = $('#all_projects').is(':checked');
                const fileInput = $('#aws_csv')[0];

                $errorMsg.hide().text(''); // Clear previous errors
                $resultsDiv.hide();
                $resultsData.text(''); // Clear previous results

                if (!cloud) {
                     $errorMsg.text('Error: Please enter the OpenStack Cloud name.').show();
                     return;
                }
                if (!allProjects && !project) {
                     $errorMsg.text('Error: Please enter a Project Name/ID or check "Process All Projects".').show();
                    return;
                }
                 if (!fileInput.files || fileInput.files.length === 0) {
                    $errorMsg.text('Error: Please select an AWS Pricing CSV file.').show();
                    return;
                 }
                const file = fileInput.files[0];
                 if (file.type && file.type !== "text/csv" && !file.name.toLowerCase().endsWith('.csv')) {
                    $errorMsg.text('Error: Invalid file type. Please upload a CSV file.').show();
                    return;
                 }
                 if (file.size > 16 * 1024 * 1024) { // Check size (matches Flask config)
                    $errorMsg.text('Error: File size exceeds 16 MB limit.').show();
                    return;
                 }

                // UI updates: disable button, show loading
                $submitButton.prop('disabled', true).text('Calculating...');
                $loading.show();

                // AJAX call
                $.ajax({
                    url: '/api/calculate', // Your calculation endpoint
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    dataType: 'json', // Expect JSON back
                    success: function(response) {
                        $loading.hide();
                        $submitButton.prop('disabled', false).text('Calculate');

                        if (response.data) {
                            // Display raw JSON response
                            const prettyJson = JSON.stringify(response.data, null, 2); // Pretty print
                            $resultsData.text(prettyJson);
                            $resultsDiv.show();
                        } else if (response.error) {
                            $errorMsg.text('Error: ' + response.error).show();
                        } else {
                             $errorMsg.text('Error: Received an unexpected response format.').show();
                        }
                    },
                    error: function(jqXHR, textStatus, errorThrown) {
                        $loading.hide();
                        $submitButton.prop('disabled', false).text('Calculate');

                        let errorText = 'An unknown error occurred.';
                         if (jqXHR.responseJSON && jqXHR.responseJSON.error) {
                            errorText = jqXHR.responseJSON.error;
                        } else if (jqXHR.status === 413) {
                            errorText = 'Error: The uploaded file is too large (max 16MB).';
                        } else if (jqXHR.responseText) {
                             try {
                                // Try parsing responseText as JSON for potential error details
                                const errData = JSON.parse(jqXHR.responseText);
                                if (errData && errData.error) {
                                    errorText = errData.error;
                                } else {
                                    errorText = `Server Error (${jqXHR.status}): ${jqXHR.responseText.substring(0, 200)}`; // Limit length
                                }
                             } catch(e) {
                                // If not JSON, show plain text excerpt
                                errorText = `Server Error (${jqXHR.status}): ${jqXHR.responseText.substring(0, 200)}`;
                             }
                        } else {
                            errorText = `Network Error: ${textStatus} - ${errorThrown}`;
                        }
                        $errorMsg.text(errorText).show();
                    }
                });
            });
        });
    </script>

</body>
</html>
"""

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the SIMPLIFIED main page."""
    # **Using the simplified template string**
    return render_template_string(SIMPLE_HTML_TEMPLATE)

@app.route('/favicon.ico')
def favicon():
     return '', 204 # Simple 204 No Content

@app.route('/api/calculate', methods=['POST'])
def calculate_costs():
    """API endpoint - REMAINS THE SAME as previous versions"""
    if 'aws_csv' not in request.files:
        return jsonify({"error": "No AWS CSV file part in the request."}), 400
    file = request.files['aws_csv']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    if not file or not _allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

    cloud_name = request.form.get('cloud', '').strip()
    project_name = request.form.get('project', '').strip() # Might be empty
    use_all_projects = request.form.get('all_projects') == 'on'

    if not cloud_name:
        return jsonify({"error": "OpenStack cloud name is required."}), 400
    if not use_all_projects and not project_name:
        return jsonify({"error": "Either Project Name/ID or 'Process All Projects' must be checked."}), 400
    if use_all_projects and project_name:
        project_name = None
        logging.info("Both project and 'All Projects' provided. Using 'All Projects'.")
        project_id_to_use = None
    elif use_all_projects:
        project_id_to_use = None
        logging.info("Processing for all projects.")
    else:
        project_id_to_use = None

    # --- OpenStack Connection and Project Resolution ---
    conn = None
    try:
        logging.info(f"Attempting connection to cloud: {cloud_name}")
        conn = connection.from_config(cloud=cloud_name)
        conn.identity.get_user('self')
        logging.info(f"Successfully connected to OpenStack cloud: {cloud_name}")

        if not use_all_projects and project_name:
            logging.info(f"Resolving project identifier: '{project_name}'")
            try:
                pr = conn.identity.find_project(project_name, ignore_missing=False)
                project_id_to_use = pr.id
                logging.info(f"Found project '{pr.name}' with ID: {project_id_to_use}")
            except OpenStackCloudException as e:
                logging.error(f"Failed to find project '{project_name}': {e}")
                return jsonify({"error": f"OpenStack project '{project_name}' not found or inaccessible. Verify name/ID and permissions."}), 404
            except Exception as e_find:
                 logging.error(f"Unexpected error resolving project '{project_name}': {e_find}")
                 return jsonify({"error": f"Unexpected error resolving project '{project_name}'. Details: {e_find}"}), 500

    except OpenStackCloudException as e:
        logging.error(f"Failed to connect to OpenStack cloud '{cloud_name}': {e}")
        return jsonify({"error": f"Failed to connect to OpenStack cloud '{cloud_name}'. Check cloud config and credentials. Details: {e}"}), 503
    except Exception as e:
        logging.error(f"Unexpected error setting up OpenStack connection '{cloud_name}': {e}")
        return jsonify({"error": f"Unexpected error connecting to OpenStack cloud '{cloud_name}'. Details: {e}"}), 500
    # --- End Connection Handling ---


    # --- Perform Calculation ---
    try:
        report_df = build_report(conn, project_id_to_use, file.stream)
        report_df_clean = report_df.where(pd.notnull(report_df), None)
        json_data = report_df_clean.to_dict(orient='records')
        return jsonify({"data": json_data}) # Success

    except (ValueError, ConnectionError, RuntimeError) as e:
         logging.error(f"Calculation failed: {e}\n{traceback.format_exc()}")
         status_code = 503 if isinstance(e, ConnectionError) else 400
         return jsonify({"error": f"Calculation Error: {str(e)}"}), status_code
    except Exception as e:
         logging.error(f"An unexpected error occurred during calculation: {e}\n{traceback.format_exc()}")
         return jsonify({"error": "An internal server error occurred during calculation."}), 500
    # --- End Calculation ---

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False) # Remember to turn debug off for production
