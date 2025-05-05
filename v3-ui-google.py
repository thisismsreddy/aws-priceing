#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file Flask web application for OpenStack -> AWS cost comparison.
"""
from __future__ import annotations

import io
import logging
import os
import re
import sys
import traceback
from pathlib import Path # Still needed for Path type hint, but not file loading
from typing import Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from openstack import connection
from openstack.exceptions import OpenStackCloudException, ResourceNotFound

# --- Constants ---
HOURS_IN_MONTH = 730
HOURS_IN_YEAR = 8_760
_SPACE_RE = re.compile(r"\s+")
ALLOWED_EXTENSIONS = {'csv'}

# --- Flask App Setup ---
app = Flask(__name__)
# No need for UPLOAD_FOLDER config if we process streams directly
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# --- Logging Setup ---
# Basic logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions (Pricing Logic - Kept Intact) ---

def _allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _load_csv(file_stream: io.BytesIO) -> pd.DataFrame:
    """Load AWS price CSV starting at header row from a file stream."""
    try:
        # Read into memory first to find header
        content = file_stream.read().decode('utf-8', errors='ignore')
        lines = content.splitlines()
        header = -1
        for i, line in enumerate(lines):
            # Handle potential quotes around SKU and leading BOM/whitespace
            clean_line = line.lstrip().strip('"')
            if clean_line.startswith("SKU"):
                header = i
                break
        else:
            # Try a looser match if exact 'SKU' not found (e.g., BOM characters, slight variations)
             for i, line in enumerate(lines):
                if 'SKU' in line.upper()[:20]: # Check near the start (case-insensitive check)
                     logging.warning(f"Found 'SKU' header heuristically on line {i+1}. CSV might have leading issues or casing differences.")
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
    # Store original columns (normalized) to check existence later
    normalized_cols_set = {_SPACE_RE.sub(" ", c.strip()).title() for c in df.columns}

    df.columns = (_SPACE_RE.sub(" ", c.strip()).title() for c in df.columns)

    # Define aliases
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
    # Drop rows where Price Per Unit is NaN AFTER coercion (those that couldn't be converted)
    df.dropna(subset=['Price Per Unit'], inplace=True)


    # Define columns used in filtering and ensure they are handled safely
    filter_cols = ['Product Family', 'Region', 'Unit', 'Currency', 'Tenancy', 'Operating System', 'Current Generation']
    # Create temporary lower/filled columns for robust filtering
    for col in filter_cols:
        if col in df.columns:
            if df[col].dtype == 'object': # Only apply string methods to object type
                 df[f'{col} Lower'] = df[col].str.lower()
            # Handle specific fillna cases
            if col == 'Tenancy':
                 df[f'{col} Filled'] = df[col].fillna('Shared')
            elif col == 'Operating System':
                 df[f'{col} Filled'] = df[col].fillna('Linux')
            elif col == 'Current Generation':
                 df[f'{col} Filled'] = df[col].fillna('Yes')
            else:
                 # For other columns, just create the lower column if applicable
                 pass # Handled by the 'object' dtype check above
        else:
            # Create dummy columns if original column is missing, so filters don't fail
            df[f'{col} Lower'] = pd.Series(dtype=str)
            df[f'{col} Filled'] = pd.Series(dtype=str) # Use string default for fillers


    # --- Base Compute Filter ---
    base = df[
        (df['Product Family Lower'] == 'compute instance') &
        (df['Region Lower'] == 'us-east-1') &
        (df['Unit'] == 'Hrs') & # 'Unit' itself is fine, no need for Lower/Filled if checked directly
        (df['Currency'] == 'USD') & # 'Currency' itself is fine
        (df['Tenancy Filled'] == 'Shared') &
        (df['OS Filled'] == 'Linux') &
        (df['Current Gen Filled'] == 'Yes') &
        (df['Price Per Unit'] > 0)
    ].copy()

    # Clean up temporary columns after filtering
    temp_filter_cols = [f'{col} Lower' for col in filter_cols if f'{col} Lower' in df.columns] + \
                       [f'{col} Filled' for col in filter_cols if f'{col} Filled' in df.columns]
    df.drop(columns=temp_filter_cols, inplace=True, errors='ignore')


    if base.empty:
        logging.warning("Base compute filter (us-east-1, Linux, Shared, Hrs, USD, Current Gen) resulted in an empty DataFrame. Check CSV content and filters.")
        return {}, {}, {}, 0.0 # Return empty structures

    # --- Spec Map ---
    spec: Dict[str, Tuple[int, int]] = {}
    required_spec_cols = ['Instance Type', 'Vcpu', 'Memory']
    if not all(col in base.columns for col in required_spec_cols):
         raise ValueError(f"Missing required columns for spec mapping: {', '.join([c for c in required_spec_cols if c not in base.columns])}")

    for itype, grp in base.groupby('Instance Type'):
        if grp.empty: continue
        # Find the first row in the group that has non-null Vcpu and Memory
        valid_row = grp.dropna(subset=['Vcpu', 'Memory']).iloc[0] if not grp.dropna(subset=['Vcpu', 'Memory']).empty else None

        if valid_row is None:
             logging.warning(f"Skipping instance type '{itype}' due to missing Vcpu or Memory in all rows.")
             continue

        try:
            mem_str = str(valid_row.get('Memory', '0 GiB')).split()[0]
            # Handle potential locale-specific floats (e.g., "1,024.5" or "1.024,5") - normalize to use '.'
            mem_str_norm = mem_str.replace(',', '') # Remove thousand separators first
            if ',' in mem_str_norm and '.' in mem_str_norm: # If both exist, likely comma is decimal
                 mem_str_norm = mem_str_norm.replace('.', '').replace(',', '.')
            elif ',' in mem_str_norm: # Only comma, likely decimal
                 mem_str_norm = mem_str_norm.replace(',', '.')

            mem_gib = int(float(mem_str_norm))

            vcpu_val_str = str(valid_row['Vcpu']).replace(',', '') # Remove thousand separators
            vcpu_val = int(float(vcpu_val_str)) # Handle Vcpu potentially being float like "2.0"

            spec[str(itype)] = (vcpu_val, mem_gib)
        except (ValueError, TypeError, IndexError) as e:
            logging.warning(f"Could not parse spec for '{itype}': {e} - Memory='{valid_row.get('Memory', 'N/A')}', Vcpu='{valid_row.get('Vcpu', 'N/A')}'. Skipping this type.")
            continue

    # --- OnDemand Pricing ---
    if 'Term Type' not in base.columns or 'Price Description' not in base.columns or 'Instance Type' not in base.columns:
         logging.warning("Missing columns required for OD pricing (Term Type, Price Description, or Instance Type). OD prices will be empty.")
         od = {}
    else:
        od_rows = base[(base['Term Type'] == 'OnDemand') &
                       (base['Price Description'].str.contains('linux on demand', case=False, na=False))]

        if od_rows.empty:
             od = {}
             logging.warning("No OnDemand rows found matching criteria. OD prices will be empty.")
        else:
             # Prefer minimum price for each instance type
             od = od_rows.loc[od_rows.groupby('Instance Type')['Price Per Unit'].idxmin()]
             od = od.set_index('Instance Type')['Price Per Unit'].to_dict()


    # --- Reserved Instance Pricing ---
    ri_cols = ['Term Type', 'Lease Contract Length', 'Purchase Option', 'Price Description', 'Instance Type']
    if not all(col in base.columns for col in ri_cols):
         logging.warning(f"Missing columns required for RI pricing ({', '.join([c for c in ri_cols if c not in base.columns])}). RI prices will be empty.")
         ri = {}
    else:
        ri_rows = base[(base['Term Type'] == 'Reserved') &
                       (base['Lease Contract Length'].str.strip().str.startswith('3', na=False)) & # Handle whitespace
                       (base['Purchase Option'] == 'No Upfront') &
                       (base['Price Description'].str.contains('linux reserved instance', case=False, na=False)) & # More specific RI match
                       (~base['Price Description'].str.contains('upfront fee', case=False, na=False))] # Exclude upfront fees

        if ri_rows.empty:
            ri = {}
            logging.warning("No 3yr No Upfront RI rows found matching criteria. RI prices will be empty.")
        else:
            # Group by instance type and find the minimum price for that type
            ri = ri_rows.loc[ri_rows.groupby('Instance Type')['Price Per Unit'].idxmin()]
            ri = ri.set_index('Instance Type')['Price Per Unit'].to_dict()

    # --- GP3 Storage Pricing ---
    # Use the original df for storage pricing (after normalization)
    storage_required_cols = ['Product Family', 'Volume Api Name', 'Region', 'Unit', 'Term Type', 'Price Per Unit']
    if not all(col in df.columns for col in storage_required_cols):
        logging.warning(f"Missing columns needed for GP3 Storage pricing: {', '.join([c for c in storage_required_cols if c not in df.columns])}. Storage cost will be 0.")
        gp3_hr = 0.0
    else:
        # Create temporary lower columns for filtering
        df['Volume Api Name Lower'] = df.get('Volume Api Name', pd.Series(dtype=str)).str.lower()
        df['Unit Lower'] = df.get('Unit', pd.Series(dtype=str)).str.lower()
        df['Term Type Lower'] = df.get('Term Type', pd.Series(dtype=str)).str.lower()
        df['Product Family Lower'] = df.get('Product Family', pd.Series(dtype=str)).str.lower()
        df['Region Lower'] = df.get('Region', pd.Series(dtype=str)).str.lower()


        gp3_rows = df[
            (df['Product Family Lower'] == 'storage') &
            (df['Volume Api Name Lower'] == 'gp3') &
            (df['Region Lower'] == 'us-east-1') &
            (df['Unit Lower'].str.contains('gb-mo', na=False)) & # Match 'gb-mo' specifically
            (df['Term Type Lower'] == 'ondemand') &
            (df['Price Per Unit'] > 0)
        ]

        # Clean up temporary columns
        df.drop(columns=[col for col in ['Product Family Lower', 'Region Lower', 'Volume Api Name Lower', 'Unit Lower', 'Term Type Lower'] if col in df.columns], inplace=True, errors='ignore')

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

    return od, ri, spec, gp3_hr


def _pick_shape(vcpu: int, ram_mb: int, prices: Dict[str, float], spec: Dict[str, Tuple[int, int]]):
    """Picks the cheapest AWS instance type meeting vCPU/RAM requirements."""
    if not prices or not spec: return (None, None)
    if vcpu is None or ram_mb is None or vcpu <= 0 or ram_mb <= 0:
        logging.debug(f"Invalid input specs: vCPU={vcpu}, RAM={ram_mb}MB. Cannot pick shape.")
        return (None, None) # Cannot match invalid requirements

    ram_gib = (ram_mb + 1023) // 1024
    choices = []
    for t, p in prices.items():
        instance_spec = spec.get(t)
        # Ensure price is valid number
        if instance_spec and isinstance(p, (int, float)) and p > 0:
            instance_vcpu, instance_ram_gib = instance_spec
            if instance_vcpu is not None and instance_ram_gib is not None and instance_vcpu >= vcpu and instance_ram_gib >= ram_gib:
                choices.append((t, p, instance_vcpu, instance_ram_gib)) # Include specs for sorting

    if not choices:
        logging.debug(f"No AWS instance type found meeting vCPU={vcpu}, RAM={ram_gib}GiB requirements.")
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
         # Continue, but all 'AWS Type' and costs will likely be 'N/A' or None
         # Depending on requirements, maybe raise error if spec is *required*
         # raise ValueError("Could not extract any valid instance specifications from the AWS CSV.")


    try:
        flavors_list = list(conn.compute.flavors())
        flavors = {f.id: f for f in flavors_list if f.id is not None} # Only add if ID exists
        # Add name mapping as fallback - careful with duplicate names
        flavors_by_name = {f.name: f for f in flavors_list if f.name is not None}
        flavors.update(flavors_by_name) # ID mapping takes precedence if collision
        logging.info(f"Fetched {len(flavors_list)} OpenStack flavors ({len(flavors)} unique references after de-duping).")
        if not flavors:
            raise ConnectionError("No flavors found in the OpenStack cloud or flavors have no ID/Name.")
    except OpenStackCloudException as e:
        logging.error(f"Failed to fetch OpenStack flavors: {e}")
        raise ConnectionError(f"Could not fetch flavors from OpenStack: {e}")
    except Exception as e:
        logging.error(f"Unexpected error fetching flavors: {e}\n{traceback.format_exc()}")
        raise ConnectionError(f"Unexpected error fetching flavors: {e}")


    seen_vol: Dict[str, int] = {} # Cache for volume sizes
    rows: List[Dict[str, object]] = []
    servers_generator = None # Initialize generator variable

    try:
        project_display_name = "ALL" # Default for logging
        # Determine if we need all projects or a specific one
        if project_id is None: # Indicates 'all projects' was selected
            servers_generator = conn.compute.servers(details=True, all_projects=True)
            logging.info("Initiating fetch for ALL servers across all projects...")
        else:
            # Project ID should already be validated in the API route, but fetch details for logging
            try:
                project = conn.identity.get_project(project_id)
                project_display_name = f"{project.name} ({project_id})"
                logging.info(f"Initiating fetch for servers in project: {project_display_name}")
                servers_generator = conn.compute.servers(details=True, project_id=project_id)
            except ResourceNotFound:
                 # This case should ideally be caught earlier, but added for robustness
                 raise ValueError(f"OpenStack project '{project_id}' not found.")
            except OpenStackCloudException as e:
                 logging.error(f"Error fetching project details {project_id} (should have been verified earlier): {e}")
                 raise ValueError(f"Could not access project '{project_id}'. Check permissions.")


        server_count = 0
        processed_count = 0
        skipped_no_flavor = 0
        skipped_no_aws_match = 0

        # Iterate through servers (potentially large list)
        for srv in servers_generator:
             server_count += 1
             if server_count % 100 == 0:
                 logging.info(f"Processing server {server_count}...")

             # Filter again just in case `all_projects=True` still returns others (unlikely)
             if project_id is not None and srv.project_id != project_id:
                 continue

             f_ref_id = srv.flavor.get('id')
             f_ref_name = srv.flavor.get('original_name') or srv.flavor.get('name')
             flav = None

             if f_ref_id:
                 flav = flavors.get(f_ref_id)
             if not flav and f_ref_name:
                 flav = flavors.get(f_ref_name)
                 if flav:
                      logging.debug(f"Server '{srv.name}' ({srv.id}): Found flavor by name '{f_ref_name}' after ID '{f_ref_id}' failed.")

             if not flav:
                 logging.warning(f"Server '{srv.name}' ({srv.id}): Flavor reference '{f_ref_id or f_ref_name or 'N/A'}' not found/resolved. Skipping server.")
                 skipped_no_flavor += 1
                 continue

             # Ensure flavor has necessary attributes and they are not None
             vcpus = getattr(flav, 'vcpus', None)
             ram_mb = getattr(flav, 'ram', None)
             root_disk = getattr(flav, 'disk', None)

             if vcpus is None or ram_mb is None:
                 logging.warning(f"Server '{srv.name}' ({srv.id}): Flavor '{flav.name}' ({flav.id}) has missing vCPU ({vcpus}) or RAM ({ram_mb}). Skipping.")
                 skipped_no_flavor += 1
                 continue

             disk = root_disk if root_disk is not None else 0 # Handle potential None root disk value


             # Get attached volumes
             try:
                 # Convert generator to list to avoid potential issues if iterated multiple times or connection drops
                 attachments = list(conn.compute.volume_attachments(server=srv))
                 logging.debug(f"Server '{srv.name}' ({srv.id}): Found {len(attachments)} volume attachments.")
                 for att in attachments:
                     vol_id = getattr(att, 'volume_id', None) # Use getattr for safety
                     if not vol_id: continue # Skip attachments with no volume_id
                     if vol_id not in seen_vol:
                         try:
                             # Fetch volume details only once and cache size
                             volume = conn.block_storage.get_volume(vol_id)
                             vol_size = getattr(volume, 'size', 0) # Get size, default to 0 if None
                             seen_vol[vol_id] = vol_size
                             logging.debug(f"Fetched volume {vol_id} size: {seen_vol[vol_id]} GB")
                         except OpenStackCloudException as e:
                             logging.warning(f"Could not get volume details for {vol_id} attached to {srv.name}: {e}. Assuming size 0.")
                             seen_vol[vol_id] = 0 # Cache failure as 0 size
                         except Exception as e:
                             logging.warning(f"Unexpected error getting volume {vol_id} for {srv.name}: {e}. Assuming size 0.")
                             seen_vol[vol_id] = 0
                     disk += seen_vol.get(vol_id, 0) # Add cached size, default to 0 if lookup failed

             except OpenStackCloudException as e:
                logging.warning(f"Could not list volume attachments for server {srv.name}: {e}. Disk size might be missing attached volumes.")
             except Exception as e:
                 logging.warning(f"Unexpected error listing volume attachments for server {srv.name}: {e}. Disk size might be missing attached volumes.")


             # Find matching AWS instance type using extracted pricing data
             itype, od_hr_inst = _pick_shape(vcpus, ram_mb, od, spec)

             if itype is None:
                 logging.debug(f"Server '{srv.name}' ({srv.id}) - vCPU={vcpus}, RAM={ram_mb}MB: No matching AWS OnDemand instance found in price list.")
                 ri_hr_inst = None # If OD match failed, RI match based on that type will also fail
                 skipped_no_aws_match += 1
             else:
                  # RI price lookup is direct using the matched OD type 'itype'
                  ri_hr_inst = ri.get(itype) if ri and itype in ri else None # Ensure ri dict exists and type is in it
                  if ri_hr_inst is None: # Check specifically for None after get()
                      logging.debug(f"Server '{srv.name}' ({srv.id}): Found OD match '{itype}', but no corresponding RI price found.")


             # Calculate storage cost safely
             # Ensure disk is a number and not None, gp3 is a number > 0
             storage_hr = (disk if isinstance(disk, (int, float)) and disk is not None else 0) * (gp3 if isinstance(gp3, (int, float)) and gp3 > 0 else 0)


             # Calculate final costs, handling None values from mapping/lookups
             # If instance cost is None, the total cost should also be None
             od_final = (od_hr_inst + storage_hr) if od_hr_inst is not None else None
             ri_final = (ri_hr_inst + storage_hr) if ri_hr_inst is not None else None

             rows.append({
                 'Project': srv.project_id,
                 'Server': srv.name,
                 'vCPU': vcpus,
                 'RAM_GiB': (ram_mb + 1023) // 1024 if isinstance(ram_mb, int) and ram_mb is not None else 0,
                 'Disk_GB': disk,
                 'AWS_Type': itype or 'N/A', # Use 'N/A' if no match found
                 'OnDemand_Hourly': od_final,
                 'RI3yr_Hourly': ri_final,
             })
             processed_count += 1

        logging.info(f"Finished processing servers for project: {project_display_name}.")
        logging.info(f"Total servers checked: {server_count}. Added to report: {processed_count}. Skipped (no flavor/spec): {skipped_no_flavor}. Skipped (no AWS instance match): {skipped_no_aws_match}.")

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
            # Convert column to numeric, coercing errors (like None/NaN) to NaN.
            # Calculations with NaN result in NaN, which is correct for missing costs.
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            df[col.replace('Hourly', 'Monthly')] = numeric_col * HOURS_IN_MONTH
            df[col.replace('Hourly', 'Yearly')] = numeric_col * HOURS_IN_YEAR
        else:
             logging.error(f"Expected column '{col}' not found in DataFrame during final calculation.")
             df[col.replace('Hourly', 'Monthly')] = None
             df[col.replace('Hourly', 'Yearly')] = None

    # Add TOTAL row
    total_row = {}
    # Select columns that *should* contain numbers for summation
    numeric_cols_to_sum = ['vCPU', 'RAM_GiB', 'Disk_GB',
                           'OnDemand_Hourly', 'RI3yr_Hourly',
                           'OnDemand_Monthly', 'OnDemand_Yearly',
                           'RI3yr_Monthly', 'RI3yr_Yearly']

    for c in df.columns:
        if c in numeric_cols_to_sum:
             # Calculate sum, skipping NaNs. Ensure the column is numeric first.
            total_row[c] = pd.to_numeric(df[c], errors='coerce').sum(skipna=True)
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


# --- HTML Template, CSS, and JavaScript ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenStack vs AWS Pricing</title>
    <!-- DataTables CSS from CDN -->
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
    <!-- Optional: DataTables Buttons CSS -->
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/2.4.1/css/buttons.dataTables.min.css">

    <style>
        /* --- Embedded CSS --- */
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0; /* Remove default body margin */
            padding: 20px; /* Add overall padding */
            background-color: #f4f7f6;
            color: #333;
            font-size: 14px;
            line-height: 1.5; /* Improved readability */
        }}
        .container {{
            max-width: 1400px; /* Increased max width */
            margin: 20px auto; /* Center with top/bottom margin */
            background-color: #fff;
            padding: 20px 30px; /* More padding on sides */
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            text-align: center;
            color: #2c3e50;
            margin-top: 0; /* Remove top margin */
        }}
        h1 {{ margin-bottom: 30px; }}
        h2 {{ margin-top: 40px; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }}

        form {{
            margin-bottom: 30px;
            padding: 25px;
            border: 1px solid #dce3e9;
            border-radius: 5px;
            background-color: #fdfdfd;
        }}
        .form-group {{
            margin-bottom: 18px;
        }}
        label {{
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            color: #555;
        }}
        input[type="text"], input[type="file"], select {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 14px;
        }}
        input[type="file"] {{
            padding: 5px;
            background-color: #fff;
        }}
        .form-row {{
            display: flex;
            gap: 25px;
            align-items: flex-end;
            flex-wrap: wrap;
        }}
        .form-row .form-group {{
            flex: 1;
            min-width: 220px;
        }}
        .form-group.checkbox-group {{
            flex: 0 0 auto;
            align-self: center;
            padding-bottom: 10px;
        }}
        .checkbox-group label {{
            display: inline;
            margin-left: 5px;
            font-weight: normal;
        }}
        input[type="checkbox"] {{
            margin-right: 5px;
            vertical-align: middle;
            transform: scale(1.2); /* Slightly larger checkbox */
        }}

        button[type="submit"] {{
            padding: 12px 25px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
            transition: background-color 0.2s ease;
        }}
        button[type="submit"]:hover {{
            background-color: #2980b9;
        }}
        button[type="submit"]:disabled {{
            background-color: #bdc3c7;
            cursor: not-allowed;
        }}

        #loading {{
            display: none;
            text-align: center;
            margin: 30px auto;
            font-size: 1.2em;
            color: #555;
        }}
        #loading img {{
            width: 40px;
            height: 40px;
            margin-top: 10px;
            opacity: 0.8;
        }}

        #error-message {{
            display: none;
            color: #c0392b;
            border: 1px solid #e74c3c;
            padding: 15px;
            margin-top: 20px;
            border-radius: 4px;
            background-color: #fbeae5;
            margin-bottom: 20px; /* Add margin below error */
        }}

        #results {{
            margin-top: 30px;
            display: none;
        }}

        /* DataTables Enhancements */
        .dataTables_wrapper {{
            overflow-x: auto;
            padding-top: 10px;
            padding-bottom: 10px; /* Add padding below table controls */
        }}
        table.dataTable {{
            width: 100% !important;
            border-collapse: collapse !important;
            margin: 0 auto;
            border: 1px solid #ddd; /* Add border around table */
        }}
        table.dataTable thead th {{
            background-color: #eef1f5;
            color: #333;
            font-weight: 600;
            border-bottom: 2px solid #ddd;
            text-align: left;
            padding: 10px 12px;
        }}
         table.dataTable tfoot th {{ /* Style for footer if used */
            background-color: #eef1f5;
            color: #333;
            font-weight: 600;
             border-top: 2px solid #ddd;
            text-align: left;
            padding: 10px 12px;
        }}
        table.dataTable tbody tr:nth-child(even) {{
            background-color: #f9fafb;
        }}
        table.dataTable tbody tr:hover {{
            background-color: #f0f5f9;
        }}
        table.dataTable tbody td {{
            padding: 8px 12px;
            border-bottom: 1px solid #ecf0f1;
            vertical-align: middle;
        }}
        /* Right-align numeric columns */
        table.dataTable td.dt-right, table.dataTable th.dt-right {{
            text-align: right;
        }}

        /* Style for the TOTAL row */
        table.dataTable tbody tr:last-child {{
            font-weight: bold;
            background-color: #e9ecef !important;
            color: #000;
        }}
        table.dataTable tbody tr:last-child td {{
             border-top: 2px solid #ccc;
        }}
        table.dataTable tbody tr:last-child td:first-child {{
            text-align: right;
        }}

        /* DataTables controls styling */
        .dataTables_length, .dataTables_filter, .dataTables_info, .dataTables_paginate {{
            margin-top: 15px;
            margin-bottom: 10px;
            padding: 0 5px;
            display: flex; /* Use flexbox for controls */
            align-items: center;
            flex-wrap: wrap; /* Allow wrapping on small screens */
        }}
         .dataTables_filter {{
            margin-left: auto; /* Push filter to the right */
         }}
         .dataTables_info {{
             margin-right: auto; /* Push info to the left of pagination */
         }}

        .dataTables_filter input {{
            margin-left: 5px;
            padding: 6px;
            border-radius: 3px;
            border: 1px solid #ccc;
            font-size: 14px;
        }}
        .dataTables_length select {{
             padding: 6px;
             border-radius: 3px;
             border: 1px solid #ccc;
             margin: 0 5px;
             width: auto;
             font-size: 14px;
        }}
         .dataTables_paginate .paginate_button {{
            padding: 5px 10px;
            margin: 0 2px;
            border: 1px solid #ddd;
            border-radius: 3px;
            cursor: pointer;
            background: #fff;
            color: #337ab7;
            text-decoration: none;
            transition: background-color 0.2s, color 0.2s;
        }}
        .dataTables_paginate .paginate_button.current,
        .dataTables_paginate .paginate_button:hover:not(.disabled) {{ /* Exclude disabled from hover */
            background: #337ab7;
            color: #fff !important; /* Use !important if necessary to override */
            border: 1px solid #337ab7;
        }}
        .dataTables_paginate .paginate_button.disabled {{
            background: #eee;
            color: #aaa !important;
            border: 1px solid #ddd;
            cursor: not-allowed;
        }}
         /* Add padding/margin around DataTables buttons */
         .dt-buttons {{
             margin-bottom: 10px;
             margin-left: 5px; /* Align buttons better */
         }}
        .dt-buttons .dt-button {{
             padding: 6px 12px; /* Adjust button padding */
             margin-right: 5px;
             border-radius: 4px;
             background-color: #6c757d; /* Bootstrap secondary button color */
             color: white;
             border: none;
             cursor: pointer;
             font-size: 13px;
             transition: background-color 0.2s;
         }}
         .dt-buttons .dt-button:hover {{
             background-color: #5a6268;
         }}


        /* --- End Embedded CSS --- */
    </style>
</head>
<body>
    <div class="container">
        <h1>OpenStack vs AWS Pricing Comparison</h1>

        <form id="pricing-form">
            <div class="form-row">
                <div class="form-group">
                    <label for="cloud">OpenStack Cloud Name:</label>
                    <input type="text" id="cloud" name="cloud" placeholder="e.g., my-cloud-config" required>
                     <small>Matches a section name in your clouds.yaml file.</small>
                </div>
                <div class="form-group">
                    <label for="project">Project Name or ID:</label>
                    <input type="text" id="project" name="project" placeholder="Leave blank if processing all projects">
                     <small>The name or ID of the OpenStack project.</small>
                </div>
                 <div class="form-group checkbox-group">
                    <input type="checkbox" id="all_projects" name="all_projects">
                    <label for="all_projects">Process All Projects</label>
                    <small>Overrides Project Name/ID if checked.</small>
                </div>
            </div>
            <div class="form-group">
                <label for="aws_csv">AWS Pricing CSV (EC2 - us-east-1):</label>
                <input type="file" id="aws_csv" name="aws_csv" accept=".csv" required>
                 <small>Download from AWS (e.g., Public Pricing page for EC2, US East (N. Virginia)).</small>
            </div>
            <button type="submit" id="submit-button">Calculate Costs</button>
        </form>

        <div id="loading">
            <p>Calculating... This may take a few minutes for large environments.</p>
            <!-- Simple spinner (CSS or SVG could also work) -->
            <img src="data:image/gif;base64,R0lGODlhEAAQAPIAAP///wAAAMLCwkJCQgAAAGJiYoKCgpKSkiH/C05FVFNDQVBFMi4wAwEAAAAh/hpDcmVhdGVkIHdpdGggYWpheGxvYWQuaW5mbwAh+QQJCgAAACwAAAAAEAAQAAADMwi63P4wyklrE2MIOggZnAdOmGYJRbExwroUmcG2LmDEwnHQLVsYOd2mBzkYDAdKa+dIAAAh+QQJCgAAACwAAAAAEAAQAAADNAi63P5OjCEgG4QMu7DmikRxQlFUYDEZIGBMRVsaqHwctXXf7WEYB4Ag1xjihkMZsiUkKhIAIfkECQoAAAAsAAAAABAAEAAAAzYIujIjK8pByJDMlFYvBoVjHA70GU7xSUJhmKtwHPAKzLO9HMaoKwJZ7Rf8AYPDDzKpZBqfvwQAIfkECQoAAAAsAAAAABAAEAAAAzMIumIlK8oyhpHsnFZfhYumCYUhDAQxRIdhHBGqRoKw0R8DYlJd8z0fMDgsGo/IpHI5TAAAIfkECQoAAAAsAAAAABAAEAAAAzIIunInK0rnZBTwGPNMgQwmdsNgXGJUlIWEuR5oWUIpz8pAEAMe6TwfwyYsGo/IpFKSAAAh+QQJCgAAACwAAAAAEAAQAAADMwi6IMKQORfjdOe82p4wGccc4CEuQradylesojEMBgsUc2G7sDX3lQGBMLAJibufbSlKAAAh+QQJCgAAACwAAAAAEAAQAAADMgi63P7wCRHZnFVdmgHu2nFwlWCI3WGc3TSWhUFGxTAUkGCbtgENBMJAEJsxgMLWzpEAACH5BAkKAAAALAAAAAAQABAAAAMyCLrc/jDKSatlQtScKdceCAjDII7HcQ4EMTCpyrCuUBjCYRgHVtqlAiB1YhiCnlsRkAAAOwAAAAAAAAAAAA==" alt="Loading..." />
        </div>
        <div id="error-message"></div>

        <div id="results">
            <h2>Results</h2>
            <!-- DataTables will render here -->
            <table id="results-table" class="display compact" style="width:100%">
                <thead>
                    <tr>
                        <!-- Column headers defined here must match JS column definitions -->
                        <th>Project</th>
                        <th>Server</th>
                        <th class="dt-right">vCPU</th>
                        <th class="dt-right">RAM (GiB)</th>
                        <th class="dt-right">Disk (GB)</th>
                        <th>AWS Type</th>
                        <th class="dt-right">OnDemand Hourly ($)</th>
                        <th class="dt-right">RI 3yr Hourly ($)</th>
                        <th class="dt-right">OnDemand Monthly ($)</th>
                        <th class="dt-right">OnDemand Yearly ($)</th>
                        <th class="dt-right">RI 3yr Monthly ($)</th>
                        <th class="dt-right">RI 3yr Yearly ($)</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- DataTables populates this -->
                </tbody>
                 <!-- Using tfoot can help with column filtering or summation if needed -->
                 <tfoot>
                     <tr>
                         <th>Project</th>
                         <th>Server</th>
                         <th>vCPU</th>
                         <th>RAM (GiB)</th>
                         <th>Disk (GB)</th>
                         <th>AWS Type</th>
                         <th>OnDemand Hourly ($)</th>
                         <th>RI 3yr Hourly ($)</th>
                         <th>OnDemand Monthly ($)</th>
                         <th>OnDemand Yearly ($)</th>
                         <th>RI 3yr Monthly ($)</th>
                         <th>RI 3yr Yearly ($)</th>
                    </tr>
                 </tfoot>
            </table>
        </div>
    </div>

    <!-- JavaScript Libraries from CDN -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <!-- Optional: DataTables Buttons extension (for CSV export etc.) -->
    <script src="https://cdn.datatables.net/buttons/2.4.1/js/dataTables.buttons.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/pdfmake.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/vfs_fonts.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.1/js/buttons.html5.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.1/js/buttons.print.min.js"></script>


    <script>
        // --- Embedded JavaScript ---
        $(document).ready(function() {
            let dataTableInstance = null; // To hold the DataTable instance

            // Function to format numbers to fixed decimal places or return 'N/A'
            // Handles null, undefined, NaN, and non-numeric strings gracefully.
            function formatNumber(num, places = 4, defaultValue = 'N/A') {
                 // Check if the input is strictly null or undefined
                 if (num === null || typeof num === 'undefined') {
                    return defaultValue;
                 }
                 // Attempt conversion to float
                 const number = parseFloat(num);
                 // Check if the conversion resulted in NaN
                 if (isNaN(number)) {
                    // Return default value if input wasn't a valid number representation
                    return defaultValue;
                 }
                 // Format the valid number
                 return number.toFixed(places);
            }

            // Define columns for DataTables (must match keys in JSON from backend)
            const tableColumns = [
                { data: 'Project', title: 'Project' },
                { data: 'Server', title: 'Server' },
                { data: 'vCPU', title: 'vCPU', className: 'dt-right' },
                { data: 'RAM_GiB', title: 'RAM (GiB)', className: 'dt-right' },
                { data: 'Disk_GB', title: 'Disk (GB)', className: 'dt-right' },
                { data: 'AWS_Type', title: 'AWS Type' },
                {
                    data: 'OnDemand_Hourly', title: 'OnDemand Hourly ($)', className: 'dt-right',
                    render: function(data, type, row) {
                        // 'display' type is for rendering, others (filter, sort) use raw data
                        return type === 'display' ? formatNumber(data, 6) : data; // 6 decimals for hourly
                    }
                },
                {
                    data: 'RI3yr_Hourly', title: 'RI 3yr Hourly ($)', className: 'dt-right',
                     render: function(data, type, row) {
                        return type === 'display' ? formatNumber(data, 6) : data;
                    }
                },
                {
                    data: 'OnDemand_Monthly', title: 'OnDemand Monthly ($)', className: 'dt-right',
                    render: function(data, type, row) {
                         return type === 'display' ? formatNumber(data, 2) : data; // 2 decimals for monthly/yearly
                    }
                },
                {
                    data: 'OnDemand_Yearly', title: 'OnDemand Yearly ($)', className: 'dt-right',
                    render: function(data, type, row) {
                        return type === 'display' ? formatNumber(data, 2) : data;
                    }
                },
                {
                    data: 'RI3yr_Monthly', title: 'RI 3yr Monthly ($)', className: 'dt-right',
                    render: function(data, type, row) {
                        return type === 'display' ? formatNumber(data, 2) : data;
                    }
                },
                {
                    data: 'RI3yr_Yearly', title: 'RI 3yr Yearly ($)', className: 'dt-right',
                    render: function(data, type, row) {
                        return type === 'display' ? formatNumber(data, 2) : data;
                    }
                }
            ];

            // Initialize DataTable with defined columns and options
            function initializeDataTable() {
                if (dataTableInstance) {
                    dataTableInstance.destroy(); // Destroy previous instance if exists
                    $('#results-table tbody').empty(); // Clear the table body manually
                    $('#results-table tfoot').empty(); // Clear footer as well if using DataTables footers
                }
                dataTableInstance = $('#results-table').DataTable({
                    columns: tableColumns,
                    data: [], // Start with empty data
                    pageLength: 25,
                    lengthMenu: [ [10, 25, 50, 100, -1], [10, 25, 50, 100, "All"] ],
                    order: [], // Initial no sorting
                    scrollX: true, // Enable horizontal scrolling
                    language: { // Customize text if needed - THIS WAS THE FIX
                        "emptyTable": "No data available in table",
                        "zeroRecords": "No matching records found",
                        "info": "Showing _START_ to _END_ of _TOTAL_ entries",
                        "infoEmpty": "Showing 0 entries",
                        "infoFiltered": "(filtered from _MAX_ total entries)"
                        // "loadingRecords": "Loading...", // Optional
                        // "processing": "Processing..." // Optional
                    },
                     dom: 'lBfrtip', // Layout: length, Buttons, filtering, table, info, pagination
                     buttons: [
                         { extend: 'copyHtml5', text: 'Copy' },
                         { extend: 'excelHtml5', text: 'Excel' },
                         { extend: 'csvHtml5', text: 'CSV' },
                         { extend: 'pdfHtml5', text: 'PDF' }
                     ],
                    // footerCallback: function ( row, data, start, end, display ) {
                    //     // Example: You could sum columns here if Python didn't provide a total row
                    //     // var api = this.api();
                    //     // $( api.column( 6 ).footer() ).html(
                    //     //     api.column( 6, { page: 'current'} ).data().sum() // Sum only current page
                    //     // );
                    // }
                });

                // Optional: Add column filters in the footer
                // $('#results-table tfoot th').each( function () {
                //     var title = $(this).text();
                //     if (title !== 'Project' && title !== 'Server' && !title.includes('$')) { // Add filters except for specific columns
                //         $(this).html( '<input type="text" placeholder="Filter '+title+'" />' );
                //     }
                // } );

                // // Apply the search
                // dataTableInstance.columns().every( function () {
                //     var that = this;
                //     $( 'input', this.footer() ).on( 'keyup change clear', function () {
                //         if ( that.search() !== this.value ) {
                //             that
                //                 .search( this.value )
                //                 .draw();
                //         }
                //     } );
                // } );

            }

            // Initialize the DataTable structure on page load
            initializeDataTable();
            $('#results').hide(); // Hide results section initially

            // Handle form submission
            $('#pricing-form').on('submit', function(event) {
                event.preventDefault(); // Prevent default page reload

                const formData = new FormData(this);
                const $submitButton = $('#submit-button');
                const $loadingIndicator = $('#loading');
                const $errorMessage = $('#error-message');
                const $resultsSection = $('#results');

                // Client-side validation
                const cloud = formData.get('cloud').trim();
                const project = formData.get('project').trim();
                const allProjects = $('#all_projects').is(':checked');
                const fileInput = $('#aws_csv')[0];
                const file = fileInput.files.length > 0 ? fileInput.files[0] : null;

                $errorMessage.hide().text(''); // Clear previous errors

                if (!cloud) {
                     $errorMessage.text('Error: Please enter the OpenStack Cloud name.').show();
                     return;
                }
                if (!allProjects && !project) {
                     $errorMessage.text('Error: Please enter a Project Name/ID or check "Process All Projects".').show();
                    return;
                }
                 if (!file) {
                    $errorMessage.text('Error: Please select an AWS Pricing CSV file.').show();
                    return;
                }
                 // Basic file extension check (more robust check happens server-side)
                 if (!file.name.toLowerCase().endsWith('.csv')) {
                     $errorMessage.text('Error: Invalid file type. Please upload a CSV file ending in .csv').show();
                     return;
                 }
                 // Optional: Check file size (matches Flask config)
                 if (file.size > 16 * 1024 * 1024) {
                    $errorMessage.text('Error: File size exceeds 16 MB limit.').show();
                    return;
                 }


                // Disable button, show loading, hide results
                $submitButton.prop('disabled', true).text('Calculating...');
                $loadingIndicator.show();
                $resultsSection.hide();

                // Clear previous results in DataTable before new request
                if (dataTableInstance) {
                    dataTableInstance.clear().draw();
                } else {
                    // Should already be initialized, but good safeguard
                    initializeDataTable();
                }

                // Make AJAX request to Flask backend
                $.ajax({
                    url: '/api/calculate',
                    type: 'POST',
                    data: formData,
                    processData: false, // Important: prevent jQuery from processing FormData
                    contentType: false, // Important: let browser set Content-Type with boundary
                    dataType: 'json',   // Expect JSON response from Flask
                    success: function(response) {
                        $loadingIndicator.hide();
                        $submitButton.prop('disabled', false).text('Calculate Costs');

                        if (response.data && Array.isArray(response.data)) {
                            console.log("Data received:", response.data.length, "rows");
                            if (dataTableInstance) {
                                dataTableInstance.rows.add(response.data); // Add new data
                                dataTableInstance.columns.adjust().draw(); // Adjust column widths and redraw
                                $resultsSection.show(); // Show the results section

                                // Check if any server data was returned (excluding the TOTAL row)
                                const serverRows = response.data.filter(row => row.Server !== 'TOTAL');
                                if (serverRows.length === 0) {
                                    // If only the TOTAL row (or empty), display a message
                                    $errorMessage.text('Info: No OpenStack servers found matching the criteria.').show();
                                     $('#results-table').hide(); // Hide the table if no data
                                } else {
                                    $('#results-table').show(); // Ensure table is visible if there's data
                                }


                            } else {
                                console.error("DataTable instance is not available.");
                                $errorMessage.text('Error: Could not display results table.').show();
                            }
                        } else if (response.error) {
                             console.error("Backend Error:", response.error);
                             $errorMessage.text('Error: ' + response.error).show();
                        } else {
                             console.error("Unexpected response format:", response);
                             $errorMessage.text('Error: Received unexpected data format from server. Check console.').show();
                        }
                    },
                    error: function(jqXHR, textStatus, errorThrown) {
                        console.error("AJAX Error:", textStatus, errorThrown, jqXHR.responseText);
                        $loadingIndicator.hide();
                        $submitButton.prop('disabled', false).text('Calculate Costs');

                        let errorMsg = 'An unknown error occurred during calculation.';
                        if (jqXHR.responseJSON && jqXHR.responseJSON.error) {
                            errorMsg = jqXHR.responseJSON.error; // Error message from Flask JSON response
                        } else if (jqXHR.status === 413) {
                            errorMsg = 'Error: The uploaded file is too large (max 16MB).';
                        } else if (jqXHR.responseText) {
                            // Attempt to parse JSON even if status code isn't standard success
                            try {
                                const jsonResponse = JSON.parse(jqXHR.responseText);
                                if (jsonResponse.error) {
                                    errorMsg = jsonResponse.error;
                                } else {
                                     // Fallback to text if JSON wasn't expected format
                                    if (jqXHR.responseText.length < 500 && !jqXHR.responseText.trim().startsWith('<')) {
                                        errorMsg = `Server Error (${jqXHR.status}): ${jqXHR.responseText}`;
                                    } else {
                                         errorMsg = `Server Error: ${jqXHR.status} ${errorThrown}. Check server logs for details.`;
                                    }
                                }
                            } catch (e) {
                                // If responseText isn't JSON or is too long/looks like HTML
                                if (jqXHR.responseText.length < 500 && !jqXHR.responseText.trim().startsWith('<')) {
                                        errorMsg = `Server Error (${jqXHR.status}): ${jqXHR.responseText}`;
                                } else {
                                     errorMsg = `Server Error: ${jqXHR.status} ${errorThrown}. Check server logs for details.`;
                                }
                            }
                        } else {
                            errorMsg = `Network Error: ${textStatus} - ${errorThrown}`;
                        }
                        $errorMessage.text(errorMsg).show();
                    }
                });
            });

            // Logic for Project Name / All Projects checkbox interaction
            $('#all_projects').on('change', function() {
                const isChecked = $(this).is(':checked');
                $('#project').prop('disabled', isChecked);
                if (isChecked) {
                    $('#project').val(''); // Clear project field when disabled
                }
            });
            // Trigger the change handler on load to set initial state
            $('#all_projects').trigger('change');

        });
        // --- End Embedded JavaScript ---
    </script>

</body>
</html>
"""

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main page with embedded HTML, CSS, and JS."""
    # No need for a return statement here, render_template_string handles it
    return render_template_string(HTML_TEMPLATE)

@app.route('/favicon.ico')
def favicon():
     # Simple 204 No Content response for favicon requests to avoid 404s
     # Or you could embed a base64 favicon in the HTML head <link> tag
     return '', 204


@app.route('/api/calculate', methods=['POST'])
def calculate_costs():
    """API endpoint to trigger the calculation."""
    if 'aws_csv' not in request.files:
        return jsonify({"error": "No AWS CSV file part in the request."}), 400
    file = request.files['aws_csv']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    if not file or not _allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type for '{file.filename}'. Please upload a CSV file."}), 400

    cloud_name = request.form.get('cloud', '').strip()
    project_name = request.form.get('project', '').strip() # Might be empty
    # Checkbox value is 'on' if checked, otherwise None/missing
    use_all_projects = request.form.get('all_projects') == 'on'

    if not cloud_name:
        return jsonify({"error": "OpenStack cloud name is required."}), 400
    if not use_all_projects and not project_name:
        return jsonify({"error": "Either Project Name/ID or 'Process All Projects' must be checked."}), 400
    if use_all_projects and project_name:
        # Prioritize 'all_projects' if both are somehow submitted
        project_name = None
        logging.info("Both project and 'All Projects' provided. Using 'All Projects'.")
        project_id_to_use = None
    elif use_all_projects:
        project_id_to_use = None # Explicitly None for clarity
        logging.info("Processing for all projects.")
    else:
        # Need to resolve project_name to project_id
        project_id_to_use = None # Initialize

    # --- OpenStack Connection and Project Resolution ---
    conn = None # Initialize connection variable
    try:
        logging.info(f"Attempting connection to cloud: {cloud_name}")
        conn = connection.from_config(cloud=cloud_name)
        # Minimal check to verify connection works
        try:
             conn.identity.get_user('self')
             logging.info(f"Successfully connected to OpenStack cloud: {cloud_name}")
        except Exception as e:
             # Catch exceptions during the get_user call specifically
             logging.error(f"Authentication/Initial connection check failed for cloud '{cloud_name}': {e}")
             raise OpenStackCloudException(f"Authentication or initial connection check failed for cloud '{cloud_name}'. Check credentials and configuration.", e)

        # Resolve Project ID if a specific project was requested
        if not use_all_projects and project_name:
            logging.info(f"Resolving project identifier: '{project_name}'")
            try:
                # find_project tries name first, then ID if name fails
                pr = conn.identity.find_project(project_name, ignore_missing=False) # Fail if not found
                if not pr: # find_project with ignore_missing=False should raise, but check explicitly
                     raise ResourceNotFound(f"Project '{project_name}' not found.")
                project_id_to_use = pr.id
                logging.info(f"Found project '{pr.name}' with ID: {project_id_to_use}")
            except ResourceNotFound:
                 return jsonify({"error": f"OpenStack project '{project_name}' not found or inaccessible. Verify name/ID and permissions."}), 404 # 404 Not Found
            except OpenStackCloudException as e:
                logging.error(f"Failed to find project '{project_name}': {e}")
                return jsonify({"error": f"Error finding OpenStack project '{project_name}'. Details: {e}"}), 500
            except Exception as e_find: # Catch broader errors during find
                 logging.error(f"Unexpected error resolving project '{project_name}': {e_find}")
                 return jsonify({"error": f"Unexpected error resolving project '{project_name}'. Details: {e_find}"}), 500

    except OpenStackCloudException as e:
        logging.error(f"Failed to connect to OpenStack cloud '{cloud_name}': {e}")
        return jsonify({"error": f"Failed to connect to OpenStack cloud '{cloud_name}'. Check cloud config and credentials. Details: {e}"}), 503 # Service Unavailable or similar
    except Exception as e: # Catch other potential connection errors (e.g., config file issues)
        logging.error(f"Unexpected error setting up OpenStack connection '{cloud_name}': {e}")
        return jsonify({"error": f"Unexpected error connecting to OpenStack cloud '{cloud_name}'. Details: {e}"}), 500
    # --- End Connection Handling ---


    # --- Perform Calculation ---
    try:
        # Reset the stream position to the beginning before passing it
        file.stream.seek(0)
        # Pass the file stream directly (file.stream is io.BytesIO)
        report_df = build_report(conn, project_id_to_use, file.stream)

        # Convert DataFrame to JSON suitable for DataTables (list of objects)
        # Handle NaN/None -> None (null in JSON) for better JS handling
        report_df_clean = report_df.where(pd.notnull(report_df), None)

        json_data = report_df_clean.to_dict(orient='records')

        return jsonify({"data": json_data})

    except (ValueError, ConnectionError, RuntimeError) as e:
         # Catch specific errors raised by build_report or helpers
         logging.error(f"Calculation failed: {e}\n{traceback.format_exc()}")
         # Use 400 for user input/data errors, 500/503 for internal/connection issues
         status_code = 503 if isinstance(e, ConnectionError) else 400
         return jsonify({"error": f"Calculation Error: {str(e)}"}), status_code
    except Exception as e:
         # Catch any other unexpected errors during the main process
         logging.error(f"An unexpected error occurred during calculation: {e}\n{traceback.format_exc()}")
         return jsonify({"error": "An internal server error occurred during calculation."}), 500
    finally:
        # Although SDK might handle cleanup, ensure connection is closed if possible
        if 'conn' in locals() and conn is not None:
            try:
                # Depending on SDK version/config, close might be needed
                # If conn doesn't have a close method or it causes issues, remove this
                if hasattr(conn, 'close'):
                    conn.close()
                    logging.debug("OpenStack connection closed.")
            except Exception:
                logging.warning("Exception occurred while trying to close OpenStack connection.")

# --- Main Execution ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to be accessible on the network
    # Use debug=False for production/sharing
    # Choose a port (e.g., 5001 or another free port)
    # The WARNING about the development server will still appear in debug=False,
    # but it's a standard Flask message for this basic run method.
    app.run(host='0.0.0.0', port=5001, debug=False)
