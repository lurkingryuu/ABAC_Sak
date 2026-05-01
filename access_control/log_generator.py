"""
Log Generator Module for ABAC Systems

Generates synthetic access logs based on the ACM (Access Control Matrix).
Takes as input:
- Number of logs to generate
- Allow percentage (0-99.9%)

Outputs a CSV file with columns: subject, object, environment, access
"""

import os
import json
import csv
import random
import numpy as np
from pathlib import Path


def read_acm(acm_path):
    """
    Read the ACM.txt file and convert it to a 3D matrix.
    
    ACM format:
    - Each row is a space-separated list of 0s and 1s
    - n1 subjects, n2 objects, n3 environments
    - Layout: A[0][0][:] (all environments for subject 0, object 0)
             A[0][1][:] (all environments for subject 0, object 1)
             ... blank line between subjects
    
    Returns:
    - A: 3D ACM matrix where A[i][j][k] = 1 (permit) or 0 (deny)
    - n1, n2, n3: dimensions
    """
    acm_data = []
    
    with open(acm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                acm_data.extend(map(int, line.split()))
    
    # Determine dimensions by analyzing the structure
    # We need to figure out n1, n2, n3 from the flattened ACM
    # The ACM has n1 * n2 rows, each with n3 columns
    total_elements = len(acm_data)
    # We need to infer n1, n2, n3 such that n1 * n2 * n3 = total_elements
    
    # Try to find reasonable dimensions
    # For now, we'll read the original config or infer from file
    n3 = None
    n2 = None
    n1 = None
    
    # Re-read the file to count rows properly
    with open(acm_path, 'r') as f:
        content = f.read()
        rows = [line.strip() for line in content.split('\n') if line.strip()]
    
    if rows:
        # First row tells us n3 (number of environments)
        n3 = len(rows[0].split())
        
        # Count how many consecutive non-empty rows we have
        # Structure: n1 groups of n2 rows each
        # Each group has n2 rows (for n2 objects)
        # Separated by blank lines
        
        # Simple approach: divide total rows by groups
        # Assuming blank lines separate subject groups
        with open(acm_path, 'r') as f:
            all_lines = f.readlines()
        
        non_empty_rows = sum(1 for line in all_lines if line.strip())
        # non_empty_rows = n1 * n2 (each subject has n2 objects, each object has one row in the flattened view)
        
        # We need to read from config.ini to get n1, n2, n3
        # For now, infer: assume we can deduce from structure
        # Actually, let's check from the calling context
    
    # Flatten the data and return with inferred dimensions
    return acm_data, total_elements  # Return as flat array; caller will reshape


def load_config_for_dimensions(config_path):
    """
    Load n1, n2, n3 from config.ini
    
    Returns: (n1, n2, n3)
    """
    import configparser
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    n1 = int(config["NUMBERS"]["n1"])
    n2 = int(config["NUMBERS"]["n2"])
    n3 = int(config["NUMBERS"]["n3"])
    
    return n1, n2, n3


def read_acm_properly(acm_path, n1, n2, n3):
    """
    Read ACM.txt and reshape into proper 3D matrix A[i][j][k]
    where k is the environment dimension.
    
    ACM.txt format:
    - n1 subject blocks separated by blank lines
    - Each block has n2 rows (one per object)
    - Each row has n3 values (one per environment)
    """
    A = [[[0] * n3 for _ in range(n2)] for _ in range(n1)]
    
    with open(acm_path, 'r') as f:
        i = 0  # subject index
        j = 0  # object index
        
        for line in f:
            line = line.strip()
            if not line:  # blank line separates subjects
                j = 0  # reset object index
                i += 1
                continue
            
            if i < n1 and j < n2:
                values = list(map(int, line.split()))
                for k in range(min(n3, len(values))):
                    A[i][j][k] = values[k]
                j += 1
    
    return A


def generate_logs(num_logs, allow_percentage, A, n1, n2, n3):
    """
    Generate synthetic access logs based on the ACM.
    
    Strategy:
    1. Calculate how many logs should be PERMIT vs DENY based on allow_percentage
    2. Randomly sample (subject, object, environment) tuples from the ACM
    3. For PERMIT logs: only select tuples where A[i][j][k] = 1
    4. For DENY logs: only select tuples where A[i][j][k] = 0
    5. Return as list of dicts with keys: subject, object, environment, access
    
    Args:
        num_logs: Total number of logs to generate
        allow_percentage: Percentage of logs that should be PERMIT (0-99.9)
        A: 3D ACM matrix
        n1, n2, n3: dimensions
    
    Returns:
        List of log dicts: [{"subject": "S_1", "object": "O_1", "environment": "E_1", "access": "permit"}, ...]
    """
    if allow_percentage < 0 or allow_percentage >= 100:
        raise ValueError(f"allow_percentage must be in range [0, 100). Got {allow_percentage}")
    
    # Calculate split
    num_permit = int(num_logs * allow_percentage / 100.0)
    num_deny = num_logs - num_permit
    
    logs = []
    
    # Find all (i, j, k) tuples where A[i][j][k] = 1 (permit) and 0 (deny)
    permit_tuples = []
    deny_tuples = []
    
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if A[i][j][k] == 1:
                    permit_tuples.append((i, j, k))
                else:
                    deny_tuples.append((i, j, k))
    
    # Check if we have enough tuples
    if not permit_tuples and num_permit > 0:
        raise ValueError(f"Cannot generate {num_permit} PERMIT logs: no PERMIT tuples in ACM")
    if not deny_tuples and num_deny > 0:
        raise ValueError(f"Cannot generate {num_deny} DENY logs: no DENY tuples in ACM")
    
    # Sample with replacement
    sampled_permit = random.choices(permit_tuples, k=num_permit) if permit_tuples else []
    sampled_deny = random.choices(deny_tuples, k=num_deny) if deny_tuples else []
    
    # Create log entries
    for i, j, k in sampled_permit:
        logs.append({
            "subject": f"S_{i+1}",
            "object": f"O_{j+1}",
            "environment": f"E_{k+1}",
            "access": "permit"
        })
    
    for i, j, k in sampled_deny:
        logs.append({
            "subject": f"S_{i+1}",
            "object": f"O_{j+1}",
            "environment": f"E_{k+1}",
            "access": "deny"
        })
    
    # Shuffle to mix permit and deny logs
    random.shuffle(logs)
    
    return logs


def write_logs_to_csv(logs, output_path):
    """
    Write logs to a CSV file with columns: subject, object, environment, access
    
    Args:
        logs: List of log dicts
        output_path: Path to output CSV file
    """
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['subject', 'object', 'environment', 'access']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(logs)


def generate_and_save_logs(config_path, acm_path, num_logs, allow_percentage, output_path):
    """
    Main function to generate logs and save to CSV.
    
    Args:
        config_path: Path to config.ini
        acm_path: Path to ACM.txt
        num_logs: Number of logs to generate
        allow_percentage: Percentage of PERMIT logs (0-99.9)
        output_path: Path to save CSV file
    
    Raises:
        ValueError: If allow_percentage is invalid or files are missing
    """
    # Validate allow_percentage
    if not isinstance(allow_percentage, (int, float)):
        raise ValueError(f"allow_percentage must be a number, got {type(allow_percentage)}")
    if allow_percentage < 0 or allow_percentage >= 100:
        raise ValueError(f"allow_percentage must be in range [0, 100). Got {allow_percentage}")
    
    # Check files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.ini not found at {config_path}")
    if not os.path.exists(acm_path):
        raise FileNotFoundError(f"ACM.txt not found at {acm_path}")
    
    # Load dimensions from config
    print(f"Loading config from {config_path}")
    n1, n2, n3 = load_config_for_dimensions(config_path)
    print(f"Dimensions: n1={n1} (subjects), n2={n2} (objects), n3={n3} (environments)")
    
    # Read ACM
    print(f"Reading ACM from {acm_path}")
    A = read_acm_properly(acm_path, n1, n2, n3)
    print(f"ACM loaded successfully")
    
    # Generate logs
    print(f"Generating {num_logs} logs with {allow_percentage}% permit rate")
    logs = generate_logs(num_logs, allow_percentage, A, n1, n2, n3)
    
    # Write to CSV
    print(f"Writing logs to {output_path}")
    write_logs_to_csv(logs, output_path)
    
    print(f"✓ Log generation complete: {len(logs)} logs written to {output_path}")
    
    # Print summary
    permit_count = sum(1 for log in logs if log['access'] == 'permit')
    deny_count = sum(1 for log in logs if log['access'] == 'deny')
    actual_permit_pct = (permit_count / len(logs) * 100) if logs else 0
    
    print(f"\nSummary:")
    print(f"  Total logs: {len(logs)}")
    print(f"  Permit: {permit_count} ({actual_permit_pct:.1f}%)")
    print(f"  Deny: {deny_count} ({100-actual_permit_pct:.1f}%)")
    
    return logs


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python log_generator.py <config.ini> <acm.txt> <num_logs> <allow_percentage>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    acm_path = sys.argv[2]
    num_logs = int(sys.argv[3])
    allow_percentage = float(sys.argv[4])
    
    # Output path: same directory as ACM.txt
    output_dir = os.path.dirname(acm_path)
    output_path = os.path.join(output_dir, "logs.csv")
    
    try:
        generate_and_save_logs(config_path, acm_path, num_logs, allow_percentage, output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
