import os
import csv
import json
from datetime import datetime, date

# Use absolute path relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIOLATIONS_FILE = os.path.join(PROJECT_ROOT, "violations.json")

def ensure_violations_file_exists():
    """Ensure violations.json exists and contains valid JSON."""
    if not os.path.exists(VIOLATIONS_FILE) or os.path.getsize(VIOLATIONS_FILE) == 0:
        with open(VIOLATIONS_FILE, 'w') as file:
            json.dump([], file)
        print(f"[DB] Initialized {VIOLATIONS_FILE}")


def log_violation(vehicle_number, violation_type, violation_image_path):
    """Log a violation to violations.json. Returns the violation dict if new, None if duplicate."""
    ensure_violations_file_exists()
    
    # Read existing violations
    with open(VIOLATIONS_FILE, 'r') as file:
        try:
            violations = json.load(file)
        except json.JSONDecodeError:
            print(f"[DB] WARNING: violations.json was corrupted, resetting.")
            violations = []
    
    # Check for existing violation today
    today = datetime.now().date()
    existing_violation = next((v for v in violations 
                                if v['vehicle_number'] == vehicle_number 
                                and datetime.strptime(v['timestamp'], "%Y-%m-%d %H:%M:%S").date() == today), 
                               None)
    
    # If no violation exists today, add new violation
    if not existing_violation:
        violation = {
            'vehicle_number': vehicle_number,
            'violation_type': violation_type,
            'violation_image': violation_image_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        violations.append(violation)
        
        # Write back to file
        with open(VIOLATIONS_FILE, 'w') as file:
            json.dump(violations, file, indent=2)
        
        print(f"[DB] Violation saved to {VIOLATIONS_FILE}: {vehicle_number}")
        return violation
    
    print(f"[DB] Duplicate for today, not saving: {vehicle_number}")
    return None


def read_violations_by_vehicle(vehicle_number):
    ensure_violations_file_exists()
    
    with open(VIOLATIONS_FILE, 'r') as file:
        try:
            violations = json.load(file)
        except json.JSONDecodeError:
            violations = []
    
    return [v for v in violations if v['vehicle_number'] == vehicle_number]