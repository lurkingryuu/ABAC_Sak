import json
import os
import re
import time
from typing import Dict, List, Any
from dotenv import load_dotenv
import google.generativeai as genai
from faker import Faker

load_dotenv()

def extract_attribute_counts(raw_values: List) -> List[int]:
    """
    Parse attribute values that can be:
    - A single number: e.g., 5 (means 5 values)
    - A pair: e.g., [5, 2] (means 5 values, 2 stars)
    
    Returns: list of just the count values (integers)
    
    Example:
        [6, [5, 2], 4, [3, 1]] -> [6, 5, 4, 3]
    """
    counts = []
    for item in raw_values:
        if isinstance(item, list):
            # Pair: [num_values, num_stars] - extract first element
            counts.append(item[0])
        else:
            # Single number
            counts.append(item)
    return counts

# Initialize Faker
fake = Faker()

# Configure Gemini API
def initialize_gemini(api_key: str = None):
    """Initialize Gemini API with your API key."""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

def extract_json_from_response(text: str) -> Dict:
    """Extract JSON from response text with better error handling."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"No valid JSON found in response: {text[:200]}")

def generate_healthcare_subjects(subject_size: int) -> List[str]:
    """Generate healthcare worker names using Faker library."""
    print(f"Generating {subject_size} healthcare worker names using Faker...")
    subjects = []
    roles = ["Dr.", "Nurse", "Tech", "Admin", "Specialist", "Consultant"]
    
    for i in range(subject_size):
        # Generate a name and add a role prefix
        name = fake.name()
        role = roles[i % len(roles)]
        subjects.append(f"{role} {name}")
    
    return subjects

def generate_all_entities(model, subject_size: int, object_size: int, 
                          environment_size: int) -> tuple:
    """Generate entities: subjects using Faker, objects and environments using Gemini."""
    
    # Generate subjects using Faker
    subjects = generate_healthcare_subjects(subject_size)
    
    # Generate objects and environments using Gemini
    prompt = f"""You MUST return ONLY a valid JSON object. No other text.

Generate exactly {object_size} medical documents and {environment_size} hospital locations.

For objects (medical documents): Use names like "PatientChart_ICU_001", "LabResult_Blood_Panel", "SurgicalNote_OR_2024", "Prescription_Pharmacy_001"
For environments (hospital locations): Use names like "ICU_Ward_A", "Emergency_Department", "Operating_Room_3", "Cardiology_Clinic", "Laboratory_Building_1"

Return this exact JSON structure:
{{
    "objects": ["PatientChart_ICU_001", "LabResult_Blood_Panel", ...],
    "environments": ["ICU_Ward_A", "Emergency_Department", ...]
}}

Generate exactly {object_size} objects and {environment_size} environments:"""
    
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    try:
        data = extract_json_from_response(text)
        return (
            subjects,
            data.get("objects", [])[:object_size],
            data.get("environments", [])[:environment_size]
        )
    except Exception as e:
        print(f"[ERROR] Entity parsing failed: {e}")
        return (
            subjects,
            [f"MedicalDocument_{i+1}" for i in range(object_size)],
            [f"Facility_{i+1}" for i in range(environment_size)]
        )

def generate_all_attributes(model, subject_attr_count: int, object_attr_count: int,
                           environment_attr_count: int) -> tuple:
    """Generate realistic healthcare attribute names."""
    prompt = f"""You MUST return ONLY a valid JSON object. No other text.

Generate {subject_attr_count} healthcare worker attributes, {object_attr_count} medical document attributes, and {environment_attr_count} facility attributes.

Subject attributes examples: role, department, clearance_level, certification_status, specialization, access_zone, employment_type
Object attributes examples: data_type, sensitivity_level, patient_category, record_status, encryption_standard, compliance_requirement
Environment attributes examples: location_type, network_security, access_control, physical_security, audit_level, isolation_level

Return this exact JSON structure:
{{
    "subject_attributes": ["role", "department", "clearance_level", ...],
    "object_attributes": ["data_type", "sensitivity_level", "patient_category", ...],
    "environment_attributes": ["location_type", "network_security", "access_control", ...]
}}

Generate exactly {subject_attr_count} subject attributes, {object_attr_count} object attributes, {environment_attr_count} environment attributes:"""
    
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    try:
        data = extract_json_from_response(text)
        return (
            data.get("subject_attributes", [])[:subject_attr_count],
            data.get("object_attributes", [])[:object_attr_count],
            data.get("environment_attributes", [])[:environment_attr_count]
        )
    except Exception as e:
        print(f"[ERROR] Attribute parsing failed: {e}")
        return (
            [f"subject_attr_{i+1}" for i in range(subject_attr_count)],
            [f"object_attr_{i+1}" for i in range(object_attr_count)],
            [f"environment_attr_{i+1}" for i in range(environment_attr_count)]
        )

def generate_all_attribute_values(model, subject_attrs: List[str], object_attrs: List[str],
                                 environment_attrs: List[str], subject_values: List[int],
                                 object_values: List[int], 
                                 environment_values: List[int]) -> tuple:
    """Generate realistic healthcare attribute values in ONE API call."""
    
    subject_spec = "\n".join([f"  {i+1}. {attr} ({count} values)" 
                             for i, (attr, count) in enumerate(zip(subject_attrs, subject_values))])
    object_spec = "\n".join([f"  {i+1}. {attr} ({count} values)" 
                            for i, (attr, count) in enumerate(zip(object_attrs, object_values))])
    environment_spec = "\n".join([f"  {i+1}. {attr} ({count} values)" 
                                 for i, (attr, count) in enumerate(zip(environment_attrs, environment_values))])
    
    prompt = f"""You MUST return ONLY a valid JSON object. No other text.

Generate realistic healthcare values for these attributes:

HEALTHCARE WORKER ATTRIBUTES (Subject):
{subject_spec}

MEDICAL DOCUMENT ATTRIBUTES (Object):
{object_spec}

FACILITY ATTRIBUTES (Environment):
{environment_spec}

Return this exact JSON structure with arrays of strings as values:
{{
    "subject_attribute_values": {{
        "{subject_attrs[0]}": ["value1", "value2", ...],
        "{subject_attrs[1]}": ["value1", "value2", ...],
        ...
    }},
    "object_attribute_values": {{
        "{object_attrs[0]}": ["value1", "value2", ...],
        ...
    }},
    "environment_attribute_values": {{
        "{environment_attrs[0]}": ["value1", "value2", ...],
        ...
    }}
}}

IMPORTANT:
- Each value array must have EXACTLY the specified number of values
- All values must be realistic for healthcare domain
- Use lowercase with underscores for multi-word values
- No duplicates within each attribute's values
- Return ONLY the JSON object, no other text"""
    
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    try:
        data = extract_json_from_response(text)
        
        subject_av = data.get("subject_attribute_values", {})
        object_av = data.get("object_attribute_values", {})
        environment_av = data.get("environment_attribute_values", {})
        
        # Validate and fix counts
        for attr, count in zip(subject_attrs, subject_values):
            if attr not in subject_av or not isinstance(subject_av[attr], list):
                subject_av[attr] = [f"{attr}_value_{i+1}" for i in range(count)]
            elif len(subject_av[attr]) < count:
                subject_av[attr] += [f"{attr}_value_{len(subject_av[attr])+i+1}" 
                                    for i in range(count - len(subject_av[attr]))]
            else:
                subject_av[attr] = subject_av[attr][:count]
        
        for attr, count in zip(object_attrs, object_values):
            if attr not in object_av or not isinstance(object_av[attr], list):
                object_av[attr] = [f"{attr}_value_{i+1}" for i in range(count)]
            elif len(object_av[attr]) < count:
                object_av[attr] += [f"{attr}_value_{len(object_av[attr])+i+1}" 
                                   for i in range(count - len(object_av[attr]))]
            else:
                object_av[attr] = object_av[attr][:count]
        
        for attr, count in zip(environment_attrs, environment_values):
            if attr not in environment_av or not isinstance(environment_av[attr], list):
                environment_av[attr] = [f"{attr}_value_{i+1}" for i in range(count)]
            elif len(environment_av[attr]) < count:
                environment_av[attr] += [f"{attr}_value_{len(environment_av[attr])+i+1}" 
                                        for i in range(count - len(environment_av[attr]))]
            else:
                environment_av[attr] = environment_av[attr][:count]
        
        return (subject_av, object_av, environment_av)
    except Exception as e:
        print(f"[ERROR] Values parsing failed: {e}")
        
        subject_av = {attr: [f"{attr}_value_{i+1}" for i in range(count)] 
                      for attr, count in zip(subject_attrs, subject_values)}
        object_av = {attr: [f"{attr}_value_{i+1}" for i in range(count)] 
                     for attr, count in zip(object_attrs, object_values)}
        environment_av = {attr: [f"{attr}_value_{i+1}" for i in range(count)] 
                          for attr, count in zip(environment_attrs, environment_values)}
        
        return (subject_av, object_av, environment_av)

def generate_healthcare_data_with_gemini(config_ini_path: str = None,
                                        api_key: str = None) -> Dict[str, Any]:
    """
    Generate complete healthcare dataset using Gemini API.
    
    Args:
        config_ini_path: Path to config.ini file (or directory containing it).
                        If None, uses ABAC_CONFIG_INI env var or default location.
        api_key: Optional API key override.
    
    Returns healthcare data in original format (names, attributes, values).
    """
    
    # Resolve config.ini path
    if config_ini_path is None:
        config_ini_path = os.environ.get("ABAC_CONFIG_INI") or os.path.join(os.path.dirname(__file__), 'config.ini')
    
    # If a directory was passed, append config.ini
    if os.path.isdir(config_ini_path):
        config_ini_path = os.path.join(config_ini_path, 'config.ini')
    
    # Resolve input.json path relative to config.ini location
    config_dir = os.path.dirname(config_ini_path)
    input_json_path = os.path.join(config_dir, 'uploads', 'input.json')

    # Load input configuration
    with open(input_json_path, 'r') as f:
        config = json.load(f)
    
    # Initialize Gemini
    print("Initializing Gemini API...")
    model = initialize_gemini(api_key)
    
    healthcare_data = {}
    
    # ==================== STEP 1: Generate All Entities ====================
    print("\n" + "="*60)
    print("STEP 1: Generating Healthcare Entities")
    print("="*60)
    print(f"Generating {config['subject_size']} subjects, {config['object_size']} objects, {config['environment_size']} environments...")
    
    subjects, objects, environments = generate_all_entities(
        model,
        config['subject_size'],
        config['object_size'],
        config['environment_size']
    )
    
    healthcare_data['subjects'] = subjects
    healthcare_data['objects'] = objects
    healthcare_data['environments'] = environments
    
    print(f"  ✓ Subjects: {subjects}")
    print(f"  ✓ Objects: {objects}")
    print(f"  ✓ Environments: {environments}")
    
    time.sleep(2)
    
    # ==================== STEP 2: Generate All Attributes ====================
    print("\n" + "="*60)
    print("STEP 2: Generating Attribute Names")
    print("="*60)
    print(f"Generating {config['subject_attributes_count']} subject, {config['object_attributes_count']} object, {config['environment_attributes_count']} environment attributes...")
    
    subject_attributes, object_attributes, environment_attributes = generate_all_attributes(
        model,
        config['subject_attributes_count'],
        config['object_attributes_count'],
        config['environment_attributes_count']
    )
    
    healthcare_data['subject_attributes'] = subject_attributes
    healthcare_data['object_attributes'] = object_attributes
    healthcare_data['environment_attributes'] = environment_attributes
    
    print(f"  ✓ Subject attributes: {subject_attributes}")
    print(f"  ✓ Object attributes: {object_attributes}")
    print(f"  ✓ Environment attributes: {environment_attributes}")
    
    time.sleep(2)
    
    # ==================== STEP 3: Generate All Attribute Values ====================
    print("\n" + "="*60)
    print("STEP 3: Generating Attribute Values")
    print("="*60)
    print("Generating all attribute values...")
    
    # Extract just the counts from mixed format (single number or [number, stars] pair)
    subject_counts = extract_attribute_counts(config['subject_attributes_values'])
    object_counts = extract_attribute_counts(config['object_attributes_values'])
    environment_counts = extract_attribute_counts(config['environment_attributes_values'])
    
    subject_attribute_values, object_attribute_values, environment_attribute_values = generate_all_attribute_values(
        model,
        subject_attributes,
        object_attributes,
        environment_attributes,
        subject_counts,
        object_counts,
        environment_counts
    )
    
    healthcare_data['subject_attribute_values'] = subject_attribute_values
    healthcare_data['object_attribute_values'] = object_attribute_values
    healthcare_data['environment_attribute_values'] = environment_attribute_values
    
    print(f"  ✓ Subject attribute values generated")
    print(f"  ✓ Object attribute values generated")
    print(f"  ✓ Environment attribute values generated")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"✓ Subjects: {len(subjects)}")
    print(f"✓ Objects: {len(objects)}")
    print(f"✓ Environments: {len(environments)}")
    print(f"✓ Subject Attributes: {len(subject_attributes)}")
    print(f"✓ Object Attributes: {len(object_attributes)}")
    print(f"✓ Environment Attributes: {len(environment_attributes)}")
    print("="*60)
    
    return healthcare_data

def convert_to_indexed_format(healthcare_data: Dict, SV: Dict, OV: Dict, EV: Dict,
                             subject_count: int, object_count: int, 
                             environment_count: int) -> Dict[str, Any]:
    """
    Convert healthcare data to actual names and values format.
    
    This function:
    1. Uses actual healthcare entity names (not indexed)
    2. Uses actual attribute names from healthcare data
    3. Maps actual attribute values based on original SV, OV, EV patterns
    """
    
    output = {}
    
    # Extract data from healthcare_data
    subjects = healthcare_data['subjects']
    objects = healthcare_data['objects']
    environments = healthcare_data['environments']
    
    subject_attrs = healthcare_data['subject_attributes']
    object_attrs = healthcare_data['object_attributes']
    environment_attrs = healthcare_data['environment_attributes']
    
    subject_av = healthcare_data['subject_attribute_values']
    object_av = healthcare_data['object_attribute_values']
    environment_av = healthcare_data['environment_attribute_values']
    
    # Step 1: Use actual entity names (NOT indexed)
    output["S_HC"] = subjects[:subject_count]
    output["O_HC"] = objects[:object_count]
    output["E_HC"] = environments[:environment_count]
    
    # Step 2: Use actual attribute names (NOT indexed)
    output["SA_HC"] = subject_attrs
    output["OA_HC"] = object_attrs
    output["EA_HC"] = environment_attrs
    
    # Step 3: Create attribute value mappings with actual values
    output["SAV_HC"] = {}
    for attr_name in subject_attrs:
        values = subject_av.get(attr_name, [])
        output["SAV_HC"][attr_name] = values
    
    output["OAV_HC"] = {}
    for attr_name in object_attrs:
        values = object_av.get(attr_name, [])
        output["OAV_HC"][attr_name] = values
    
    output["EAV_HC"] = {}
    for attr_name in environment_attrs:
        values = environment_av.get(attr_name, [])
        output["EAV_HC"][attr_name] = values
    
    # Step 4: Extract indices from original SV, OV, EV
    def extract_indices(entity_dict, entity_count, num_attrs):
        """Extract value indices from original format."""
        indices = {}
        for i in range(entity_count):
            entity_key = f"S_{i+1}" if "S_" in str(list(entity_dict.keys())[0] if entity_dict else "") else \
                         f"O_{i+1}" if "O_" in str(list(entity_dict.keys())[0] if entity_dict else "") else f"E_{i+1}"
            
            for possible_key in [f"S_{i+1}", f"O_{i+1}", f"E_{i+1}"]:
                if possible_key in entity_dict:
                    entity_key = possible_key
                    break
            
            values_list = entity_dict.get(entity_key, [])
            attr_indices = []
            for val in values_list:
                parts = str(val).split("_")
                if len(parts) >= 3:
                    try:
                        idx = int(parts[-1])
                        attr_indices.append(idx)
                    except:
                        attr_indices.append(1)
                else:
                    attr_indices.append(1)
            indices[i] = attr_indices
        return indices
    
    # Extract indices from original data
    sv_indices = extract_indices(SV, subject_count, len(subject_attrs))
    ov_indices = extract_indices(OV, object_count, len(object_attrs))
    ev_indices = extract_indices(EV, environment_count, len(environment_attrs))
    
    # Step 5: Create SV_HC, OV_HC, EV_HC with actual values mapped from original indices
    output["SV_HC"] = {}
    for i in range(subject_count):
        subject_name = subjects[i] if i < len(subjects) else f"Subject_{i+1}"
        output["SV_HC"][subject_name] = []
        
        attr_indices = sv_indices.get(i, [1] * len(subject_attrs))
        
        for attr_idx, attr_name in enumerate(subject_attrs):
            values = subject_av.get(attr_name, [f"{attr_name}_value_1"])
            
            if attr_idx < len(attr_indices):
                idx = attr_indices[attr_idx] - 1  # Convert to 0-based
                if 0 <= idx < len(values):
                    output["SV_HC"][subject_name].append(values[idx])
                else:
                    output["SV_HC"][subject_name].append(values[0] if values else f"{attr_name}_value_1")
            else:
                output["SV_HC"][subject_name].append(values[0] if values else f"{attr_name}_value_1")
    
    output["OV_HC"] = {}
    for i in range(object_count):
        object_name = objects[i] if i < len(objects) else f"Object_{i+1}"
        output["OV_HC"][object_name] = []
        
        attr_indices = ov_indices.get(i, [1] * len(object_attrs))
        
        for attr_idx, attr_name in enumerate(object_attrs):
            values = object_av.get(attr_name, [f"{attr_name}_value_1"])
            
            if attr_idx < len(attr_indices):
                idx = attr_indices[attr_idx] - 1
                if 0 <= idx < len(values):
                    output["OV_HC"][object_name].append(values[idx])
                else:
                    output["OV_HC"][object_name].append(values[0] if values else f"{attr_name}_value_1")
            else:
                output["OV_HC"][object_name].append(values[0] if values else f"{attr_name}_value_1")
    
    output["EV_HC"] = {}
    for i in range(environment_count):
        environment_name = environments[i] if i < len(environments) else f"Environment_{i+1}"
        output["EV_HC"][environment_name] = []
        
        attr_indices = ev_indices.get(i, [1] * len(environment_attrs))
        
        for attr_idx, attr_name in enumerate(environment_attrs):
            values = environment_av.get(attr_name, [f"{attr_name}_value_1"])
            
            if attr_idx < len(attr_indices):
                idx = attr_indices[attr_idx] - 1
                if 0 <= idx < len(values):
                    output["EV_HC"][environment_name].append(values[idx])
                else:
                    output["EV_HC"][environment_name].append(values[0] if values else f"{attr_name}_value_1")
            else:
                output["EV_HC"][environment_name].append(values[0] if values else f"{attr_name}_value_1")
    
    return output
