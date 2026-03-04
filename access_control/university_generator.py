import json
import os
import re
import time
from typing import Dict, List, Any
from dotenv import load_dotenv
import google.generativeai as genai
from faker import Faker

load_dotenv()

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

def generate_university_subjects(subject_size: int) -> List[str]:
    """Generate university person names using Faker library."""
    print(f"Generating {subject_size} university person names using Faker...")
    subjects = []
    roles = ["Student", "Prof.", "Dr.", "Staff", "Admin", "Lecturer"]
    
    for i in range(subject_size):
        name = fake.name()
        role = roles[i % len(roles)]
        subjects.append(f"{role} {name}")
    
    return subjects

def generate_all_entities(model, subject_size: int, object_size: int, 
                          environment_size: int) -> tuple:
    """Generate entities: subjects using Faker, objects and environments using Gemini."""
    
    # Generate subjects using Faker
    subjects = generate_university_subjects(subject_size)
    
    # Generate objects and environments using Gemini
    prompt = f"""You MUST return ONLY a valid JSON object. No other text.

Generate exactly {object_size} university resource names and {environment_size} campus locations.

For objects (resources): Use names like "Course_CS101_Syllabus", "Transcript_Student_456", "ResearchPaper_AI_2024", "GradesDatabase_Sem1", "LibraryCatalog"
For environments (campus locations): Use names like "Library_Main", "LectureHall_Building_A_3", "Dormitory_Building_C", "DataCenter_IT", "AdminOffice_Building_B"

Return this exact JSON structure:
{{
    "objects": ["Course_CS101_Syllabus", "Transcript_Student_456", ...],
    "environments": ["Library_Main", "LectureHall_Building_A_3", ...]
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
        print(f"[ERROR] University entity parsing failed: {e}")
        return (
            subjects,
            [f"Resource_{i+1}" for i in range(object_size)],
            [f"Campus_{i+1}" for i in range(environment_size)]
        )

def generate_all_attributes(model, subject_attr_count: int, object_attr_count: int,
                           environment_attr_count: int) -> tuple:
    """Generate realistic university attribute names."""
    prompt = f"""You MUST return ONLY a valid JSON object. No other text.

Generate {subject_attr_count} university person attributes, {object_attr_count} resource attributes, and {environment_attr_count} campus attributes.

Subject attributes examples: role, department, year_level, major, status, faculty_type, access_level, student_id
Object attributes examples: resource_type, access_level, course_code, owner_type, sensitivity, classification, data_owner
Environment attributes examples: building_type, location_zone, access_zone, network_segment, security_level, room_type

Return this exact JSON structure:
{{
    "subject_attributes": ["role", "department", "year_level", "major", ...],
    "object_attributes": ["resource_type", "access_level", "course_code", "owner_type", ...],
    "environment_attributes": ["building_type", "location_zone", "access_zone", "network_segment", ...]
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
        print(f"[ERROR] University attribute parsing failed: {e}")
        return (
            [f"subject_attr_{i+1}" for i in range(subject_attr_count)],
            [f"object_attr_{i+1}" for i in range(object_attr_count)],
            [f"environment_attr_{i+1}" for i in range(environment_attr_count)]
        )

def generate_all_attribute_values(model, subject_attrs: List[str], object_attrs: List[str],
                                 environment_attrs: List[str], subject_values: List[int],
                                 object_values: List[int], environment_values: List[int]) -> tuple:
    """Generate realistic university attribute values in ONE API call."""
    subject_spec = "\n".join([f"  {i+1}. {attr} ({count} values)" 
                            for i, (attr, count) in enumerate(zip(subject_attrs, subject_values))])
    object_spec = "\n".join([f"  {i+1}. {attr} ({count} values)" 
                            for i, (attr, count) in enumerate(zip(object_attrs, object_values))])
    environment_spec = "\n".join([f"  {i+1}. {attr} ({count} values)" 
                                 for i, (attr, count) in enumerate(zip(environment_attrs, environment_values))])

    prompt = f"""You MUST return ONLY a valid JSON object. No other text.

Generate realistic university values for these attributes:

SUBJECT ATTRIBUTES:
{subject_spec}

OBJECT ATTRIBUTES:
{object_spec}

ENVIRONMENT ATTRIBUTES:
{environment_spec}

Return this exact JSON structure with arrays of strings as values:
{{
    "subject_attribute_values": {{
        "{subject_attrs[0]}": ["value1", "value2", ...],
        ...
    }},
    "object_attribute_values": {{
        "{object_attrs[0]}": ["value1", ...],
        ...
    }},
    "environment_attribute_values": {{
        "{environment_attrs[0]}": ["value1", ...],
        ...
    }}
}}

IMPORTANT:
- Each value array must have EXACTLY the specified number of values
- All values should use lowercase with underscores for multi-word values
- Return ONLY the JSON object, no other text"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    try:
        data = extract_json_from_response(text)
        subject_av = data.get("subject_attribute_values", {})
        object_av = data.get("object_attribute_values", {})
        environment_av = data.get("environment_attribute_values", {})

        # Validate counts
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
        print(f"[ERROR] University attribute values parsing failed: {e}")
        subject_av = {attr: [f"{attr}_value_{i+1}" for i in range(count)] 
                     for attr, count in zip(subject_attrs, subject_values)}
        object_av = {attr: [f"{attr}_value_{i+1}" for i in range(count)] 
                    for attr, count in zip(object_attrs, object_values)}
        environment_av = {attr: [f"{attr}_value_{i+1}" for i in range(count)] 
                         for attr, count in zip(environment_attrs, environment_values)}
        return (subject_av, object_av, environment_av)

def generate_university_data_with_gemini(config_ini_path: str = None,
                                        api_key: str = None) -> Dict[str, Any]:
    """Generate full university dataset using Gemini (names, attributes, values)."""
    # Resolve config.ini path
    if config_ini_path is None:
        config_ini_path = os.environ.get("ABAC_CONFIG_INI") or os.path.join(os.path.dirname(__file__), 'config.ini')
    
    if os.path.isdir(config_ini_path):
        config_ini_path = os.path.join(config_ini_path, 'config.ini')
    
    config_dir = os.path.dirname(config_ini_path)
    input_json_path = os.path.join(config_dir, 'uploads', 'input.json')

    with open(input_json_path, 'r') as f:
        config = json.load(f)

    print("Initializing Gemini API for university domain...")
    model = initialize_gemini(api_key)

    uni_data = {}

    # Entities
    subjects, objects, environments = generate_all_entities(
        model,
        config['subject_size'],
        config['object_size'],
        config['environment_size']
    )

    uni_data['subjects'] = subjects
    uni_data['objects'] = objects
    uni_data['environments'] = environments

    # Attributes
    subject_attrs, object_attrs, environment_attrs = generate_all_attributes(
        model,
        config['subject_attributes_count'],
        config['object_attributes_count'],
        config['environment_attributes_count']
    )

    uni_data['subject_attributes'] = subject_attrs
    uni_data['object_attributes'] = object_attrs
    uni_data['environment_attributes'] = environment_attrs

    # Attribute values
    subject_av, object_av, environment_av = generate_all_attribute_values(
        model,
        subject_attrs,
        object_attrs,
        environment_attrs,
        config['subject_attributes_values'],
        config['object_attributes_values'],
        config['environment_attributes_values']
    )

    uni_data['subject_attribute_values'] = subject_av
    uni_data['object_attribute_values'] = object_av
    uni_data['environment_attribute_values'] = environment_av

    return uni_data

def convert_to_university_format(uni_data: Dict, SV: Dict, OV: Dict, EV: Dict,
                                subject_count: int, object_count: int, environment_count: int) -> Dict[str, Any]:
    """Convert university data to actual names and values following SV/OV/EV indices."""
    output = {}
    
    subjects = uni_data['subjects']
    objects = uni_data['objects']
    environments = uni_data['environments']

    subject_attrs = uni_data['subject_attributes']
    object_attrs = uni_data['object_attributes']
    environment_attrs = uni_data['environment_attributes']

    subject_av = uni_data['subject_attribute_values']
    object_av = uni_data['object_attribute_values']
    environment_av = uni_data['environment_attribute_values']

    # Entities and attribute names
    output['S_UNI'] = subjects[:subject_count]
    output['O_UNI'] = objects[:object_count]
    output['E_UNI'] = environments[:environment_count]

    output['SA_UNI'] = subject_attrs
    output['OA_UNI'] = object_attrs
    output['EA_UNI'] = environment_attrs

    # Value maps
    output['SAV_UNI'] = subject_av
    output['OAV_UNI'] = object_av
    output['EAV_UNI'] = environment_av

    # Helper to extract indices from entity dictionaries
    def extract_indices(entity_dict, entity_count, num_attrs):
        indices = {}
        for i in range(entity_count):
            # Find the correct key for this entity
            entity_key = None
            for possible_key in [f"S_{i+1}", f"O_{i+1}", f"E_{i+1}"]:
                if possible_key in entity_dict:
                    entity_key = possible_key
                    break
            
            if entity_key is None:
                entity_key = f"S_{i+1}"

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

    s_indices = extract_indices(SV, subject_count, len(subject_attrs))
    o_indices = extract_indices(OV, object_count, len(object_attrs))
    e_indices = extract_indices(EV, environment_count, len(environment_attrs))

    # Build subject values mapping
    output['SV_UNI'] = {}
    for i in range(subject_count):
        subject_name = subjects[i] if i < len(subjects) else f"Subject_{i+1}"
        output['SV_UNI'][subject_name] = []
        attr_indices = s_indices.get(i, [1] * len(subject_attrs))
        for attr_idx, attr_name in enumerate(subject_attrs):
            values = subject_av.get(attr_name, [f"{attr_name}_value_1"])
            if attr_idx < len(attr_indices):
                idx = attr_indices[attr_idx] - 1
                if 0 <= idx < len(values):
                    output['SV_UNI'][subject_name].append(values[idx])
                else:
                    output['SV_UNI'][subject_name].append(values[0] if values else f"{attr_name}_value_1")
            else:
                output['SV_UNI'][subject_name].append(values[0] if values else f"{attr_name}_value_1")

    # Build object values mapping
    output['OV_UNI'] = {}
    for i in range(object_count):
        object_name = objects[i] if i < len(objects) else f"Object_{i+1}"
        output['OV_UNI'][object_name] = []
        attr_indices = o_indices.get(i, [1] * len(object_attrs))
        for attr_idx, attr_name in enumerate(object_attrs):
            values = object_av.get(attr_name, [f"{attr_name}_value_1"])
            if attr_idx < len(attr_indices):
                idx = attr_indices[attr_idx] - 1
                if 0 <= idx < len(values):
                    output['OV_UNI'][object_name].append(values[idx])
                else:
                    output['OV_UNI'][object_name].append(values[0] if values else f"{attr_name}_value_1")
            else:
                output['OV_UNI'][object_name].append(values[0] if values else f"{attr_name}_value_1")

    # Build environment values mapping
    output['EV_UNI'] = {}
    for i in range(environment_count):
        environment_name = environments[i] if i < len(environments) else f"Environment_{i+1}"
        output['EV_UNI'][environment_name] = []
        attr_indices = e_indices.get(i, [1] * len(environment_attrs))
        for attr_idx, attr_name in enumerate(environment_attrs):
            values = environment_av.get(attr_name, [f"{attr_name}_value_1"])
            if attr_idx < len(attr_indices):
                idx = attr_indices[attr_idx] - 1
                if 0 <= idx < len(values):
                    output['EV_UNI'][environment_name].append(values[idx])
                else:
                    output['EV_UNI'][environment_name].append(values[0] if values else f"{attr_name}_value_1")
            else:
                output['EV_UNI'][environment_name].append(values[0] if values else f"{attr_name}_value_1")

    return output
