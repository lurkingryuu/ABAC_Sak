# Star Count Feature Implementation - Complete Guide

## 📌 Overview

This feature allows you to control the exact number of wildcard stars (*) in generated ABAC rules. Instead of using fixed probability, you can now specify exactly how many stars each attribute should have in rule generation.

## 🎯 Key Features

### 1. **Mixed Array Format in JSON**
You can now use a combination of:
- **Single numbers**: `5` means 5 values, uses `global_stars` setting
- **Pairs**: `[5, 2]` means 5 values with 2 stars (overrides `global_stars`)

### 2. **Global Stars Setting**
Optional `global_stars` field acts as a default for all attributes without explicit star configuration.

### 3. **Flexible Configuration**
- Mix single numbers and pairs in the same array
- Override global setting per attribute
- Set stars to 0 to completely disable wildcards for specific attributes

## 📋 JSON Structure

### Minimal Example
```json
{
  "subject_size": 10,
  "object_size": 15,
  "environment_size": 5,
  "subject_attributes_count": 3,
  "object_attributes_count": 4,
  "environment_attributes_count": 2,
  "global_stars": 1,
  "subject_attributes_values": [
    [6, 2],    // 6 values, 2 stars
    5,         // 5 values, use global_stars=1
    [4, 1]     // 4 values, 1 star
  ],
  "object_attributes_values": [
    2,         // 2 values, use global_stars=1
    [3, 2],    // 3 values, 2 stars
    4,         // 4 values, use global_stars=1
    [5, 1]     // 5 values, 1 star
  ],
  "environment_attributes_values": [
    [1, 1],    // 1 value, 1 star
    2          // 2 values, use global_stars=1
  ],
  "subject_distributions": [...],
  "object_distributions": [...],
  "environment_distributions": [...],
  "accepted_rules_count": 20,
  "denied_rules_count": 5
}
```

## 🔄 How It Works

### Rule Generation Process

For each attribute in a rule:

1. **Extract Configuration**: 
   - If `[6, 2]`: 6 values, 2 stars
   - If `5` with global_stars=1: 5 values, 1 star
   - If `[4, 0]`: 4 values, 0 stars

2. **Create Choice Pool**:
   - Combine concrete values + star wildcards
   - Example: `[4, 0]` → 4 choices: [value1, value2, value3, value4]
   - Example: `[4, 2]` → 6 choices: [value1, value2, value3, value4, *, *]

3. **Random Selection**:
   - Uniformly random selection from the choice pool
   - For `[4, 2]`: 67% chance of concrete, 33% chance of wildcard

### Example Probabilities

| Format | Total Choices | Concrete % | Wildcard % |
|--------|---------------|-----------|-----------|
| [6, 2] | 8 | 75% | 25% |
| [5, 1] | 6 | 83% | 17% |
| [4, 0] | 4 | 100% | 0% |
| [2, 2] | 4 | 50% | 50% |

## 📁 Modified Files

### 1. **input.py**
- Added `parse_attribute_values_with_stars()` function
- Parses mixed format into (count, stars) tuples
- Stores both in config.ini

### 2. **gen.py**
- Updated `parse_attributes()` to read stars from config
- Passes star counts to rule generation function

### 3. **gen_rules.py**
- Modified `generate_rules_2()` function signature
- Accepts `subject_stars`, `object_stars`, `environment_stars` parameters
- Uses choice list approach instead of probability

### 4. **healthcare_generator.py**
- Added `extract_attribute_counts()` helper function
- Converts mixed format to integer list before processing

### 5. **university_generator.py**
- Added `extract_attribute_counts()` helper function
- Same as healthcare_generator

## 🚀 Usage Instructions

### Step 1: Prepare Input JSON
Create or modify `uploads/input.json` with the new format:

```json
{
  "global_stars": 1,
  "subject_attributes_values": [[6, 2], 5, [4, 1], 3, [2, 1], [3, 0]],
  "object_attributes_values": [[4, 2], 3, [2, 1], 2, [3, 1]],
  "environment_attributes_values": [[2, 1], 2, [2, 2], 1],
  ...
}
```

### Step 2: Run Configuration Parser
```bash
python3 access_control/input.py
```

This generates `config.ini` with both values and stars sections:
```ini
[SUBJECT_ATTRIBUTES]
values = 6,5,4,3,2,3
stars = 2,1,1,0,1,0
```

### Step 3: Run Data Generation
```bash
python3 access_control/gen.py
```

This generates:
- `output.json` - Regular ABAC data with star-controlled rules
- `output_healthcare.json` - Healthcare domain data
- `output_university.json` - University domain data
- `ACM.txt` - Access Control Matrix
- `access_data.txt` - Access control records

## 📊 Example Configuration

### Healthcare Use Case
```json
{
  "subject_size": 4,
  "object_size": 3,
  "environment_size": 2,
  "subject_attributes_count": 6,
  "object_attributes_count": 5,
  "environment_attributes_count": 4,
  "global_stars": 1,
  "subject_attributes_values": [
    [6, 2],     // Job roles: mostly specific, sometimes any
    5,          // Departments: default 1 star
    [4, 1],     // Certifications: mostly specific
    3,          // Zones: default 1 star
    [2, 1],     // Shifts: mostly specific
    [3, 0]      // Employment types: never wildcard
  ],
  "object_attributes_values": [
    [4, 2],     // Data types: flexible
    3,          // Sensitivity: default 1 star
    [2, 1],     // Categories: mostly specific
    2,          // Status: default 1 star
    [3, 1]      // Compliance: mostly specific
  ],
  "environment_attributes_values": [
    [2, 1],     // Location types: mostly specific
    2,          // Network security: default 1 star
    [2, 2],     // Access control: flexible
    1           // Physical security: default 1 star
  ],
  ...
}
```

## ✅ Verification

### Check Config.ini
Verify that `access_control/config.ini` contains both values and stars:

```ini
[SUBJECT_ATTRIBUTES]
values = 6,5,4,3,2,3
stars = 2,1,1,0,1,0
```

### Check Generated Rules
Look at generated rules in `outputs/output.json`. Example:
```json
{
  "SA_1 = SA_1_3, SA_2 = *, SA_3 = SA_3_2, ...",
  "SA_1 = *, SA_2 = SA_2_5, SA_3 = SA_3_1, ...",
  ...
}
```

### Analyze Star Distribution
- SA_1 with [6, 2]: ~25% of rules should have `SA_1 = *`
- SA_2 with global_stars=1: ~17% of rules should have `SA_2 = *`
- SA_6 with [3, 0]: 0% of rules should have `SA_6 = *`

## ⚠️ Error Handling

### Missing global_stars
If not specified, defaults to 0 (no wildcards for single numbers)

### Mixed array mismatch
Array can contain in any order:
- Single integers: `5`
- Pairs: `[5, 2]`

Example: `[5, [3, 2], 4, [2, 1]]` is valid

### Invalid formats
- `[5, 2, 3]` - Won't work, expected 2-element arrays
- `"5"` - Must be integer, not string
- Missing pairs - Use single number instead

## 🔧 Troubleshooting

### TypeError: '<' not supported between instances of 'int' and 'list'
**Cause**: Healthcare generator received mixed format array
**Fix**: Use the helper function `extract_attribute_counts()` to convert before passing

### Config parsing errors
**Cause**: Input JSON has invalid format
**Fix**: Ensure all attribute values are either:
- Single integers: `5`
- 2-element arrays: `[5, 2]`

### Rules not generated with expected star distribution
**Cause**: Star counts not properly passed to rule generator
**Fix**: Verify `config.ini` has both `values` and `stars` keys

## 📈 Performance Considerations

- More stars = more rule diversity
- Zero stars = deterministic rules based on attribute values only
- Optimal range: 0-3 stars per attribute for balanced randomness

## 🎓 Example Walkthrough

## Sample Files

See `dataset/input_with_stars.json` for a complete working example.

