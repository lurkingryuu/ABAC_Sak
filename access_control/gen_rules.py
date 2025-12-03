import random

# ********** WEIGHTED FUNCTIONS ARE INCOMPLETE **********

def generate_rules_1(N, n4, n5, n6, SA_values, OA_values, EA_values):
    rules = []
    
    for _ in range(N):
        rule_parts = []
        
        # Generate SA attributes
        for i in range(1, n4 + 1):
            key = f"SA_{i}"
            choices = SA_values.get(key, []) + ['*']
            rule_parts.append(f"{key} = {random.choice(choices)}")
        
        # Generate OA attributes
        for i in range(1, n5 + 1):
            key = f"OA_{i}"
            choices = OA_values.get(key, []) + ['*']
            rule_parts.append(f"{key} = {random.choice(choices)}")
        
        # Generate EA attributes
        for i in range(1, n6 + 1):
            key = f"EA_{i}"
            choices = EA_values.get(key, []) + ['*']
            rule_parts.append(f"{key} = {random.choice(choices)}")
        
        rules.append(", ".join(rule_parts))
        # print(rule_parts)
    
    return rules

# def generate_rules_2(N, n4, n5, n6, SA_values, OA_values, EA_values):
#     rules = []
    
#     for _ in range(N):
#         rule_parts = []
        
#         # Generate SA attributes
#         for i in range(1, n4 + 1):
#             key = f"SA_{i}"
#             choices = SA_values.get(key, []) + ['*', '*']  # Double the probability of '*'
#             rule_parts.append(f"{key} = {random.choice(choices)}")
        
#         # Generate OA attributes
#         for i in range(1, n5 + 1):
#             key = f"OA_{i}"
#             choices = OA_values.get(key, []) + ['*', '*']  # Double the probability of '*'
#             rule_parts.append(f"{key} = {random.choice(choices)}")
        
#         # Generate EA attributes
#         for i in range(1, n6 + 1):
#             key = f"EA_{i}"
#             choices = EA_values.get(key, []) + ['*', '*']  # Double the probability of '*'
#             rule_parts.append(f"{key} = {random.choice(choices)}")
        
#         rules.append(", ".join(rule_parts))
    
#     return rules



def generate_rules_2(N, n4, n5, n6, SV, OV, EV):
    """
    Generate N rules by picking one subject S_x, one object O_y and one env E_z per rule.
    Use the chosen instance's concrete attribute tokens for ALL SA_i / OA_i / EA_i.
    Include '*' for any attribute with probability star_prob (10^-5).
    If no instance has all required attributes, raise an error.
    """
    import random

    rules = []
    star_prob = 0.03

    subject_keys = list(SV.keys()) if SV else []
    object_keys = list(OV.keys()) if OV else []
    env_keys = list(EV.keys()) if EV else []

    if not subject_keys or not object_keys or not env_keys:
        raise ValueError("SV/OV/EV must contain at least one instance each")

    def build_attr_map(inst_list):
        """Map attr key 'SA_1' -> token 'SA_1_15' for a single instance list."""
        m = {}
        for token in inst_list:
            if not isinstance(token, str):
                continue
            parts = token.split('_')
            if len(parts) >= 3:
                attr = f"{parts[0]}_{parts[1]}"
                m[attr] = token
        return m

    def pick_instance_with_all(prefix, keys_list, all_map, required_count, attempts=200):
        for _ in range(attempts):
            inst = random.choice(keys_list)
            m = build_attr_map(all_map.get(inst, []))
            if all(f"{prefix}_{i}" in m for i in range(1, required_count + 1)):
                return inst, m
        raise RuntimeError(f"No instance found with complete {prefix} attributes after {attempts} attempts")

    for _ in range(N):
        # pick subject/object/env that actually have all corresponding attributes
        s, s_map = pick_instance_with_all('SA', subject_keys, SV, n4)
        o, o_map = pick_instance_with_all('OA', object_keys, OV, n5)
        e, e_map = pick_instance_with_all('EA', env_keys, EV, n6)

        parts = []
        # use exact tokens from chosen subject or '*' with star_prob
        for i in range(1, n4 + 1):
            key = f"SA_{i}"
            val = s_map[key]
            chosen = '*' if random.random() < star_prob else val
            parts.append(f"{key} = {chosen}")

        # use exact tokens from chosen object or '*' with star_prob
        for i in range(1, n5 + 1):
            key = f"OA_{i}"
            val = o_map[key]
            chosen = '*' if random.random() < star_prob else val
            parts.append(f"{key} = {chosen}")

        # use exact tokens from chosen environment or '*' with star_prob
        for i in range(1, n6 + 1):
            key = f"EA_{i}"
            val = e_map[key]
            chosen = '*' if random.random() < star_prob else val
            parts.append(f"{key} = {chosen}")

        rules.append(", ".join(parts))

    return rules








def generate_rules_half(N, n4, n5, n6, SA_values, OA_values, EA_values):
    rules = []
    
    for _ in range(N):
        rule_parts = []
        
        # Function to choose with 50% probability for '*'
        def choose_value(values):
            if random.random() < 0.5:  # 50% probability of choosing '*'
                return '*'
            return random.choice(values) if values else '*'

        # Generate SA attributes
        for i in range(1, n4 + 1):
            key = f"SA_{i}"
            rule_parts.append(f"{key} = {choose_value(SA_values.get(key, []))}")
        
        # Generate OA attributes
        for i in range(1, n5 + 1):
            key = f"OA_{i}"
            rule_parts.append(f"{key} = {choose_value(OA_values.get(key, []))}")
        
        # Generate EA attributes
        for i in range(1, n6 + 1):
            key = f"EA_{i}"
            rule_parts.append(f"{key} = {choose_value(EA_values.get(key, []))}")
        
        rules.append(", ".join(rule_parts))
    
    return rules

def flatten_and_compute_weights(matrix):
    value_counts = {}
    total_count = 0

    for row in matrix:
        for value in row:
            value_counts[value] = value_counts.get(value, 0) + 1
            total_count += 1

    # Convert counts to probabilities
    weighted_values = list(value_counts.keys())
    weights = [count / total_count for count in value_counts.values()]

    print("Values and their probabilities:")
    for val, weight in zip(weighted_values, weights):
        print(f"{val}: {weight:.4f}")

    return weighted_values, weights

def generate_rules_weighted(N, n4, n5, n6, SV, OV, EV):
    rules = []

    # Compute weighted value distributions
    SA_values_, SA_weights_ = flatten_and_compute_weights(SV)
    OA_values_, OA_weights_ = flatten_and_compute_weights(OV)
    EA_values_, EA_weights_ = flatten_and_compute_weights(EV)

    for _ in range(N):
        rule_parts = []

        # Function to sample a value based on weights with '*' having fixed probability
        def choose_weighted(values, weights, star_prob=0.5):
            return random.choices(values, weights=weights, k=1)[0]

        # Generate SA attributes
        for i in range(1, n4 + 1):
            rule_parts.append(f"SA_{i} = {choose_weighted(SA_values_, SA_weights_)}")

        # Generate OA attributes
        for i in range(1, n5 + 1):
            rule_parts.append(f"OA_{i} = {choose_weighted(OA_values_, OA_weights_)}")

        # Generate EA attributes
        for i in range(1, n6 + 1):
            rule_parts.append(f"EA_{i} = {choose_weighted(EA_values_, EA_weights_)}")

        rules.append(", ".join(rule_parts))

    return rules

def flatten_and_compute_weights_1(matrix, star_weight):
    value_counts = {}
    total_count = 0

    for row in matrix:
        for value in row:
            value_counts[value] = value_counts.get(value, 0) + 1
            total_count += 1

    # Add '*' to the value list with a custom weight
    value_counts['*'] = star_weight
    total_count += star_weight

    # Convert counts to probabilities
    weighted_values = list(value_counts.keys())
    weights = [count / total_count for count in value_counts.values()]

    return weighted_values, weights

def generate_rules_weighted_1(N, n4, n5, n6, SV, OV, EV, star_weight=1):
    rules = []

    # Compute weighted value distributions with star_weight
    SA_values_, SA_weights_ = flatten_and_compute_weights_1(SV, star_weight)
    OA_values_, OA_weights_ = flatten_and_compute_weights_1(OV, star_weight)
    EA_values_, EA_weights_ = flatten_and_compute_weights_1(EV, star_weight)

    for _ in range(N):
        rule_parts = []

        # Function to sample a value based on weights
        def choose_weighted(values, weights):
            return random.choices(values, weights=weights, k=1)[0]

        # Generate SA attributes
        for i in range(1, n4 + 1):
            rule_parts.append(f"SA_{i} = {choose_weighted(SA_values_, SA_weights_)}")

        # Generate OA attributes
        for i in range(1, n5 + 1):
            rule_parts.append(f"OA_{i} = {choose_weighted(OA_values_, OA_weights_)}")

        # Generate EA attributes
        for i in range(1, n6 + 1):
            rule_parts.append(f"EA_{i} = {choose_weighted(EA_values_, EA_weights_)}")

        rules.append(", ".join(rule_parts))

    return rules

def generate_rules_weighted_half(N, n4, n5, n6, SV, OV, EV):
    rules = []

    # Compute weighted value distributions
    SA_values_, SA_weights_ = flatten_and_compute_weights(SV)
    OA_values_, OA_weights_ = flatten_and_compute_weights(OV)
    EA_values_, EA_weights_ = flatten_and_compute_weights(EV)

    for _ in range(N):
        rule_parts = []

        # Function to sample a value based on weights with '*' having fixed probability
        def choose_weighted(values, weights, star_prob=0.5):
            if random.random() < star_prob:
                return '*'
            return random.choices(values, weights=weights, k=1)[0]

        # Generate SA attributes
        for i in range(1, n4 + 1):
            rule_parts.append(f"SA_{i} = {choose_weighted(SA_values_, SA_weights_)}")

        # Generate OA attributes
        for i in range(1, n5 + 1):
            rule_parts.append(f"OA_{i} = {choose_weighted(OA_values_, OA_weights_)}")

        # Generate EA attributes
        for i in range(1, n6 + 1):
            rule_parts.append(f"EA_{i} = {choose_weighted(EA_values_, EA_weights_)}")

        rules.append(", ".join(rule_parts))

    return rules
