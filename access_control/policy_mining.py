import config

n1 = config.n1
n2 = config.n2
n3 = config.n3
n4 = config.n4
n5 = config.n5
n6 = config.n6
subject_attributes = config.subject_attributes
object_attributes = config.object_attributes
environment_attributes = config.environment_attributes
N = config.n

# ATTRIBUTES
SA = [f"SA_{i}" for i in range(1, n4 + 1)]
OA = [f"OA_{i}" for i in range(1, n5 + 1)]
EA = [f"EA_{i}" for i in range(1, n6 + 1)]

attributes = SA + OA + EA


def compute_gini_index(access_data, attribute_index):
    yes_no_counts = {}
    for entry in access_data:
        key = entry[attribute_index]
        if key not in yes_no_counts:
            yes_no_counts[key] = [0, 0]  # [no_count, yes_count]
        yes_no_counts[key][entry[-1]] += 1

    total_entries = len(access_data)
    gini_index = 0
    for key, (no_count, yes_count) in yes_no_counts.items():
        total = no_count + yes_count
        gini_coefficient = 1 - (yes_count / total) ** 2 - (no_count / total) ** 2
        gini_index += (total / total_entries) * gini_coefficient

    return gini_index

def split_access_data(access_data, attribute_index):
    split_data = {}
    for entry in access_data:
        key = entry[attribute_index]
        if key not in split_data:
            split_data[key] = []
        split_data[key].append(entry)
    return split_data

def all_decisions_uniform(access_data):
    first_decision = access_data[0][-1]
    return all(entry[-1] == first_decision for entry in access_data)

def recursive_policy_mining(access_data, attributes, current_rule):
    if all_decisions_uniform(access_data):
        decision = "permit" if access_data[0][-1] == 1 else "deny"
        return [{"rule": current_rule, "decision": decision}]
    
    gini_list = [(compute_gini_index(access_data, i), i) for i in range(len(attributes))]
    gini_list.sort()

    best_attribute_index = gini_list[0][1]
    best_attribute = attributes[best_attribute_index]
    
    split_data = split_access_data(access_data, best_attribute_index)
    policy_rules = []

    for attribute_value, subset in split_data.items():
        new_rule = current_rule + [(best_attribute, attribute_value)]
        policy_rules.extend(recursive_policy_mining(subset, attributes, new_rule))

    return policy_rules

def policy_mining(access_data, attributes):
    return recursive_policy_mining(access_data, attributes, [])

access_data = []
with open(r"access_data.txt", "r") as file:
    for line in file:
        access_data.append(line.strip().split())

access_data = [[int(x) if x.isdigit() else x for x in row] for row in access_data]
# print(access_data)

policy = policy_mining(access_data, attributes)

# WRITE POLICY TO FILE
with open(r"rules.txt", "w") as file:
    for rule in policy:
        file.write(str(rule) + "\n")
    file.write("\n")
