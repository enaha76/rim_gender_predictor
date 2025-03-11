
import hashlib
import numpy as np

def extract_name_features(name):
    """Extract features from a name that might be indicative of gender"""
    first_name = name.split()[0].lower()
    features = []

    # Extract suffix features
    features.append(f"last_letter_{first_name[-1]}")
    if len(first_name) >= 2:
        features.append(f"last_two_{first_name[-2:]}")
    if len(first_name) >= 3:
        features.append(f"last_three_{first_name[-3:]}")

    # Extract prefix features
    features.append(f"first_letter_{first_name[0]}")
    if len(first_name) >= 2:
        features.append(f"first_two_{first_name[:2]}")
    if len(first_name) >= 3:
        features.append(f"first_three_{first_name[:3]}")

    # Length-based features
    length_category = "short" if len(first_name) < 5 else "medium" if len(first_name) < 8 else "long"
    features.append(f"length_{length_category}")
    features.append(f"exact_length_{len(first_name)}")

    # Vowel/consonant analysis
    vowels = 'aeiou'
    vowel_count = sum(1 for char in first_name if char in vowels)
    consonant_count = len(first_name) - vowel_count
    vowel_ratio = round(vowel_count / len(first_name), 1) if len(first_name) > 0 else 0
    features.append(f"vowel_ratio_{vowel_ratio}")

    if vowel_count > consonant_count:
        features.append("more_vowels")
    elif consonant_count > vowel_count:
        features.append("more_consonants")
    else:
        features.append("equal_vowels_consonants")

    # Specific patterns common in Mauritanian names
    if 'ou' in first_name:
        features.append("contains_ou")
    if 'ah' in first_name:
        features.append("contains_ah")
    if 'ma' in first_name:
        features.append("contains_ma")
    if 'med' in first_name:
        features.append("contains_med")

    return features

def hash_features(features, num_features=1000):
    """Convert feature list to a fixed-size vector using hashing."""
    feature_vector = np.zeros(num_features)
    for feature in features:
        hash_obj = hashlib.md5(feature.encode('utf-8'))
        idx = int(hash_obj.hexdigest(), 16) % num_features
        feature_vector[idx] += 1  # Simple term frequency
    return feature_vector
