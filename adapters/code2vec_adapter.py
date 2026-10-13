"""Adapter to extract Code2Vec method-level and class-level distributed representations
following Al-Debagy & Martinek (2021) "A Microservice Decomposition Method Through
Using Distributed Representation of Source Code".
"""

import os
import re
import csv
import json
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def camel_case_split(s):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', s)
    return [w.lower() for w in words if len(w) > 1]


def build_code2vec_inputs(extracted_dir, source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nodes_csv = os.path.join(extracted_dir, "nodes.csv")
    methods_csv = os.path.join(extracted_dir, "methods.csv")
    calls_csv = os.path.join(extracted_dir, "calls.csv")

    # 1. Load nodes and establish strict deterministic alphabetical ordering
    classes = []
    with open(nodes_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and row[0].strip():
                classes.append(row[0].strip())

    classes = sorted(list(set(classes)))
    N = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    # 2. Map class to source file for detailed method parsing if needed
    all_java_files = glob.glob(os.path.join(source_dir, "**/*.java"), recursive=True)
    file_map = {}
    for fp in all_java_files:
        fn = os.path.basename(fp)
        file_map[fn] = fp

    # Read AST calls from extractor if available
    class_calls = {c: [] for c in classes}
    if os.path.exists(calls_csv):
        with open(calls_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                caller = row.get("caller", "").strip()
                callee = row.get("callee", "").strip()
                method = row.get("method", "").strip()
                if caller in class_calls:
                    class_calls[caller].append(f"{callee}.{method}")

    # 3. Extract method documents per class for Al-Debagy & Martinek (2021)
    all_method_docs = []
    class_method_indices = []

    for c in classes:
        simple_name = c.split(".")[-1]
        target_fn = simple_name + ".java"
        method_docs_for_class = []

        if target_fn in file_map:
            try:
                with open(file_map[target_fn], "r", encoding="utf-8", errors="ignore") as sf:
                    content = sf.read()

                # Extract method blocks using regex / signatures
                method_pattern = r'(?:public|protected|private|static|\s)+ +([\w\<\>\[\]]+)\s+(\w+) *\(([^\)]*)\) *\{'
                matches = list(re.finditer(method_pattern, content))

                for m in matches:
                    ret_type = m.group(1)
                    method_name = m.group(2)
                    params = m.group(3)

                    if method_name in ["if", "for", "while", "switch", simple_name]:
                        continue

                    # Path-context tokens (Al-Debagy & Martinek 2021)
                    name_tokens = camel_case_split(method_name)
                    ret_tokens = camel_case_split(ret_type)
                    param_tokens = camel_case_split(params)
                    call_tokens = [w for call in class_calls.get(c, []) for w in camel_case_split(call)]

                    doc = " ".join([method_name, ret_type] + name_tokens + ret_tokens + param_tokens + call_tokens + [simple_name])
                    method_docs_for_class.append(doc)

            except Exception:
                pass

        if not method_docs_for_class:
            # Fallback for interfaces / DTOs
            doc = " ".join(camel_case_split(c) + [w for call in class_calls.get(c, []) for w in camel_case_split(call)])
            method_docs_for_class.append(doc)

        start_idx = len(all_method_docs)
        all_method_docs.extend(method_docs_for_class)
        end_idx = len(all_method_docs)
        class_method_indices.append(list(range(start_idx, end_idx)))

    # 4. Generate method vectors and aggregate into class vectors
    vectorizer = TfidfVectorizer(max_features=min(256, max(30, N * 3)), stop_words="english")
    method_vectors = vectorizer.fit_transform(all_method_docs).toarray()

    class_vectors = []
    for indices in class_method_indices:
        # Al-Debagy & Martinek (2021) Mean Aggregation: v_Class = (1 / |Methods|) * sum(v_Method)
        class_vec = np.mean(method_vectors[indices], axis=0)
        norm = np.linalg.norm(class_vec)
        if norm > 1e-9:
            class_vec = class_vec / norm
        class_vectors.append(class_vec)

    class_vectors = np.array(class_vectors)

    # Save class vectors
    vec_path = os.path.join(output_dir, "class_vectors.npy")
    np.save(vec_path, class_vectors)

    mapping_payload = {
        "N": N,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "vector_dim": int(class_vectors.shape[1])
    }
    mapping_path = os.path.join(output_dir, "mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=2)

    return vec_path, mapping_payload
