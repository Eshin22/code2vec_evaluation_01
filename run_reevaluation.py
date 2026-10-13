"""Master re-evaluation script for Code2Vec (Al-Debagy & Martinek, 2021)
on Java benchmark datasets using the AST-based JavaParser extractor and canonical metrics
(with Seniors' NED method and BCP). Executes 10 runs per dataset to report mean and standard deviation.
"""

import os
import sys
import json
import csv
import subprocess
import numpy as np

# Ensure reevlauation_code2vec is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "tools", "code2vec"))

from adapters.code2vec_adapter import build_code2vec_inputs
from tools.code2vec.runner import run_code2vec_clustering
from metrics.metrics import (
    load_edges_csv,
    load_calls_csv,
    load_entrypoints,
    build_reachability_usecases,
    compute_all,
)

DATASETS_DIR = r"D:\Final Year Project\Cogcn_evalaution\dataset"
EXTRACTOR_JAR = os.path.join(BASE_DIR, "extractor", "extractor-0.1.0.jar")

DATASET_CONFIGS = [
    {"name": "acmeair", "k": 4},
    {"name": "autocare-nepal", "k": 9},
    {"name": "bear-board", "k": 5},
    {"name": "daytrader7", "k": 6},
    {"name": "javafx-pos", "k": 3},
    {"name": "jforum", "k": 19},
    {"name": "jpetstore-6", "k": 3},
    {"name": "pharmacy", "k": 9},
    {"name": "plantsbywebsphere", "k": 4},
    {"name": "roller", "k": 19},
    {"name": "spring-petclinic", "k": 8},
    {"name": "springblog", "k": 8},
]

RUNS_COUNT = 10
SEEDS = [42 + i for i in range(RUNS_COUNT)]


def run_pipeline():
    os.makedirs(os.path.join(BASE_DIR, "extracted"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "adapted"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "runs"), exist_ok=True)

    summary_rows = []

    print("=" * 80)
    print("  STARTING Code2Vec RE-EVALUATION ACROSS 12 JAVA DATASETS (10 RUNS EACH)")
    print("  Using JavaParser AST Extractor + Canonical TSE 2021 Metrics + Seniors' NED")
    print("=" * 80)

    for item in DATASET_CONFIGS:
        ds_name = item["name"]
        k = item["k"]
        ds_path = os.path.join(DATASETS_DIR, ds_name)
        extracted_path = os.path.join(BASE_DIR, "extracted", ds_name)
        adapted_path = os.path.join(BASE_DIR, "adapted", ds_name)
        output_ds_dir = os.path.join(BASE_DIR, "outputs", ds_name)
        os.makedirs(output_ds_dir, exist_ok=True)

        print(f"\n[{ds_name.upper()}] (Target K={k})")

        # Step 1: JavaParser AST Extraction
        if not os.path.exists(os.path.join(extracted_path, "nodes.csv")):
            print(f"  -> Step 1: Running JavaParser Extractor on {ds_name}...")
            os.makedirs(extracted_path, exist_ok=True)
            cmd = ["java", "-jar", EXTRACTOR_JAR, ds_path, extracted_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  [ERROR] Extractor failed on {ds_name}: {res.stderr}")
                continue
            print(f"     {res.stdout.strip()}")
        else:
            print(f"  -> Step 1: AST Extraction cached.")

        # Step 2: Code2Vec Adapter (class_vectors.npy + mapping.json)
        vec_path = os.path.join(adapted_path, "class_vectors.npy")
        mapping_path = os.path.join(adapted_path, "mapping.json")
        if os.path.exists(vec_path) and os.path.exists(mapping_path):
            print(f"  -> Step 2: Code2Vec Adapter cached.")
            with open(mapping_path, "r", encoding="utf-8") as mf:
                mapping = json.load(mf)
            mapping["idx_to_class"] = {int(k): v for k, v in mapping["idx_to_class"].items()}
            N = mapping["N"]
        else:
            print(f"  -> Step 2: Running Code2Vec Adapter (Mean Method Aggregation)...")
            vec_path, mapping = build_code2vec_inputs(extracted_path, ds_path, adapted_path)
            N = mapping["N"]
            print(f"     Generated class vectors ({N} classes, dim={mapping['vector_dim']})")

        # Load graph and calls for evaluation
        edges_path = os.path.join(extracted_path, "edges.csv")
        calls_path = os.path.join(extracted_path, "calls.csv")
        edges = load_edges_csv(edges_path)
        calls = load_calls_csv(calls_path) if os.path.exists(calls_path) else None

        # Load business use-case entrypoints for BCP metric
        config_ep_path = os.path.join(BASE_DIR, "configs", ds_name, "entrypoints.txt")
        if os.path.exists(config_ep_path):
            entrypoints = load_entrypoints(config_ep_path)
            class_usecases = build_reachability_usecases(edges, entrypoints)
            print(f"     Loaded {len(entrypoints)} business use cases for BCP reachability.")
        else:
            class_usecases = None

        # Step 3: Run Code2Vec Clustering across 10 random seeds
        print(f"  -> Step 3: Executing Code2Vec clustering across {RUNS_COUNT} random seeds...")
        run_results = []
        for r_idx, seed in enumerate(SEEDS):
            membership = run_code2vec_clustering(vec_path, target_k=k, seed=seed)
            partition = {mapping["idx_to_class"][i]: int(membership[i]) for i in range(N)}

            part_path = os.path.join(output_ds_dir, f"run_{r_idx + 1}.json")
            with open(part_path, "w", encoding="utf-8") as pf:
                json.dump(partition, pf, indent=2)

            metrics_val = compute_all(partition, edges, calls=calls, class_usecases=class_usecases)
            metrics_val["run"] = r_idx + 1
            metrics_val["seed"] = seed
            run_results.append(metrics_val)

        # Step 4: Aggregate 10-run statistics
        sm_vals = [m["SM"] for m in run_results]
        icp_vals = [m["ICP"] for m in run_results]
        bcp_vals = [m["BCP"] for m in run_results if "BCP" in m]
        ifn_vals = [m["IFN"] for m in run_results]
        ned_vals = [m["NED"] for m in run_results]
        ned_cr_vals = [m["NED_class_ratio"] for m in run_results]
        chm_vals = [m["CHM"] for m in run_results if "CHM" in m]
        chd_vals = [m["CHD"] for m in run_results if "CHD" in m]

        dataset_summary = {
            "dataset": ds_name,
            "classes_N": N,
            "target_k": k,
            "runs": RUNS_COUNT,
            "SM_mean": float(np.mean(sm_vals)),
            "SM_std": float(np.std(sm_vals)),
            "ICP_mean": float(np.mean(icp_vals)),
            "ICP_std": float(np.std(icp_vals)),
            "BCP_mean": float(np.mean(bcp_vals)) if bcp_vals else None,
            "BCP_std": float(np.std(bcp_vals)) if bcp_vals else None,
            "IFN_mean": float(np.mean(ifn_vals)),
            "IFN_std": float(np.std(ifn_vals)),
            "NED_mean": float(np.mean(ned_vals)),
            "NED_std": float(np.std(ned_vals)),
            "NED_class_ratio_mean": float(np.mean(ned_cr_vals)),
            "NED_class_ratio_std": float(np.std(ned_cr_vals)),
            "CHM_mean": float(np.mean(chm_vals)) if chm_vals else None,
            "CHM_std": float(np.std(chm_vals)) if chm_vals else None,
            "CHD_mean": float(np.mean(chd_vals)) if chd_vals else None,
            "CHD_std": float(np.std(chd_vals)) if chd_vals else None,
            "all_runs": run_results
        }

        # Save dataset summary JSON
        with open(os.path.join(output_ds_dir, "results_summary.json"), "w", encoding="utf-8") as sf:
            json.dump(dataset_summary, sf, indent=2)

        summary_rows.append(dataset_summary)

        print(f"     => SM:  {dataset_summary['SM_mean']:.4f} ± {dataset_summary['SM_std']:.4f}")
        print(f"     => ICP: {dataset_summary['ICP_mean']:.4f} ± {dataset_summary['ICP_std']:.4f}")
        if dataset_summary["BCP_mean"] is not None:
            print(f"     => BCP: {dataset_summary['BCP_mean']:.4f} ± {dataset_summary['BCP_std']:.4f}")
        print(f"     => IFN: {dataset_summary['IFN_mean']:.4f} ± {dataset_summary['IFN_std']:.4f}")
        print(f"     => NED (Seniors, lower is better): {dataset_summary['NED_mean']:.4f} ± {dataset_summary['NED_std']:.4f}")
        if dataset_summary["CHM_mean"] is not None:
            print(f"     => CHM: {dataset_summary['CHM_mean']:.4f}, CHD: {dataset_summary['CHD_mean']:.4f}")

    # Step 5: Write master results.csv
    csv_file = os.path.join(BASE_DIR, "results.csv")
    fieldnames = [
        "method", "dataset", "classes_N", "target_K", "runs",
        "SM_mean", "SM_std",
        "ICP_mean", "ICP_std",
        "BCP_mean", "BCP_std",
        "IFN_mean", "IFN_std",
        "NED_seniors_mean", "NED_seniors_std",
        "NED_class_ratio_mean", "NED_class_ratio_std",
        "CHM_mean", "CHM_std",
        "CHD_mean", "CHD_std"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary_rows:
            writer.writerow({
                "method": "Code2Vec",
                "dataset": s["dataset"],
                "classes_N": s["classes_N"],
                "target_K": s["target_k"],
                "runs": s["runs"],
                "SM_mean": f"{s['SM_mean']:.4f}",
                "SM_std": f"{s['SM_std']:.4f}",
                "ICP_mean": f"{s['ICP_mean']:.4f}",
                "ICP_std": f"{s['ICP_std']:.4f}",
                "BCP_mean": f"{s['BCP_mean']:.4f}" if s["BCP_mean"] is not None else "N/A",
                "BCP_std": f"{s['BCP_std']:.4f}" if s["BCP_std"] is not None else "N/A",
                "IFN_mean": f"{s['IFN_mean']:.4f}",
                "IFN_std": f"{s['IFN_std']:.4f}",
                "NED_seniors_mean": f"{s['NED_mean']:.4f}",
                "NED_seniors_std": f"{s['NED_std']:.4f}",
                "NED_class_ratio_mean": f"{s['NED_class_ratio_mean']:.4f}",
                "NED_class_ratio_std": f"{s['NED_class_ratio_std']:.4f}",
                "CHM_mean": f"{s['CHM_mean']:.4f}" if s["CHM_mean"] is not None else "N/A",
                "CHM_std": f"{s['CHM_std']:.4f}" if s["CHM_std"] is not None else "N/A",
                "CHD_mean": f"{s['CHD_mean']:.4f}" if s["CHD_mean"] is not None else "N/A",
                "CHD_std": f"{s['CHD_std']:.4f}" if s["CHD_std"] is not None else "N/A",
            })

    # Step 6: Generate Markdown Matrix
    md_file = os.path.join(BASE_DIR, "SUMMARY_MATRIX.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Code2Vec Re-Evaluation Benchmarking Matrix\n\n")
        f.write("Generated using **JavaParser AST Structural Extractor** and **TSE 2021 Canonical Metrics** with **Seniors' NED Method** and **BCP (Kalia et al. 2021)**.\n\n")
        f.write("Averages and standard deviations over **10 independent runs** per dataset.\n\n")
        f.write("| Dataset | Classes ($N$) | Target $K$ | SM ↑ | ICP ↓ | BCP ↓ | IFN ↓ | NED (Seniors) ↓ | NED (Class-Ratio) ↑ | CHM ↑ | CHD ↑ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for s in summary_rows:
            bcp_str = f"`{s['BCP_mean']:.4f} ± {s['BCP_std']:.4f}`" if s['BCP_mean'] is not None else "—"
            chm_str = f"`{s['CHM_mean']:.4f}`" if s['CHM_mean'] is not None else "—"
            chd_str = f"`{s['CHD_mean']:.4f}`" if s['CHD_mean'] is not None else "—"
            f.write(f"| **{s['dataset']}** | {s['classes_N']} | {s['target_k']} | "
                    f"`{s['SM_mean']:.4f} ± {s['SM_std']:.4f}` | "
                    f"`{s['ICP_mean']:.4f} ± {s['ICP_std']:.4f}` | "
                    f"{bcp_str} | "
                    f"`{s['IFN_mean']:.4f} ± {s['IFN_std']:.4f}` | "
                    f"`{s['NED_mean']:.4f} ± {s['NED_std']:.4f}` | "
                    f"`{s['NED_class_ratio_mean']:.4f} ± {s['NED_class_ratio_std']:.4f}` | "
                    f"{chm_str} | {chd_str} |\n")

    print("\n" + "=" * 80)
    print("  Code2Vec RE-EVALUATION FINISHED SUCCESSFULLY!")
    print(f"  Results saved to:\n   - {csv_file}\n   - {md_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
