# Code2Vec Benchmarking Re-Evaluation (`reevlauation_code2vec`)

This project implements the verified, reproducible benchmarking pipeline for **Code2Vec (Al-Debagy & Martinek, 2021)**: *"A Microservice Decomposition Method Through Using Distributed Representation of Source Code"*, fully aligned with the **`fyp-benchmarking` AST architecture**.

---

## 🔍 Key Discrepancies in Old `Code2Vec_Evaluation` vs. New Pipeline

| Dimension | Old `Code2Vec_Evaluation` (Flawed) | New `reevlauation_code2vec` (FYP Standard) |
| :--- | :--- | :--- |
| **AST Extractor** | Python `javalang` (no type solver, no full FQN resolution, misses multi-package method calls) | **JavaParser + JavaSymbolSolver** (`extractor-0.1.0.jar`) with exact resolved types |
| **Structural Graph** | Approximate simple name matching | Exact typed relations (`CALL`, `EXTENDS`, `IMPLEMENTS`, `FIELD`) |
| **NED Formula** | Class ratio $\frac{\text{Classes in [5, 20]}}{\text{Total classes}}$ (higher is better) | **Seniors' Equation 4**: $1 - \frac{\sum n_k}{K}$ (lower is better, 0 = balanced) |
| **Metrics Standard** | Heuristic Python functions with target overrides | **Canonical Jin et al. (TSE 2021)** metric engine (`metrics/metrics.py`) |
| **Protocol** | 1 unseeded run / package baseline | **10 independent runs (seeds 42..51)** reporting Mean ± Std |

---

## 📐 Canonical Formulations: Seniors' NED & BCP

### 1. Seniors' NED Formulation (Wu et al. Eq. 4)

$$\text{NED} = 1 - \frac{\sum_{k=0}^{N} n_k}{|N|}$$

- $|N|$ (or $K$) is the **total number of microservices / clusters**.
- $n_k = 1$ if the size of cluster $k$ satisfies $5 \le |k| \le 20$ (non-extreme), and $0$ otherwise.
- **Lower NED values reflect more balanced microservice size distributions** ($0.0$ is ideal).

### 2. Business Context Purity (BCP) Formulation (Kalia et al. 2021)

$$\text{BCP} = \frac{1}{M}\sum_{i=1}^{M} bcp_i, \quad \text{where } bcp_i = -\sum_{j=1}^{m_i} \frac{1}{m_i}\log\left(\frac{1}{m_i}\right) = \log(m_i)$$

- $m_i$ represents the number of distinct business use cases implemented by partition $i \in M$.
- Derived via `CALL` and `FIELD` call-graph reachability from entrypoints in `configs/{dataset}/entrypoints.txt`.
- Lower values indicate higher functional cohesion ($0.0$ is ideal).

---

## 📊 Benchmark Results (Mean ± Std over 10 Runs)

| Dataset | Classes ($N$) | Target $K$ | SM ↑ | ICP ↓ | BCP ↓ | IFN ↓ | NED (Seniors) ↓ | NED (Class-Ratio) ↑ | CHM ↑ | CHD ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **acmeair** | 84 | 4 | `0.0786 ± 0.0240` | `0.4694 ± 0.0636` | `1.3968 ± 0.1183` | `9.4000 ± 0.5723` | `0.5250 ± 0.0750` | `0.2429 ± 0.0443` | `0.3392` | `0.2379` |
| **autocare-nepal** | 48 | 9 | `0.0861 ± 0.0233` | `0.7081 ± 0.0548` | `0.5773 ± 0.0791` | `1.8556 ± 0.1863` | `0.5333 ± 0.0831` | `0.7708 ± 0.0527` | `0.7390` | `0.6571` |
| **bear-board** | 34 | 5 | `0.0202 ± 0.0176` | `0.5579 ± 0.1469` | `1.1586 ± 0.1163` | `1.0000 ± 0.2191` | `0.1600 ± 0.0800` | `0.9324 ± 0.0457` | `0.7387` | `0.6964` |
| **daytrader7** | 115 | 6 | `0.0221 ± 0.0065` | `0.5515 ± 0.0963` | `0.8376 ± 0.1247` | `4.0000 ± 0.4282` | `0.3000 ± 0.1000` | `0.5365 ± 0.0628` | `0.5678` | `0.4969` |
| **javafx-pos** | 34 | 3 | `0.0399 ± 0.0056` | `0.3604 ± 0.1000` | `1.6036 ± 0.2109` | `3.2667 ± 0.2000` | `0.0667 ± 0.1333` | `0.8735 ± 0.2530` | `0.3558` | `0.2634` |
| **jforum** | 362 | 19 | `0.0805 ± 0.0081` | `0.7677 ± 0.0301` | `2.2057 ± 0.1156` | `11.1789 ± 0.2198` | `0.4526 ± 0.0482` | `0.3309 ± 0.0483` | `0.3502` | `0.2130` |
| **jpetstore-6** | 24 | 3 | `0.0176 ± 0.0000` | `0.6692 ± 0.0000` | `1.3863 ± 0.0000` | `3.0000 ± 0.0000` | `0.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.3006` | `0.1693` |
| **pharmacy** | 59 | 9 | `0.0907 ± 0.0148` | `0.3632 ± 0.0619` | `1.3464 ± 0.0950` | `1.2000 ± 0.1296` | `0.2556 ± 0.0868` | `0.8864 ± 0.0634` | `0.7168` | `0.6643` |
| **plantsbywebsphere** | 37 | 4 | `0.0388 ± 0.0045` | `0.4288 ± 0.0239` | `1.2033 ± 0.0794` | `4.3750 ± 0.1677` | `0.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.4199` | `0.1699` |
| **roller** | 534 | 19 | `0.0528 ± 0.0080` | `0.6376 ± 0.0200` | `2.3679 ± 0.1668` | `11.8842 ± 0.2444` | `0.6000 ± 0.0752` | `0.1753 ± 0.0391` | `0.3430` | `0.1759` |
| **spring-petclinic** | 26 | 8 | `0.1262 ± 0.0202` | `0.5694 ± 0.0451` | `0.7030 ± 0.0711` | `0.9750 ± 0.1225` | `0.7000 ± 0.0612` | `0.5346 ± 0.0885` | `0.5919` | `0.5267` |
| **springblog** | 47 | 8 | `0.0426 ± 0.0134` | `0.6367 ± 0.0597` | `1.0542 ± 0.0852` | `2.1500 ± 0.2151` | `0.3875 ± 0.1179` | `0.8128 ± 0.0778` | `0.5583` | `0.5154` |

---

## 📁 Directory Layout

```
reevlauation_code2vec/
├── configs/                           # Entrypoints per dataset for BCP reachability
├── extractor/
│   └── extractor-0.1.0.jar           # JavaParser AST structural extractor
├── adapters/
│   └── code2vec_adapter.py           # Method-level AST path to class vector adapter
├── tools/code2vec/
│   └── runner.py                     # Multi-seed Code2Vec clustering executor
├── metrics/
│   └── metrics.py                     # Canonical TSE 2021 metrics + Seniors' NED + BCP
├── extracted/                         # AST extraction outputs (nodes.csv, edges.csv, calls.csv)
├── adapted/                           # Code2Vec class representations & mappings
├── outputs/                           # Partition JSONs (run_1.json .. run_10.json)
├── results.csv                        # 10-run summary table (mean ± std)
├── SUMMARY_MATRIX.md                  # Markdown comparison matrix
└── run_reevaluation.py                # Master 10-run driver script
```

---

## 🚀 How to Re-Run

```powershell
cd "D:\Final Year Project\reevlauation_code2vec"
python run_reevaluation.py
```
