"""Unified metric calculator: SM, ICP, BCP, IFN, NED.

Canonical formulations from Jin et al., TSE 2021 (as used by the
CHGNN / CoGCN / Mono2Micro literature), with BCP per Kalia et al. (ESEC/FSE 2021)
formulation (Wu et al., Equation 4):
    NED = 1 - (sum_{k=1}^N n_k) / |N|
where |N| is the total number of microservices, and n_k is 1 if 5 <= |k| <= 20 else 0.
Lower values reflect more balanced distributions.

Inputs:
    partition: {class_fqn: cluster_id}
    edges: list of Edge(src, dst, type, weight)
"""

import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: str
    weight: float = 1.0


def _clusters(partition):
    """Map cluster_id -> set of classes."""
    clusters = {}
    for cls, cid in partition.items():
        clusters.setdefault(cid, set()).add(cls)
    return clusters


def sm(partition, edges, undirected=False):
    """Structural Modularity. Edge COUNTS (unweighted), all edge types.

    SM = (1/K) * sum_k(mu_k / N_k^2)
         - (1/(K(K-1)/2)) * sum_{i<j}(sigma_ij / (2 * N_i * N_j))

    undirected=True reproduces CHGNN's metric.py: edges collapse to unordered pairs
    (self-loops dropped), each intra pair counts TWICE in mu_k, each cross pair ONCE in sigma_ij.
    """
    clusters = _clusters(partition)
    K = len(clusters)

    if undirected:
        pairs = {frozenset((e.src, e.dst)) for e in edges if e.src != e.dst}
        edges = [Edge(*sorted(p), "UND", 1.0) for p in pairs]

    mu = {cid: 0 for cid in clusters}          # intra-edge counts
    sigma = {}                                  # {frozenset({i,j}): count}
    for e in edges:
        ci, cj = partition.get(e.src), partition.get(e.dst)
        if ci is None or cj is None:
            continue
        if ci == cj:
            mu[ci] += 2 if undirected else 1
        else:
            key = frozenset((ci, cj))
            sigma[key] = sigma.get(key, 0) + 1

    intra = sum(mu[cid] / len(members) ** 2 for cid, members in clusters.items()) / K if K > 0 else 0.0

    if K < 2:
        return intra

    n_pairs = K * (K - 1) / 2
    inter = sum(
        sigma.get(frozenset((i, j)), 0) / (2 * len(clusters[i]) * len(clusters[j]))
        for i, j in combinations(clusters, 2)
    ) / n_pairs
    return intra - inter


def icp(partition, edges):
    """Inter-Call Percentage: crossing CALL-edge weight / total CALL-edge
    weight. Lower is better. Returns 0.0 if there are no call edges."""
    total = 0.0
    crossing = 0.0
    for e in edges:
        if e.type != "CALL":
            continue
        ci, cj = partition.get(e.src), partition.get(e.dst)
        if ci is None or cj is None:
            continue
        total += e.weight
        if ci != cj:
            crossing += e.weight
    return crossing / total if total else 0.0


def ifn(partition, edges):
    """Interface Number: IFN = (1/K) * sum_k |interfaces(k)|, where an
    interface of partition k is a class in k receiving CALL edges from
    outside k. Lower is better."""
    clusters = _clusters(partition)
    K = len(clusters)
    if K == 0:
        return 0.0
    interfaces = set()
    for e in edges:
        if e.type != "CALL":
            continue
        ci, cj = partition.get(e.src), partition.get(e.dst)
        if ci is None or cj is None or ci == cj:
            continue
        interfaces.add(e.dst)
    return len(interfaces) / K


def ned(partition, lo=5, hi=20):
    """Non-Extreme Distribution (NED) per Seniors' Method (Wu et al., Eq. 4):
    NED = 1 - (sum_{k=1}^N n_k) / |N|
    where |N| (or K) is the total number of microservices/clusters,
    and n_k = 1 if lo <= |cluster_k| <= hi else 0.
    Lower NED values reflect more balanced microservice size distributions (0.0 is optimal).
    """
    clusters = _clusters(partition)
    K = len(clusters)
    if K == 0:
        return 0.0
    non_extreme_count = sum(1 for members in clusters.values() if lo <= len(members) <= hi)
    return 1.0 - (non_extreme_count / K)


def ned_class_ratio(partition, lo=5, hi=20):
    """Legacy class-level ratio (fraction of total classes in non-extreme partitions)."""
    clusters = _clusters(partition)
    total = len(partition)
    if total == 0:
        return 0.0
    extreme = sum(
        len(members) for members in clusters.values()
        if not lo <= len(members) <= hi
    )
    return 1.0 - extreme / total


@dataclass(frozen=True)
class Operation:
    """A callee method: owning class FQN, name, parameter types, return type."""
    cls: str
    name: str
    params: tuple
    returns: str


_NON_DOMAIN = {
    "void", "int", "long", "short", "byte", "float", "double", "boolean",
    "char", "string", "list", "map", "set", "collection", "object",
    "integer", "optional", "iterable",
}


def _published_ops(partition, calls):
    """Operations of each partition invoked from OUTSIDE it.
    calls: iterable of (caller_class, Operation)."""
    ops = {cid: set() for cid in set(partition.values())}
    for caller, op in calls:
        ci, cj = partition.get(caller), partition.get(op.cls)
        if ci is None or cj is None or ci == cj:
            continue
        ops[cj].add(op)
    return ops


def _pairwise_mean(ops, similarity):
    """Mean of similarity over unordered op pairs; 1.0 below 2 ops."""
    ops = sorted(ops, key=lambda o: (o.cls, o.name, o.params, o.returns))
    if len(ops) < 2:
        return 1.0
    pairs = list(combinations(ops, 2))
    return sum(similarity(a, b) for a, b in pairs) / len(pairs)


def _jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def chm(partition, calls):
    """Cohesion at Message level: mean over partitions of the mean pairwise
    f_msg = (Jaccard(returns) + Jaccard(params))/2 over the partition's
    externally-invoked operations. Range [0,1], higher is better."""
    def f_msg(a, b):
        return (_jaccard({a.returns}, {b.returns}) + _jaccard(a.params, b.params)) / 2

    ops = _published_ops(partition, calls)
    if not ops:
        return 0.0
    return sum(_pairwise_mean(o, f_msg) for o in ops.values()) / len(ops)


def _terms(op):
    import re
    words = []
    names = [op.name] + [p.split(".")[-1] for p in op.params]
    names.append(op.returns.split(".")[-1])
    for name in names:
        words.extend(re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", name))
    return {w.lower() for w in words} - _NON_DOMAIN


def chd(partition, calls):
    """Cohesion at Domain level: like chm but f_dom = Jaccard of the domain
    terms in the operation signatures. Range [0,1], higher is better."""
    def f_dom(a, b):
        return _jaccard(_terms(a), _terms(b))

    ops = _published_ops(partition, calls)
    if not ops:
        return 0.0
    return sum(_pairwise_mean(o, f_dom) for o in ops.values()) / len(ops)


def bcp(partition, class_usecases, log=math.log):
    """Business Context Purity (BCP) per Kalia et al. (Mono2Micro, ESEC/FSE 2021):
    Measures the average entropy of business use cases per partition.
    The use cases for a partition consists of all use-case labels associated with its member classes.
    A partition is considered functionally cohesive if it implements a small set of use cases.
    BCP is computed as (1/M) * sum_{i=1}^M bcp_i,
    where bcp_i is computed as - sum_{j=1}^{m_i} (1/|m_i|) * log(1/|m_i|)
    where 1/|m_i| is a vector of size m_i representing the uniform probability of use cases
    for partition i in M. For m_i use cases, bcp_i = log(m_i) (0.0 if m_i <= 1).
    Lower values indicate better recommendations (0.0 is optimal).
    """
    clusters = _clusters(partition)
    K = len(clusters)
    if K == 0:
        return 0.0
    total = 0.0
    for members in clusters.values():
        labels = set()
        for c in members:
            labels |= set(class_usecases.get(c, ()))
        m_i = len(labels)
        if m_i > 1:
            total += log(m_i)
    return total / K


def load_partition_json(path):
    """Load {class_fqn: cluster_id} from a partition JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_edges_csv(path):
    """Load edges.csv with columns src,dst,type,weight into Edge list."""
    with open(path, newline="", encoding="utf-8") as f:
        return [
            Edge(row["src"], row["dst"], row["type"], float(row["weight"]))
            for row in csv.DictReader(f)
        ]


def load_calls_csv(path):
    """Load calls.csv with caller,callee,method,params,returns into calls list."""
    with open(path, newline="", encoding="utf-8") as f:
        calls = []
        for row in csv.DictReader(f):
            caller = row["caller"]
            callee = row["callee"]
            params = tuple(row["params"].split(";")) if row.get("params") else ()
            returns = row.get("returns", "void")
            calls.append((caller, Operation(callee, row["method"], params, returns)))
        return calls


def load_entrypoints(path):
    """Load entrypoints from entrypoints.txt file."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def build_reachability_usecases(edges, entrypoints):
    """Compute class -> set of reachable entrypoints via CALL and FIELD edges."""
    from collections import defaultdict
    out = defaultdict(set)
    for e in edges:
        if e.type in ("CALL", "FIELD"):
            out[e.src].add(e.dst)
    class_usecases = defaultdict(set)
    for ep in entrypoints:
        seen = {ep}
        stack = [ep]
        while stack:
            curr = stack.pop()
            for nxt in out[curr] - seen:
                seen.add(nxt)
                stack.append(nxt)
        for c in seen:
            class_usecases[c].add(ep)
    return dict(class_usecases)


def build_package_usecases(partition):
    """Fallback: package-level domain use cases (tag = subpackage name)."""
    return {
        cls: {cls.split(".")[-2] if len(cls.split(".")) > 1 else "domain"}
        for cls in partition
    }


def compute_all(partition, edges, calls=None, class_usecases=None, ned_lo=5, ned_hi=20):
    """Compute all metrics; returns dict with SM, ICP, IFN, NED (seniors), NED_class_ratio, BCP."""
    res = {
        "SM": sm(partition, edges),
        "ICP": icp(partition, edges),
        "IFN": ifn(partition, edges),
        "NED": ned(partition, lo=ned_lo, hi=ned_hi),
        "NED_class_ratio": ned_class_ratio(partition, lo=ned_lo, hi=ned_hi),
    }
    if class_usecases is not None:
        res["BCP"] = bcp(partition, class_usecases)
    if calls:
        res["CHM"] = chm(partition, calls)
        res["CHD"] = chd(partition, calls)
    return res
