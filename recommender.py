import pandas as pd
import numpy as np
import heapq
from pathlib import Path

# -------------------------
# Industry templates
# -------------------------
WEIGHTS = {
    "office":           dict(cpu=0.50, gpu=0.05, ram=0.30, mb=0.10, psu=0.05),
    "gaming":           dict(cpu=0.25, gpu=0.55, ram=0.10, mb=0.05, psu=0.05),
    "engineering":      dict(cpu=0.45, gpu=0.20, ram=0.25, mb=0.05, psu=0.05),
    "content_creation": dict(cpu=0.30, gpu=0.45, ram=0.15, mb=0.05, psu=0.05),
}

def mins(industry):
    if industry == "office":
        return dict(min_cores=4, min_vram=0,  min_ram=8,  min_psu=350)
    if industry == "gaming":
        return dict(min_cores=6, min_vram=8,  min_ram=16, min_psu=550)
    if industry == "engineering":
        return dict(min_cores=8, min_vram=6,  min_ram=32, min_psu=650, min_slots=4)
    if industry == "content_creation":
        return dict(min_cores=8, min_vram=10, min_ram=32, min_psu=650)
    return dict(min_cores=6, min_vram=8,  min_ram=16, min_psu=550)

# -------------------------
# Reasonable per-part caps (NOT strict fractions)
# Avoids "MB <= $100" / "PSU <= $100" failures at lower budgets.
# -------------------------
MAX_FRAC = dict(cpu=0.65, gpu=0.80, ram=0.45, mb=0.40, psu=0.30)
MIN_FLOOR = dict(cpu=0.0, gpu=0.0, ram=0.0, mb=120.0, psu=80.0)

def part_cap(total_budget: float, part: str) -> float:
    cap = max(MIN_FLOOR.get(part, 0.0), float(total_budget) * MAX_FRAC.get(part, 1.0))
    return min(cap, float(total_budget))

# -------------------------
# Helpers
# -------------------------
def to_num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").fillna(default)

def norm_ddr(x):
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).upper().strip()
    if "DDR5" in s: return "DDR5"
    if "DDR4" in s: return "DDR4"
    if "DDR3" in s: return "DDR3"
    return "UNKNOWN"

def safe_str(x, default="Unknown"):
    if pd.isna(x):
        return default
    s = str(x).strip()
    return s if s else default

def priced(df):
    df = df.copy()
    df["price"] = to_num(df.get("price"), np.nan)
    return df[df["price"].notna() & (df["price"] > 0)].copy()

def gpu_watts_proxy(vram_gb: float) -> float:
    if vram_gb <= 4:  return 75
    if vram_gb <= 6:  return 120
    if vram_gb <= 8:  return 170
    if vram_gb <= 12: return 230
    if vram_gb <= 16: return 285
    if vram_gb <= 24: return 350
    return 420

def est_draw(cpu_row, gpu_row) -> float:
    return float(cpu_row["tdp"]) + gpu_watts_proxy(float(gpu_row["vram_gb"])) + 150.0

def norm01(x, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0, 1))

def util_score(total_price: float, budget: float, target: float = 0.95) -> float:
    if budget <= 0:
        return 0.0
    u = total_price / budget
    return float(np.clip(1.0 - abs(u - target) / target, 0.0, 1.0))

# -------------------------
# Load CSVs (single folder)
# -------------------------
def load_csv_parts(data_dir: str = "data") -> dict:
    data_path = Path(data_dir)

    files = {
        "cpu": data_path / "CPU.csv",
        "gpu": data_path / "GPU.csv",
        "ram": data_path / "RAM.csv",
        "mb":  data_path / "Motherboard.csv",
        "psu": data_path / "PSU.csv",
    }

    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing CSV(s) in '{data_dir}': {missing}. "
            f"Expected files: CPU.csv, GPU.csv, RAM.csv, Motherboard.csv, PSU.csv"
        )

    return {
        "cpu": pd.read_csv(files["cpu"]),
        "gpu": pd.read_csv(files["gpu"]),
        "ram": pd.read_csv(files["ram"]),
        "mb":  pd.read_csv(files["mb"]),
        "psu": pd.read_csv(files["psu"]),
    }

def recommend_builds_from_csv_dir(
    data_dir: str,
    industry: str,
    total_budget: float,
    top_k: int = 50,
    random_state: int = 42,
):
    dfs = load_csv_parts(data_dir=data_dir)
    return recommend_builds(
        dfs["cpu"], dfs["gpu"], dfs["ram"], dfs["mb"], dfs["psu"],
        industry=industry,
        total_budget=total_budget,
        top_k=top_k,
        random_state=random_state,
    )

# -------------------------
# Main recommender
# -------------------------
def recommend_builds(
    cpu_df: pd.DataFrame,
    gpu_df: pd.DataFrame,
    ram_df: pd.DataFrame,
    mb_df: pd.DataFrame,
    psu_df: pd.DataFrame,
    industry: str,
    total_budget: float,
    top_k: int = 50,
    random_state: int = 42,
):
    if industry not in WEIGHTS:
        industry = "gaming"

    req = mins(industry)
    W = WEIGHTS[industry]

    cpu_df, gpu_df, ram_df, mb_df, psu_df = cpu_df.copy(), gpu_df.copy(), ram_df.copy(), mb_df.copy(), psu_df.copy()

    # CPU
    cpu_df["model"] = cpu_df.get("model", cpu_df.get("name", "Unknown"))
    cpu_df["cores"] = to_num(cpu_df.get("cores", cpu_df.get("core_count")), 4).astype(int)
    cpu_df["tdp"] = to_num(cpu_df.get("tdp"), 65)
    cpu_df["boost_ghz"] = to_num(cpu_df.get("boost_clock"), np.nan) / 1e9
    cpu_df["cpu_socket"] = cpu_df.get("cpu_socket", cpu_df.get("socket"))
    cpu_df["cpu_socket_norm"] = cpu_df["cpu_socket"].astype(str).str.upper().str.replace(" ", "")

    # GPU
    gpu_df["model"] = gpu_df.get("model", gpu_df.get("name", "Unknown"))
    gpu_df["vram_gb"] = to_num(gpu_df.get("vram"), 0) / 1e9
    gpu_df["boost_ghz"] = to_num(gpu_df.get("boost_clock"), np.nan) / 1e9

    # RAM
    ram_df["model"] = ram_df.get("model", ram_df.get("name", "Unknown"))
    ram_df["modules"] = to_num(ram_df.get("number_of_modules"), 0).astype(int)
    ram_df["module_gb"] = to_num(ram_df.get("module_size"), np.nan) / 1e9
    ram_df["total_ram_gb"] = ram_df["modules"] * ram_df["module_gb"]
    ram_df["ram_ddr"] = ram_df.get("module_type", "").apply(norm_ddr)

    # MB
    mb_df["model"] = mb_df.get("model", mb_df.get("name", "Unknown"))
    mb_df["socket"] = mb_df.get("socket", "UNKNOWN")
    mb_df["mb_socket_norm"] = mb_df["socket"].astype(str).str.upper().str.replace(" ", "")
    mb_df["ram_slots"] = to_num(mb_df.get("ram_slots"), 0).astype(int)
    mb_df["mb_ddr"] = mb_df.get("mb_ddr", mb_df.get("ram_type", "")).apply(norm_ddr)

    # PSU
    psu_df["model"] = psu_df.get("model", psu_df.get("name", "Unknown"))
    psu_df["wattage"] = to_num(psu_df.get("wattage"), 0)

    # priced-only
    cpu_df = priced(cpu_df)
    gpu_df = priced(gpu_df)
    ram_df = priced(ram_df)
    mb_df  = priced(mb_df)
    psu_df = priced(psu_df)

    # Prune by requirements + reasonable caps
    cpu_cap = part_cap(total_budget, "cpu")
    gpu_cap = part_cap(total_budget, "gpu")
    ram_cap = part_cap(total_budget, "ram")
    mb_cap  = part_cap(total_budget, "mb")
    psu_cap = part_cap(total_budget, "psu")

    cpu_f = cpu_df[(cpu_df["cores"] >= req["min_cores"]) & (cpu_df["price"] <= cpu_cap)].copy()
    gpu_f = gpu_df[(gpu_df["vram_gb"] >= req["min_vram"]) & (gpu_df["price"] <= gpu_cap)].copy()
    ram_f = ram_df[
        (ram_df["total_ram_gb"].fillna(0) >= req["min_ram"]) &
        (ram_df["modules"] > 0) &
        (ram_df["price"] <= ram_cap)
    ].copy()
    mb_f = mb_df[(mb_df["ram_slots"] > 0) & (mb_df["price"] <= mb_cap)].copy()
    psu_f = psu_df[(psu_df["wattage"] >= req["min_psu"]) & (psu_df["price"] <= psu_cap)].copy()

    if "min_slots" in req:
        mb_f = mb_f[mb_f["ram_slots"] >= req["min_slots"]].copy()

    if any(len(x) == 0 for x in [cpu_f, gpu_f, ram_f, mb_f, psu_f]):
        return pd.DataFrame()

    # performance
    cpu_f["cpu_perf"] = cpu_f["cores"] + 0.5 * cpu_f["boost_ghz"].fillna(cpu_f["boost_ghz"].median())
    gpu_f["gpu_perf"] = gpu_f["vram_gb"] + 0.3 * gpu_f["boost_ghz"].fillna(gpu_f["boost_ghz"].median())
    ram_f["ram_perf"] = ram_f["total_ram_gb"].fillna(0)
    mb_f["mb_perf"]   = mb_f["ram_slots"] + (mb_f["mb_ddr"].eq("DDR5") * 0.1)
    psu_f["psu_perf"] = psu_f["wattage"]

    # Candidate pools
    cpu_top = cpu_f.sort_values(["cpu_perf", "price"], ascending=[False, True]).head(220)
    gpu_top = gpu_f.sort_values(["gpu_perf", "price"], ascending=[False, True]).head(300)
    ram_top = ram_f.sort_values(["ram_perf", "price"], ascending=[False, True]).head(300)
    mb_top  = mb_f.sort_values(["mb_perf",  "price"], ascending=[False, True]).head(260)
    psu_top = psu_f.sort_values(["psu_perf", "price"], ascending=[False, True]).head(220)

    cpu_lo, cpu_hi = cpu_top["cpu_perf"].min(), cpu_top["cpu_perf"].max()
    gpu_lo, gpu_hi = gpu_top["gpu_perf"].min(), gpu_top["gpu_perf"].max()
    ram_lo, ram_hi = ram_top["ram_perf"].min(), ram_top["ram_perf"].max()
    mb_lo,  mb_hi  = mb_top["mb_perf"].min(),  mb_top["mb_perf"].max()
    psu_lo, psu_hi = psu_top["psu_perf"].min(), psu_top["psu_perf"].max()

    # bundles by (socket, ddr)
    bundles_by_key = {}
    for _, mb in mb_top.iterrows():
        key = (mb["mb_socket_norm"], str(mb["mb_ddr"]).upper())
        bundles_by_key.setdefault(key, [])

        ok_ram = ram_top[ram_top["modules"] <= int(mb["ram_slots"])]
        if mb["mb_ddr"] != "UNKNOWN":
            ok_ram = ok_ram[ok_ram["ram_ddr"] == mb["mb_ddr"]]
        if ok_ram.empty:
            continue

        for _, ram in ok_ram.head(6).iterrows():
            bundles_by_key[key].append((mb, ram))

    def socket_ok(cpu_row, mb_row):
        cs = cpu_row["cpu_socket_norm"]
        ms = mb_row["mb_socket_norm"]
        if pd.isna(cs) or cs in ("", "NAN", "NONE"):
            return False
        return cs == ms

    def ddr_ok(ram_row, mb_row):
        mb_ddr = str(mb_row["mb_ddr"]).upper()
        ram_ddr = str(ram_row["ram_ddr"]).upper()
        if mb_ddr == "UNKNOWN" or ram_ddr == "UNKNOWN":
            return False
        return mb_ddr == ram_ddr

    def ram_slots_ok(ram_row, mb_row):
        return int(ram_row["modules"]) <= int(mb_row["ram_slots"])

    def psu_ok(cpu_row, gpu_row, psu_row):
        return float(psu_row["wattage"]) >= 0.90 * est_draw(cpu_row, gpu_row)

    # Search throttles
    CPU_PER_GPU = 70
    PSU_PER_PAIR = 12
    MBRAM_BUNDLES_PER_PAIR = 26

    # Slightly more emphasis on spending near budget
    PERF_WEIGHT = 0.70
    UTIL_WEIGHT = 0.30
    UTIL_TARGET = 0.95

    KEEP = top_k * 8
    heap = []
    tie = 0

    def push(score, build):
        nonlocal tie
        tie += 1
        item = (score, tie, build)
        if len(heap) < KEEP:
            heapq.heappush(heap, item)
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, item)

    for _, gpu in gpu_top.iterrows():
        for _, cpu in cpu_top.head(CPU_PER_GPU).iterrows():
            if pd.isna(cpu["cpu_socket"]) or str(cpu["cpu_socket"]).strip() == "":
                continue

            draw = est_draw(cpu, gpu)
            psu_candidates = psu_top[psu_top["wattage"] >= 0.90 * draw].head(PSU_PER_PAIR)
            if psu_candidates.empty:
                continue

            key_candidates = [
                (cpu["cpu_socket_norm"], "DDR5"),
                (cpu["cpu_socket_norm"], "DDR4"),
                (cpu["cpu_socket_norm"], "DDR3"),
            ]

            bundles = []
            for k in key_candidates:
                if k in bundles_by_key:
                    bundles.extend(bundles_by_key[k][:MBRAM_BUNDLES_PER_PAIR])

            if not bundles:
                continue

            for _, psu in psu_candidates.iterrows():
                for mb, ram in bundles:
                    if not socket_ok(cpu, mb):            continue
                    if not ddr_ok(ram, mb):               continue
                    if not ram_slots_ok(ram, mb):         continue
                    if not psu_ok(cpu, gpu, psu):         continue

                    total_price = float(cpu["price"] + gpu["price"] + mb["price"] + ram["price"] + psu["price"])
                    if total_price > total_budget:
                        continue  # hard budget wall

                    perf_score = (
                        W["cpu"] * norm01(cpu["cpu_perf"], cpu_lo, cpu_hi) +
                        W["gpu"] * norm01(gpu["gpu_perf"], gpu_lo, gpu_hi) +
                        W["ram"] * norm01(ram["ram_perf"], ram_lo, ram_hi) +
                        W["mb"]  * norm01(mb["mb_perf"],  mb_lo,  mb_hi) +
                        W["psu"] * norm01(psu["psu_perf"], psu_lo, psu_hi)
                    )
                    u = util_score(total_price, total_budget, UTIL_TARGET)
                    final_score = PERF_WEIGHT * perf_score + UTIL_WEIGHT * u

                    build = {
                        "industry": industry,
                        "final_score": round(final_score, 4),
                        "perf_score": round(perf_score, 4),
                        "util_score": round(u, 4),
                        "total_price": round(total_price, 2),

                        "cpu": safe_str(cpu.get("model")),
                        "cpu_socket": safe_str(cpu.get("cpu_socket")),
                        "cpu_cores": int(cpu["cores"]),
                        "cpu_price": float(cpu["price"]),

                        "gpu": safe_str(gpu.get("model")),
                        "gpu_vram_gb": round(float(gpu["vram_gb"]), 1),
                        "gpu_price": float(gpu["price"]),

                        "motherboard": safe_str(mb.get("model")),
                        "mb_socket": safe_str(mb.get("socket")),
                        "mb_ddr": safe_str(mb.get("mb_ddr")),
                        "mb_price": float(mb["price"]),

                        "ram": safe_str(ram.get("model")),
                        "ram_total_gb": int(round(float(ram["total_ram_gb"]))) if pd.notna(ram["total_ram_gb"]) else None,
                        "ram_ddr": safe_str(ram.get("ram_ddr")),
                        "ram_price": float(ram["price"]),

                        "psu": safe_str(psu.get("model")),
                        "psu_wattage": int(round(float(psu["wattage"]))),
                        "psu_price": float(psu["price"]),

                        "est_draw_w": int(round(draw)),
                    }
                    push(final_score, build)

    results = [x[2] for x in sorted(heap, key=lambda t: t[0], reverse=True)][:top_k]
    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("final_score", ascending=False)
