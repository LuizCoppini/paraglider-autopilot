"""
eval_ppo_jsbsim.py — Avaliação rigorosa do modelo PPO cone v3 final.

Carrega o modelo treinado (congelado, sem aprendizado) e roda EXATAMENTE
o mesmo protocolo experimental do pid_baseline_jsbsim.py:
  • Mesma env (ParachuteConeEnvV3)
  • Mesmas 32 condições (8 spawn positions × 4 wind levels)
  • Mesmo número de repetições configurável (--n-repetitions)
  • Mesmo formato de flight_log.csv → drop-in no compare_pid_vs_ppo.py

PROPÓSITO: produzir uma avaliação científicamente comparável ao PID
baseline. O log do treino (flight_log.csv do training_*) mistura todas
as fases de aprendizado (incluindo colapsos e recuperações); esta
avaliação usa o checkpoint final em modo determinístico, garantindo
reprodutibilidade.

USO:
  poetry run python src/rl/eval_ppo_jsbsim.py --n-repetitions 100

  Opções:
    --model      Caminho do .zip do PPO (default: cone v3 final)
    --vecnorm    Caminho do VecNormalize .pkl (default: cone v3 final)
    --winds      Níveis de vento (default: 4 8 14 20)
    --n-repetitions N    Repetições por cenário (default: 100)
"""

import argparse
import csv
import math
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np

# Adiciona src/ ao path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from rl.parachute_cone_env_v3 import ParachuteConeEnvV3, haversine
except ModuleNotFoundError:
    from parachute_cone_env_v3 import ParachuteConeEnvV3, haversine

from stable_baselines3 import PPO


# =====================================================================
# CONFIG (idêntica ao PID baseline)
# =====================================================================

TARGET_LAT = 34.9055
TARGET_LON = -117.8830

DEFAULT_MODEL_PATH = (
    r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_v3_method"
    r"\training_20260601_230833\parachute_cone_v3_model_final.zip"
)
DEFAULT_VECNORM_PATH = (
    r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_v3_method"
    r"\training_20260601_230833\vec_normalize_cone_v3.pkl"
)

DEFAULT_WINDS_FPS = (4.0, 8.0, 14.0, 20.0)
DEFAULT_N_POSITIONS = 8
BASE_OUT_DIR = r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records"


# =====================================================================
# EPISÓDIO
# =====================================================================

def run_one_episode(env, model, vec_norm, wind_fps, pos_id,
                    wind_idx, n_winds, ep_num):
    """
    Roda um episódio com o modelo PPO. Estrutura idêntica ao
    pid_baseline_jsbsim.run_one_episode — só muda a fonte da ação.

    Sequência:
      1. env.episode forçado pra ditar (pos_id, wind_idx)
      2. env.reset() retorna obs RAW (6 floats)
      3. Loop: obs → VecNormalize → PPO.predict(deterministic=True) →
               action → env.step(action)
      4. Termina quando done ou truncated
      5. Lê estado final do fdm pra métricas
    """
    per_pos = 125 * n_winds
    env.episode = 125 * wind_idx + per_pos * pos_id

    obs, _ = env.reset()

    step_num = 0
    done = False
    truncated = False

    while not (done or truncated):
        step_num += 1
        # Normaliza obs (VecNormalize aprendido durante o treino)
        obs_norm = vec_norm.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(obs_norm, deterministic=True)
        action = action.flatten().astype(np.float32)
        obs, _reward, done, truncated, _info = env.step(action)

    fdm = env.fdm
    lat_f = fdm["position/lat-gc-deg"]
    lon_f = fdm["position/long-gc-deg"]
    alt_f_m = fdm["position/h-sl-ft"] * 0.3048
    v_down = abs(fdm["velocities/h-dot-fps"])
    dist_f = haversine(lat_f, lon_f, TARGET_LAT, TARGET_LON)

    return {
        "episode": ep_num,
        "pos_id": pos_id,
        "wind_fps": wind_fps,
        "wind_dir_deg": 0.0,
        "steps": step_num,
        "dist_final": dist_f,
        "v_down_final": v_down,
        "alt_final_m": alt_f_m,
        "lat_final": lat_f,
        "lon_final": lon_f,
        "total_reward": env.total_reward,
        "gr": 1.88,
    }


# =====================================================================
# STATS HELPERS (clones do PID baseline)
# =====================================================================

def _percentile(values, p):
    finite = sorted(v for v in values if math.isfinite(v))
    n = len(finite)
    if n == 0:
        return float("nan")
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return finite[f]
    return finite[f] + (k - f) * (finite[c] - finite[f])


def _print_partial_stats(results, winds, ep_done, ep_total):
    pct = 100 * ep_done / ep_total
    dists = [r["dist_final"] for r in results]
    med = _percentile(dists, 50)
    succ = sum(1 for d in dists if math.isfinite(d) and d < 500) / max(1, len(dists))
    print(f"  ─── checkpoint @ {ep_done}/{ep_total} ({pct:.1f}%): "
          f"global d_f median = {med:.1f}m, success = {100*succ:.1f}% ───")


def _print_final_stats(results, winds):
    print(f"{'Wind (fps)':>10} | {'n':>5} | {'d_f median':>11} | "
          f"{'d_f q1':>8} | {'d_f q3':>8} | "
          f"{'v_d median':>11} | {'success<500m':>15}")
    print("-" * 76)

    for wind in winds:
        subset = [r for r in results if r["wind_fps"] == wind]
        if not subset:
            continue
        dists = [r["dist_final"] for r in subset]
        vds = [r["v_down_final"] for r in subset]
        n = len(dists)
        med = _percentile(dists, 50)
        q1 = _percentile(dists, 25)
        q3 = _percentile(dists, 75)
        vmed = _percentile(vds, 50)
        success = sum(1 for d in dists if math.isfinite(d) and d < 500)

        print(f"{wind:>10.1f} | {n:>5d} | {med:>11.1f} | {q1:>8.1f} | "
              f"{q3:>8.1f} | {vmed:>11.1f} | "
              f"{success:>5d}/{n} ({100*success/n:>5.1f}%)")

    all_dists = [r["dist_final"] for r in results]
    all_vds = [r["v_down_final"] for r in results]
    n = len(all_dists)
    succ_n = sum(1 for d in all_dists if math.isfinite(d) and d < 500)
    print("-" * 76)
    print(f"{'GLOBAL':>10} | {n:>5d} | "
          f"{_percentile(all_dists, 50):>11.1f} | "
          f"{_percentile(all_dists, 25):>8.1f} | "
          f"{_percentile(all_dists, 75):>8.1f} | "
          f"{_percentile(all_vds, 50):>11.1f} | "
          f"{succ_n:>5d}/{n} ({100*succ_n/n:>5.1f}%)")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Avaliação PPO cone v3 (modelo congelado)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path do .zip do PPO")
    parser.add_argument("--vecnorm", default=DEFAULT_VECNORM_PATH,
                        help="Path do .pkl do VecNormalize")
    parser.add_argument("--winds", type=float, nargs="+",
                        default=list(DEFAULT_WINDS_FPS))
    parser.add_argument("--n-positions", type=int, default=DEFAULT_N_POSITIONS)
    parser.add_argument("--n-repetitions", type=int, default=100,
                        help="Repetições por cenário (default: 100 = 3200 voos)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    # Validação dos paths
    if not os.path.exists(args.model):
        print(f"ERRO: modelo PPO não encontrado: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.vecnorm):
        print(f"ERRO: VecNormalize não encontrado: {args.vecnorm}")
        sys.exit(1)

    # Saída
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(BASE_OUT_DIR, f"eval_ppo_final_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "flight_log.csv")

    n_scenarios = len(args.winds) * args.n_positions
    n_total = n_scenarios * args.n_repetitions

    # Carrega modelo + VecNormalize
    print("[init] Carregando modelo PPO...")
    model = PPO.load(args.model)
    print(f"[init] Modelo carregado: {args.model}")

    print("[init] Carregando VecNormalize...")
    with open(args.vecnorm, "rb") as f:
        vec_norm = pickle.load(f)
    # Garante modo de avaliação (não atualizar estatísticas)
    vec_norm.training = False
    vec_norm.norm_reward = False
    print(f"[init] VecNormalize carregado em modo eval.")
    print(f"        obs_mean = {vec_norm.obs_rms.mean}")
    print(f"        obs_var  = {vec_norm.obs_rms.var}")

    print()
    print("=" * 76)
    print(" PPO EVAL (modelo final cone v3, modo determinístico)")
    print("=" * 76)
    print(f"  target          = ({TARGET_LAT}, {TARGET_LON}) Mojave")
    print(f"  winds           = {args.winds} fps")
    print(f"  positions       = {args.n_positions}")
    print(f"  repetitions     = {args.n_repetitions}")
    print(f"  TOTAL DE VOOS   = {n_total}")
    print(f"  modelo          = {os.path.basename(args.model)}")
    print(f"  vecnorm         = {os.path.basename(args.vecnorm)}")
    print(f"  saída           = {csv_path}")
    print("=" * 76)
    print()

    # >>> UM ÚNICO env reusado (mesma estratégia do PID baseline) <<<
    print("[init] Criando env JSBSim (reusado em todos os episódios)...")
    env = ParachuteConeEnvV3(
        TARGET_LAT, TARGET_LON,
        wind_speeds_fps=tuple(args.winds),
    )
    n_winds = len(args.winds)
    print(f"[init] Env criado. log_file interno = {env.log_file}")
    print()

    all_results = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "pos_id", "wind_speed_fps", "wind_dir_deg",
            "flight_time_s", "final_dist", "reward",
            "lat", "lon", "gr",
            "v_down_final_fps", "alt_final_m",
            "repetition",
        ])

        ep_num = 0
        t_start = time.time()

        for rep in range(args.n_repetitions):
            for wind_idx, wind in enumerate(args.winds):
                for pos_id in range(args.n_positions):
                    ep_num += 1

                    t_ep = time.time()
                    try:
                        result = run_one_episode(
                            env, model, vec_norm, wind, pos_id,
                            wind_idx, n_winds, ep_num,
                        )
                    except Exception as e:
                        print(f"  [Ep {ep_num}] EXCEÇÃO: {e} (pulando)")
                        continue
                    t_dur = time.time() - t_ep

                    if not (math.isfinite(result["dist_final"])
                            and math.isfinite(result["v_down_final"])):
                        print(f"  [Ep {ep_num}] valores não-finitos (pulando)")
                        continue

                    writer.writerow([
                        result["episode"], result["pos_id"],
                        result["wind_fps"], result["wind_dir_deg"],
                        result["steps"], round(result["dist_final"], 2),
                        round(result["total_reward"], 1),
                        round(result["lat_final"], 7), round(result["lon_final"], 7),
                        result["gr"],
                        round(result["v_down_final"], 2),
                        round(result["alt_final_m"], 2),
                        rep,
                    ])
                    f.flush()
                    all_results.append(result)

                    elapsed = time.time() - t_start
                    avg_per_ep = elapsed / ep_num
                    remaining = n_total - ep_num
                    eta_s = remaining * avg_per_ep
                    eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.1f}min"

                    print(f"[{ep_num:>5d}/{n_total}] rep={rep:>2d} "
                          f"w={wind:>4.0f} p={pos_id} "
                          f"d_f={result['dist_final']:>7.1f}m "
                          f"v_d={result['v_down_final']:>5.1f}fps "
                          f"({t_dur:.1f}s) ETA={eta_str}")

                    if ep_num % args.checkpoint_every == 0:
                        _print_partial_stats(all_results, args.winds, ep_num, n_total)

        t_total = time.time() - t_start

    # Sumário final
    print()
    print("=" * 76)
    print(" RESUMO ESTATÍSTICO FINAL — POR VENTO")
    print("=" * 76)
    _print_final_stats(all_results, args.winds)

    print()
    print(f"Tempo total: {t_total/3600:.2f} h ({t_total/60:.1f} min)")
    print(f"Episódios computados (no log): {len(all_results)}/{n_total}")
    print(f"CSV salvo:   {csv_path}")
    print()
    print("Próximos passos:")
    print(f"  poetry run python src/rl/compare_pid_vs_ppo.py \\")
    print(f"      --pid-csv \"<path PID flight_log.csv>\" \\")
    print(f"      --ppo-csv \"{csv_path}\" \\")
    print(f"      --out-tex tabela4_pid_vs_ppo.tex")


if __name__ == "__main__":
    main()
