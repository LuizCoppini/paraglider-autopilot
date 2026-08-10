"""
pid_baseline_jsbsim.py — Baseline PID controller for parafoil guidance.

Avaliado nas MESMAS 32 condições do treino PPO (8 spawn positions × 4 wind
levels), no MESMO env JSBSim (ParachuteConeEnvV3) e produzindo CSV no
MESMO formato (flight_log.csv) — comparação direta head-to-head.

PROPÓSITO: responder à crítica recorrente dos revisores sobre ausência de
baseline clássico. A diferença observada entre o PID e o PPO nas mesmas
condições é a medida quantitativa do ganho do método DRL.

CONTROLLER DESIGN:
  • Aileron: PD controller sobre o erro de proa (bearing_err em graus).
    Saída em [-1, +1]. Convenção: erro positivo (alvo à direita) →
    aileron positivo (curva à direita).
  • Elevator: lógica determinística baseada em altitude AGL:
      alt > 300m   → elevator = 0   (planeio máximo)
      50 < alt ≤ 300 → ramp linear 0 → 0.5
      alt ≤ 50m    → elevator = 1   (flare total)
  • Os ganhos foram pré-calibrados empiricamente; podem ser ajustados
    via flags da CLI (--kp, --kd).

EXECUÇÃO:
  poetry run python src/rl/pid_baseline_jsbsim.py

  Opções:
    --winds 4 8 14 20    Níveis de vento (default: 4 níveis)
    --n-positions 8      Posições de spawn (default: 8)
    --kp 0.015           Ganho proporcional do aileron
    --kd 0.005           Ganho derivativo do aileron
    --flare-start 300    Altitude AGL (m) de início da ramp
    --flare-full 50      Altitude AGL (m) de flare total
"""

import argparse
import csv
import math
import os
import statistics
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


# =====================================================================
# CONFIG (idêntica ao treino PPO)
# =====================================================================

TARGET_LAT = 34.9055
TARGET_LON = -117.8830
MOJAVE_GROUND_ALT_M = 700.0

DEFAULT_WINDS_FPS = (4.0, 8.0, 14.0, 20.0)
DEFAULT_N_POSITIONS = 8
BASE_OUT_DIR = r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records"


# =====================================================================
# PID CORE
# =====================================================================

def bearing_deg(lat1, lon1, lat2, lon2):
    """Azimute (lat1,lon1) → (lat2,lon2) em graus [0, 360)."""
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2r)
    x = (math.cos(lat1r) * math.sin(lat2r)
         - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


class PIDController:
    """PID clássico com anti-windup por clipping do integrador."""

    def __init__(self, kp, ki, kd,
                 output_min=-1.0, output_max=1.0,
                 integral_limit=50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = output_min
        self.out_max = output_max
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.last_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0

    def step(self, error, dt):
        # Proporcional
        p = self.kp * error
        # Integral com anti-windup
        self.integral += error * dt
        self.integral = max(-self.integral_limit,
                            min(self.integral_limit, self.integral))
        i = self.ki * self.integral
        # Derivativo
        d = self.kd * (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error
        # Saída
        output = p + i + d
        return max(self.out_min, min(self.out_max, output))


class ParafoilPIDController:
    """
    Controlador cascateado pra parafoil:
      - Aileron: PD sobre bearing_err.
      - Elevator: lógica determinística de flare baseada em altitude AGL.
    """

    def __init__(self, target_lat, target_lon,
                 ground_alt_m=MOJAVE_GROUND_ALT_M,
                 kp_ail=0.015, ki_ail=0.0, kd_ail=0.005,
                 flare_start_agl=300.0, flare_full_agl=50.0):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.ground_alt_m = ground_alt_m

        self.aileron_pid = PIDController(
            kp=kp_ail, ki=ki_ail, kd=kd_ail,
            output_min=-1.0, output_max=1.0,
        )
        self.flare_start = flare_start_agl
        self.flare_full = flare_full_agl

    def reset(self):
        self.aileron_pid.reset()

    def step(self, env, dt=1.0):
        """
        Lê estado direto do env.fdm (sem ruído sensorial — baseline
        clássico tipicamente assume sensores ideais). Calcula ação.
        Retorna (aileron, elevator) no action space do env.
        """
        lat = env.fdm["position/lat-gc-deg"]
        lon = env.fdm["position/long-gc-deg"]
        alt_ft = env.fdm["position/h-sl-ft"]
        alt_m = alt_ft * 0.3048
        alt_agl_m = max(0.0, alt_m - self.ground_alt_m)
        heading_deg = env.fdm["attitude/psi-deg"]

        # Erro de proa (alvo à direita = positivo; à esquerda = negativo)
        target_brg = bearing_deg(lat, lon, self.target_lat, self.target_lon)
        bearing_err = (target_brg - heading_deg + 540.0) % 360.0 - 180.0

        aileron = self.aileron_pid.step(bearing_err, dt)

        # Elevator (flare logic)
        if alt_agl_m > self.flare_start:
            elevator = 0.0
        elif alt_agl_m > self.flare_full:
            frac = (self.flare_start - alt_agl_m) / (self.flare_start - self.flare_full)
            elevator = 0.5 * frac
        else:
            elevator = 1.0

        return aileron, elevator


# =====================================================================
# EVAL LOOP
# =====================================================================

def run_one_episode(controller, wind_fps, pos_id, ep_num, args):
    """
    Executa um episódio com vento e posição forçados. Retorna dict com
    métricas finais (dist, v_down, steps, total_reward).

    O env é instanciado por episódio (libera handle JSBSim entre runs).
    """
    env = ParachuteConeEnvV3(
        TARGET_LAT, TARGET_LON,
        wind_speeds_fps=(wind_fps,),
        # ATENÇÃO: o env v3 inclui sensor noise (gps, heading, alt) na obs.
        # Para o baseline PID, lemos estado direto do fdm (sem ruído) —
        # essa é a hipótese padrão do controle clássico. Se quiser fazer
        # comparação ainda mais justa, dá pra zerar os ruídos:
        #   gps_noise_m=0.0, heading_noise_deg=0.0, alt_noise_m=0.0
        # Mas isso prejudicaria o env de treino do PPO. Como aqui usamos
        # o env só pra dinâmica (PID não consome obs do env), o ruído da
        # obs não afeta o PID.
    )

    # Força (pos_id, wind) no próximo reset.
    # Como wind_speeds_fps tem só 1 valor, n_winds=1, per_pos = 125*1 = 125.
    # env.episode = pos_id * 125 → após reset, episode = pos_id*125 + 1
    # → (episode-1) // per_pos = pos_id ✓
    env.episode = pos_id * 125

    controller.reset()
    _obs, _ = env.reset()

    step_num = 0
    done = False
    truncated = False

    while not (done or truncated):
        step_num += 1
        aileron, elevator = controller.step(env, dt=1.0)
        action = np.array([aileron, elevator], dtype=np.float32)
        _obs, _reward, done, truncated, _info = env.step(action)

    # Final state
    fdm = env.fdm
    lat_f = fdm["position/lat-gc-deg"]
    lon_f = fdm["position/long-gc-deg"]
    alt_f_m = fdm["position/h-sl-ft"] * 0.3048
    v_down = abs(fdm["velocities/h-dot-fps"])
    dist_f = haversine(lat_f, lon_f, TARGET_LAT, TARGET_LON)

    # Libera handle do JSBSim
    env.fdm = None

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


def main():
    parser = argparse.ArgumentParser(description="PID baseline para parafoil em JSBSim")
    parser.add_argument("--winds", type=float, nargs="+", default=list(DEFAULT_WINDS_FPS),
                        help=f"Níveis de vento em fps (default: {DEFAULT_WINDS_FPS})")
    parser.add_argument("--n-positions", type=int, default=DEFAULT_N_POSITIONS,
                        help="Número de posições de spawn (default: 8)")
    parser.add_argument("--n-repetitions", type=int, default=80,
                        help="Repetições de cada cenário pos×wind "
                        "(default: 80 → ~2560 voos ≈ 5h wall). Cada repetição "
                        "tem proa inicial aleatória e rajadas randomizadas.")
    parser.add_argument("--kp", type=float, default=0.015, help="Ganho P do aileron")
    parser.add_argument("--ki", type=float, default=0.0, help="Ganho I do aileron")
    parser.add_argument("--kd", type=float, default=0.005, help="Ganho D do aileron")
    parser.add_argument("--flare-start", type=float, default=300.0,
                        help="Altitude AGL (m) onde começa o flare ramp")
    parser.add_argument("--flare-full", type=float, default=50.0,
                        help="Altitude AGL (m) onde elevator = 1")
    parser.add_argument("--out-dir", default=None, help="Pasta de saída")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="A cada N voos, print stats parciais (default: 50)")
    args = parser.parse_args()

    # Saída
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(BASE_OUT_DIR, f"pid_baseline_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "flight_log.csv")

    n_scenarios = len(args.winds) * args.n_positions
    n_total = n_scenarios * args.n_repetitions

    print("=" * 76)
    print(" PID BASELINE — parafoil guidance, JSBSim isolado")
    print("=" * 76)
    print(f"  target          = ({TARGET_LAT}, {TARGET_LON}) Mojave")
    print(f"  winds           = {args.winds} fps")
    print(f"  positions       = {args.n_positions}")
    print(f"  repetitions     = {args.n_repetitions} (proa inicial e rajadas aleatórias)")
    print(f"  cenários únicos = {n_scenarios}")
    print(f"  TOTAL DE VOOS   = {n_total}")
    print(f"  PID gains       = Kp={args.kp}  Ki={args.ki}  Kd={args.kd}")
    print(f"  flare ramp      = {args.flare_start}m → {args.flare_full}m AGL")
    print(f"  saída           = {csv_path}")
    print("=" * 76)
    print()

    controller = ParafoilPIDController(
        TARGET_LAT, TARGET_LON,
        kp_ail=args.kp, ki_ail=args.ki, kd_ail=args.kd,
        flare_start_agl=args.flare_start,
        flare_full_agl=args.flare_full,
    )

    all_results = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "pos_id", "wind_speed_fps", "wind_dir_deg",
            "flight_time_s", "final_dist", "reward",
            "lat", "lon", "gr",
            "v_down_final_fps", "alt_final_m",
            "repetition",  # extra: índice da repetição dentro do cenário
        ])

        ep_num = 0
        t_start = time.time()

        # Ordem: repetição externa, vento intermediário, posição interna.
        # Isso garante que se o usuário interromper no meio, a distribuição
        # já tem cobertura uniforme dos cenários (cada rep cobre todos os 32).
        for rep in range(args.n_repetitions):
            for wind in args.winds:
                for pos_id in range(args.n_positions):
                    ep_num += 1

                    t_ep = time.time()
                    try:
                        result = run_one_episode(controller, wind, pos_id, ep_num, args)
                    except Exception as e:
                        print(f"  [Ep {ep_num}] EXCEÇÃO: {e} (pulando)")
                        continue
                    t_dur = time.time() - t_ep

                    # Detecta NaN/inf no resultado
                    if not (math.isfinite(result["dist_final"])
                            and math.isfinite(result["v_down_final"])):
                        print(f"  [Ep {ep_num}] valores não-finitos (pulando do log)")
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

                    # Progress + ETA
                    elapsed = time.time() - t_start
                    avg_per_ep = elapsed / ep_num
                    remaining = n_total - ep_num
                    eta_s = remaining * avg_per_ep
                    eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.1f}min"

                    # Print compacto a cada voo
                    print(f"[{ep_num:>5d}/{n_total}] rep={rep:>2d} "
                          f"w={wind:>4.0f} p={pos_id} "
                          f"d_f={result['dist_final']:>7.1f}m "
                          f"v_d={result['v_down_final']:>5.1f}fps "
                          f"({t_dur:.1f}s) ETA={eta_str}")

                    # Checkpoint de stats a cada N voos
                    if ep_num % args.checkpoint_every == 0:
                        _print_partial_stats(all_results, args.winds, ep_num, n_total)

        t_total = time.time() - t_start

    # --- Stats sumárias finais ---
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
    print("  1. Compare este CSV com o flight_log.csv do treino PPO (mesma estrutura).")
    print("  2. Use compare_pid_vs_ppo.py pra gerar a Tabela 4 final.")


def _percentile(values, p):
    """Percentil sem depender de numpy. p ∈ [0, 100]. Robusto a NaN."""
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
    """Stats parciais durante a execução, sem ser tão verboso."""
    pct = 100 * ep_done / ep_total
    dists = [r["dist_final"] for r in results]
    med = _percentile(dists, 50)
    succ = sum(1 for d in dists if math.isfinite(d) and d < 500) / max(1, len(dists))
    print(f"  ─── checkpoint @ {ep_done}/{ep_total} ({pct:.1f}%): "
          f"global d_f median = {med:.1f}m, success = {100*succ:.1f}% ───")


def _print_final_stats(results, winds):
    """Tabela formatada com mediana, quartis, v_d e success rate por vento."""
    print(f"{'Wind (fps)':>10} | {'n':>5} | {'d_f median':>11} | {'d_f q1':>8} | "
          f"{'d_f q3':>8} | {'v_d median':>11} | {'success<500m':>15}")
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
              f"{q3:>8.1f} | {vmed:>11.1f} | {success:>5d}/{n} ({100*success/n:>5.1f}%)")

    # Global
    all_dists = [r["dist_final"] for r in results]
    all_vds = [r["v_down_final"] for r in results]
    n = len(all_dists)
    print("-" * 76)
    succ_n = sum(1 for d in all_dists if math.isfinite(d) and d < 500)
    print(f"{'GLOBAL':>10} | {n:>5d} | "
          f"{_percentile(all_dists, 50):>11.1f} | "
          f"{_percentile(all_dists, 25):>8.1f} | "
          f"{_percentile(all_dists, 75):>8.1f} | "
          f"{_percentile(all_vds, 50):>11.1f} | "
          f"{succ_n:>5d}/{n} ({100*succ_n/n:>5.1f}%)")


if __name__ == "__main__":
    main()