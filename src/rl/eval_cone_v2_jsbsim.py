"""
eval_cone_v2_jsbsim.py — Avalia o cone model V2 em JSBSim PURO.

Diferenças em relação ao eval_cone_jsbsim.py (v1):
  - Usa ParachuteConeEnvV2 (env do treino v2) em vez do v1
  - Paths apontam para o treino v2 (cone_v2_method)
  - GR=1.88 já é default do env v2
  - Vento norte→sul e proa aleatória já são default do env v2

Como rodar:
  Coloque em src/rl/eval_cone_v2_jsbsim.py do seu projeto:
    .venv\\Scripts\\python.exe src\\rl\\eval_cone_v2_jsbsim.py
"""

import os
import sys
import csv
import math
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.parachute_cone_env_v2 import ParachuteConeEnvV2, haversine

# --- CONFIGURAÇÕES (ajuste o timestamp da pasta do seu treino) ---
MODEL_PATH = r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_v2_method\training_20260502_114555\parachute_cone_v2_model_final.zip"
VEC_NORMALIZE_PATH = r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_v2_method\training_20260502_114555\vec_normalize_cone_v2.pkl"

TARGET_LAT = 34.9055
TARGET_LON = -117.8830
GLIDE_RATIO = 1.88   # mesmo do treino v2


def main():
    print("=" * 70)
    print("Avaliação cone v2 em JSBSim PURO")
    print("=" * 70)
    print(f"Target:  lat={TARGET_LAT}, lon={TARGET_LON} (Mojave)")
    print(f"GR:      {GLIDE_RATIO}")
    print(f"Modelo:  {MODEL_PATH}")
    print()

    # 1. Carrega modelo
    model = PPO.load(MODEL_PATH)
    print("[1/3] Modelo PPO carregado.")

    # 2. Cria env (v2, com vento norte-sul e heading aleatório).
    # IMPORTANTE: passar a MESMA faixa de vento do treino. Caso contrário
    # o eval tenta condições que o modelo nunca viu (extrapolação).
    raw_env = ParachuteConeEnvV2(
        TARGET_LAT, TARGET_LON,
        glide_ratio_target=GLIDE_RATIO,
        include_gr_in_obs=False,
        wind_speeds_fps=(4.0, 8.0, 14.0, 20.0),  # = WIND_SPEEDS_FPS do treino
    )
    venv = DummyVecEnv([lambda: raw_env])
    env = VecNormalize.load(VEC_NORMALIZE_PATH, venv)
    env.training = False
    env.norm_reward = False
    print(f"[2/3] VecNormalize carregado.")
    print(f"      obs_mean = {env.obs_rms.mean}")
    print(f"      obs_var  = {env.obs_rms.var}")
    print(f"      log_file = {raw_env.log_file}")
    print()

    # 3. Roda 32 episódios cobrindo 8 posições × 4 níveis de vento
    configs = [(p, w) for p in range(8) for w in range(4)]

    print("[3/3] Rodando 32 episódios (8 posições × 4 ventos)...")
    print()
    print(" ep | pos | vento (fps@°)  | passos")
    print("----|-----|----------------|--------")

    for i, (p, w) in enumerate(configs, 1):
        # Força (pos_id=p, wind_type_id=w) no próximo reset
        raw_env.episode = 500 * p + 125 * w

        obs = env.reset()
        done = [False]
        steps = 0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
            if steps > 5000:
                break

        wind_speed = raw_env.current_wind_speed
        wind_dir = raw_env.current_wind_dir
        print(f" {i:2d} |  {p}  | {wind_speed:4.0f} @ {wind_dir:5.0f}° | {steps:5d}")

    # 4. Lê o flight_log.csv (uma linha por episódio)
    print()
    print(f"Lendo {raw_env.log_file}")
    rows = []
    with open(raw_env.log_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("ERRO: flight_log.csv vazio.")
        return

    print(f"Lidas {len(rows)} linhas do log.")
    print()
    print(" ep | pos | vento (fps@°)  |  t (s) |  dist fim (m) |  reward total | lat fim    | lon fim")
    print("----|-----|----------------|--------|---------------|---------------|------------|-----------")

    distances = []
    rewards = []
    for r in rows:
        ep = int(r["episode"])
        pos_id = int(r["pos_id"])
        ws = float(r["wind_speed_fps"])
        wd = float(r["wind_dir_deg"])
        ft = float(r["flight_time_s"])
        d = float(r["final_dist"])
        rew = float(r["reward"])
        lat = float(r["lat"])
        lon = float(r["lon"])
        distances.append(d)
        rewards.append(rew)
        print(f" {ep:4d} |  {pos_id}  | {ws:4.0f} @ {wd:5.0f}° | {ft:6.0f} | {d:13.2f} | {rew:13.0f} | {lat:10.5f} | {lon:10.5f}")

    distances = np.array(distances)
    rewards = np.array(rewards)
    print()
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"n             = {len(distances)}")
    print(f"distância (m) — min={np.min(distances):.1f}  mediana={np.median(distances):.1f}  "
          f"média={np.mean(distances):.1f}  max={np.max(distances):.1f}  std={np.std(distances):.1f}")
    print(f"reward total  — min={np.min(rewards):.0f}  mediana={np.median(rewards):.0f}  "
          f"média={np.mean(rewards):.0f}  max={np.max(rewards):.0f}")
    print()
    n = len(distances)
    print(f"Voos < 100m  : {np.sum(distances < 100):2d}/{n} ({100*np.sum(distances<100)/n:5.1f}%)")
    print(f"Voos < 500m  : {np.sum(distances < 500):2d}/{n} ({100*np.sum(distances<500)/n:5.1f}%)")
    print(f"Voos < 1000m : {np.sum(distances < 1000):2d}/{n} ({100*np.sum(distances<1000)/n:5.1f}%)")
    print(f"Voos < 2000m : {np.sum(distances < 2000):2d}/{n} ({100*np.sum(distances<2000)/n:5.1f}%)")
    print()
    print("DIAGNÓSTICO:")
    if np.median(distances) < 100:
        print("  ✓ Modelo v2 converge muito bem em JSBSim puro.")
        print("  → Pronto pra validar no FG (modo 7 do main.py).")
    elif np.median(distances) < 500:
        print("  ✓ Modelo v2 converge razoavelmente em JSBSim puro.")
        print("  → Vai pro FG, mas espere distâncias maiores no FG por causa")
        print("    do gap residual (timestep, integração).")
    else:
        print("  ✗ Modelo v2 não convergiu bem nem em JSBSim puro.")
        print("  → Algo deu errado no treino — talvez precise mais timesteps,")
        print("    ou os hyperparams precisem ajuste para o GR=1.88.")


if __name__ == "__main__":
    main()