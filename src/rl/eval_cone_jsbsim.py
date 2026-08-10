"""
eval_cone_jsbsim.py — Avalia o cone model em JSBSim PURO (sem FlightGear)
com o alvo deslocado para Mojave/Edwards AFB.

CORRIGIDO: lê o flight_log.csv que o env escreve em vez de ler o estado
do raw_env após o while (que já foi auto-resetado pelo DummyVecEnv).

Como rodar:
  Coloque este arquivo em src/rl/eval_cone_jsbsim.py do seu projeto.
    .venv\\Scripts\\python.exe src\\rl\\eval_cone_jsbsim.py
"""

import os
import sys
import csv
import math
import numpy as np

# Garante que o src do projeto está no path quando rodado como script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.parachute_cone_env import ParachuteConeEnv, haversine

# --- CONFIGURAÇÕES ---
MODEL_PATH = r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_method\training_20260406_200222\parachute_cone_model_final.zip"
VEC_NORMALIZE_PATH = r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_method\training_20260406_200222\vec_normalize_cone.pkl"

# Alvo de Mojave (mesmo que o validador FG usa)
TARGET_LAT = 34.9055
TARGET_LON = -117.8830


def main():
    print("=" * 70)
    print("Avaliação cone model em JSBSim PURO (sem FlightGear)")
    print("=" * 70)
    print(f"Target:  lat={TARGET_LAT}, lon={TARGET_LON} (Mojave)")
    print(f"Treino:  lat=-26.2385, lon=-48.884 (Itajaí/SC)")
    print(f"Modelo:  {MODEL_PATH}")
    print()

    # 1. Carrega modelo
    model = PPO.load(MODEL_PATH)
    print("[1/3] Modelo PPO carregado.")

    # 2. Cria env com target Mojave + VecNormalize
    raw_env = ParachuteConeEnv(TARGET_LAT, TARGET_LON)
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
        # Força (pos_id=p, wind_type_id=w) na próxima reset
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

    # 4. Lê o flight_log.csv que o env escreveu (uma linha por episódio)
    print()
    print(f"Lendo {raw_env.log_file}")
    rows = []
    with open(raw_env.log_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("ERRO: flight_log.csv vazio. Não dá pra avaliar.")
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
        print(f" {ep:2d} |  {pos_id}  | {ws:4.0f} @ {wd:5.0f}° | {ft:6.0f} | {d:13.2f} | {rew:13.0f} | {lat:10.5f} | {lon:10.5f}")

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
    if np.median(distances) < 200:
        print("  ✓ Modelo converge em JSBSim puro com Mojave.")
        print("  → O gap no FG é específico (timestep/wind/integração).")
        print("  → Caminho prático: usar modelo antigo no FG, ou retreinar com FG no loop.")
    elif np.median(distances) < 1000:
        print("  ~ Convergência parcial. Modelo funciona mas com erro maior do")
        print("    que esperado.")
    else:
        print("  ✗ Modelo NÃO converge nem em JSBSim puro com Mojave.")
        print("  → O modelo provavelmente overfittou no lat/lon do treino")
        print("    (-26.23, -48.884), ou no padrão de magnetic variation,")
        print("    ou em algum aspecto da geometria local.")
        print("  → Precisa retreinar com targets variados ou normalizar a obs")
        print("    para ser invariante à posição absoluta.")


if __name__ == "__main__":
    main()