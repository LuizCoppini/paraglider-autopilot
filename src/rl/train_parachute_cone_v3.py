"""
train_parachute_cone_v3.py — treino do método CONE v3.

Idêntico ao train_parachute_cone_v2.py em hyperparams, callbacks e
saída — só troca o env de ParachuteConeEnvV2 para ParachuteConeEnvV3.

v3 adiciona cobertura sim-to-real mínima sobre o v2:
  - Action lag (50-200 ms)
  - Observation noise (GPS ±3m, heading ±0.5°, alt ±2m)
  - Wind gusts (±20% mag, ±15° dir, cada 3-8 s)

NÃO faz Domain Randomization de GR (mantém fixo em 1.88 pra Parachutist),
porque a dissertação trata GR como propriedade da aeronave: cada parafoil
ganhará seu próprio modelo treinado especificamente.

Como rodar:
  Coloca em src/rl/train_parachute_cone_v3.py do seu projeto:
    poetry run python src/rl/train_parachute_cone_v3.py
"""

import os
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime

try:
    from rl.parachute_cone_env_v3 import ParachuteConeEnvV3
except ModuleNotFoundError:
    from parachute_cone_env_v3 import ParachuteConeEnvV3


# Target — Mojave (mesmo do v2, mesma comparabilidade no eval)
TARGET_LAT = 34.9055
TARGET_LON = -117.8830

# GR fixo do Parachutist (medido no JSBSim ~1.88)
PARACHUTIST_GR = 1.88

# Se True, treina com GR na obs (modelo único pra vários parafoils).
# Default False: modelo dedicado pra este parafoil (igual v2).
INCLUDE_GR_IN_OBS = False

# Mesma faixa de vento do v2 (todos fisicamente solúveis pelo parafoil)
WIND_SPEEDS_FPS = (4.0, 8.0, 14.0, 20.0)

# >>> Hiperparâmetros sim-to-real (cobre o gap até o ESP32 + parafoil real) <<<
# Action lag: 50-200 ms (= 6-24 sim_ticks a 120 Hz). Coberto:
#   - Leitura GPS: ~50 ms (NEO-6M a 5 Hz)
#   - ESP32 PPO inference: ~20-50 ms
#   - PWM servo SG90 resposta: ~50-100 ms
ACTION_LAG_TICKS_RANGE = (6, 24)

# Sensor noise — defaults conservadores pros sensores do projeto
GPS_NOISE_M = 3.0          # NEO-6M com SBAS = ~2-5m
HEADING_NOISE_DEG = 0.5    # MPU6050 fusion = ~0.3-1° drift
ALT_NOISE_M = 2.0          # BMP180 = ~±8m absoluto, ±2m relativo

# Wind gusts — turbulência leve a moderada
WIND_GUST_MAG_PCT = 0.20
WIND_GUST_DIR_DEG = 15.0
WIND_GUST_PERIOD_S = 5.0

# Total de timesteps (overnight)
TOTAL_TIMESTEPS = 2_500_000


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_models_path = r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_v3_method"
    session_dir = os.path.join(base_models_path, f"training_{timestamp}")
    checkpoint_dir = os.path.join(session_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    raw_env = ParachuteConeEnvV3(
        TARGET_LAT, TARGET_LON,
        glide_ratio_target=PARACHUTIST_GR,
        include_gr_in_obs=INCLUDE_GR_IN_OBS,
        wind_speeds_fps=WIND_SPEEDS_FPS,
        action_lag_ticks_range=ACTION_LAG_TICKS_RANGE,
        gps_noise_m=GPS_NOISE_M,
        heading_noise_deg=HEADING_NOISE_DEG,
        alt_noise_m=ALT_NOISE_M,
        wind_gust_mag_pct=WIND_GUST_MAG_PCT,
        wind_gust_dir_deg=WIND_GUST_DIR_DEG,
        wind_gust_period_s=WIND_GUST_PERIOD_S,
    )
    env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_dir,
        name_prefix='parachute_cone_v3_model',
    )

    model = sb3.PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tensorboard/cone_v3/",
    )

    print("=" * 72)
    print("--- Treino CONE v3 (cone v2 + cobertura sim-to-real mínima) ---")
    print("=" * 72)
    print(f"  método: cone v2 (mantido)")
    print(f"  glide_ratio_target = {PARACHUTIST_GR} (GR fixo do Parachutist)")
    print(f"  GR na obs = {INCLUDE_GR_IN_OBS}")
    print(f"  obs dim = {raw_env.observation_space.shape}")
    print(f"  target = ({TARGET_LAT}, {TARGET_LON})")
    print(f"  spawn radius = {raw_env.spawn_radius_m} m, alt = {raw_env.start_alt_ft} ft")
    print(f"  ventos base = {raw_env.wind_speeds} fps")
    print(f"  ── novidades v3 ──")
    print(f"  action lag  = {ACTION_LAG_TICKS_RANGE} ticks ({ACTION_LAG_TICKS_RANGE[0]*1000//120}-{ACTION_LAG_TICKS_RANGE[1]*1000//120} ms)")
    print(f"  GPS noise   = ±{GPS_NOISE_M:.1f} m (1σ)")
    print(f"  heading nz  = ±{HEADING_NOISE_DEG:.2f}° (1σ)")
    print(f"  alt noise   = ±{ALT_NOISE_M:.1f} m (1σ)")
    print(f"  wind gusts  = ±{WIND_GUST_MAG_PCT*100:.0f}% mag, ±{WIND_GUST_DIR_DEG:.0f}° dir, cada ~{WIND_GUST_PERIOD_S:.0f}s")
    print(f"  total timesteps = {TOTAL_TIMESTEPS:,}")
    print(f"  saída em = {session_dir}")
    print("=" * 72)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário. Salvando progresso atual...")

    model.save(os.path.join(session_dir, "parachute_cone_v3_model_final"))
    env.save(os.path.join(session_dir, "vec_normalize_cone_v3.pkl"))
    print(f"✅ Treino CONE v3 concluído. Arquivos em: {session_dir}")
    env.close()


if __name__ == "__main__":
    main()