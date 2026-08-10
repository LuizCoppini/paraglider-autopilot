"""
train_parachute_cone_v2.py — treino do método CONE com:
  - vento funcionando (bug do v1 corrigido)
  - glide_ratio_target ajustado pra GR REAL do Parachutist (~1.88)

O método continua sendo o do cone — agente aprende a tracking da
borda do cone que aponta para o alvo. Só que agora o cone tem
inclinação que corresponde à capacidade real do parafoil, não a
um valor arbitrário pequeno demais.

Como rodar:
  Coloca em src/rl/train_parachute_cone_v2.py do seu projeto:
    poetry run python src/rl/train_parachute_cone_v2.py
"""

import os
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime

try:
    from rl.parachute_cone_env_v2 import ParachuteConeEnvV2
except ModuleNotFoundError:
    from parachute_cone_env_v2 import ParachuteConeEnvV2


# Target — Mojave (mesmo da validação no FG)
TARGET_LAT = 34.9055
TARGET_LON = -117.8830

# GR REAL do Parachutist medido no JSBSim:
#   vg ~30 fps / v_down ~16 fps = 1.88
# Esse é o GR sem freio (envelope máximo). O cone passa a representar
# o limite físico real da aeronave.
PARACHUTIST_GR = 1.88

# Se True, treina com GR na obs — modelo único pra qualquer parafoil.
# Se False (default), treino dedicado para essa aeronave.
INCLUDE_GR_IN_OBS = False

# Faixa de ventos do treino. A airspeed do Parachutist é ~30 fps.
# Vento > 25 fps no rumo errado torna alguns cenários FISICAMENTE
# IMPOSSÍVEIS (ground_speed = 30 - 40 = -10 fps, parafoil não anda
# pra frente). Episódios impossíveis poluem o gradiente do PPO.
# Ficamos em [4, 8, 14, 20] fps — todos solúveis.
WIND_SPEEDS_FPS = (4.0, 8.0, 14.0, 20.0)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_models_path = r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_v2_method"
    session_dir = os.path.join(base_models_path, f"training_{timestamp}")
    checkpoint_dir = os.path.join(session_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    raw_env = ParachuteConeEnvV2(
        TARGET_LAT, TARGET_LON,
        glide_ratio_target=PARACHUTIST_GR,
        include_gr_in_obs=INCLUDE_GR_IN_OBS,
        wind_speeds_fps=WIND_SPEEDS_FPS,
    )
    env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_dir,
        name_prefix='parachute_cone_v2_model',
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
        tensorboard_log="./tensorboard/cone_v2/",
    )

    print("--- Treino CONE v2 ---")
    print(f"  método: cone (mantido)")
    print(f"  glide_ratio_target = {PARACHUTIST_GR} (GR real do Parachutist)")
    print(f"  GR na obs = {INCLUDE_GR_IN_OBS}")
    print(f"  obs dim = {raw_env.observation_space.shape}")
    print(f"  target = ({TARGET_LAT}, {TARGET_LON})")
    print(f"  spawn radius = {raw_env.spawn_radius_m} m, alt = {raw_env.start_alt_ft} ft")
    print(f"  ventos = {raw_env.wind_speeds} fps (todos solúveis pelo parafoil)")
    print(f"  saída em = {session_dir}")

    try:
        model.learn(
            total_timesteps=2_500_000,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário. Salvando progresso atual...")

    model.save(os.path.join(session_dir, "parachute_cone_v2_model_final"))
    env.save(os.path.join(session_dir, "vec_normalize_cone_v2.pkl"))
    print(f"✅ Treino CONE v2 concluído. Arquivos em: {session_dir}")
    env.close()


if __name__ == "__main__":
    main()
