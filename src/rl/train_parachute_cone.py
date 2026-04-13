import os
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime
try:
    from rl.parachute_cone_env import ParachuteConeEnv
except ModuleNotFoundError:
    from parachute_cone_env import ParachuteConeEnv

TARGET_LAT = -26.2385
TARGET_LON = -48.884

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Pasta específica para diferenciar os modelos de cone dos modelos de distância
    base_models_path = r"D:\workspace\Pycharm\paraglider-autopilot\models\cone_method"
    session_dir = os.path.join(base_models_path, f"training_{timestamp}")
    checkpoint_dir = os.path.join(session_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Ambiente de Cone
    # O ParachuteConeEnv já possui a lógica de recompensa baseada no raio ideal (altitude * glide_ratio)
    raw_env = ParachuteConeEnv(TARGET_LAT, TARGET_LON)
    env = DummyVecEnv([lambda: raw_env])

    # Normalização: norm_reward=True pode ser útil aqui pois a recompensa do cone
    # é contínua e baseada em erro de distância, diferente do bônus esparso anterior.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_dir,
        name_prefix='parachute_cone_model'
    )

    # 3. Modelo PPO Ajustado para Seguimento de Trajetória (Cone)
    model = sb3.PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        ent_coef=0.01,
        gamma=0.99,  # Gamma ligeiramente menor (0.99) foca mais no erro do cone imediato
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tensorboard/cone_logic/"
    )

    print(f"--- Iniciando Treino com Lógica de Cone (Target Glide Ratio: 0.8) ---")

    try:
        # 4. Total de passos (Mantido conforme sua estratégia de 4000 saltos)
        model.learn(
            total_timesteps=2500000,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário. Salvando progresso atual...")

    # 5. Salvamento Final
    model_name = "parachute_cone_model_final"
    model.save(os.path.join(session_dir, model_name))
    env.save(os.path.join(session_dir, "vec_normalize_cone.pkl"))

    print(f"✅ Treino de Cone concluído! Arquivos em: {session_dir}")
    env.close()

if __name__ == "__main__":
    main()