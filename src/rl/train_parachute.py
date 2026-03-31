import os
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime
from parachute_env import ParachuteEnv

TARGET_LAT = -26.2385
TARGET_LON = -48.884


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_models_path = r"D:\workspace\Pycharm\paraglider-autopilot\models"
    session_dir = os.path.join(base_models_path, f"training_{timestamp}")
    checkpoint_dir = os.path.join(session_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Ambiente
    # Importante: Não use semente fixa aqui para permitir que as direções de vento
    # variem conforme o episódio no ambiente.
    raw_env = ParachuteEnv(TARGET_LAT, TARGET_LON)
    env = DummyVecEnv([lambda: raw_env])

    # Ajuste: norm_reward=False ajuda a IA a valorizar o bônus de pouso suave de 1000 pts
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # 2. Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Aumentei para salvar a cada ~150 voos
        save_path=checkpoint_dir,
        name_prefix='parachute_model'
    )

    # 3. Modelo PPO Ajustado para as Novas Regras
    model = sb3.PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,  # Maior janela de observação para trajetórias longas
        batch_size=128,  # Maior estabilidade no gradiente
        n_epochs=10,
        ent_coef=0.01,  # Exploração moderada (foco em refinamento)
        gamma=0.995,  # Maior foco em recompensas de longo prazo (o pouso final)
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tensorboard/"
    )

    print(f"--- Iniciando Treino Reprodutível (8 Pontos / 4 Ventos) ---")

    try:
        # 4. Total de passos para cobrir os 4000 saltos planejados
        model.learn(
            total_timesteps=2500000,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nInterrompido. Salvando...")

    # 5. Salvamento Final
    model.save(os.path.join(session_dir, "parachute_model_final_4000_jumps"))
    env.save(os.path.join(session_dir, "vec_normalize.pkl"))

    print(f"Treino concluído! Salvo em: {session_dir}")
    env.close()


if __name__ == "__main__":
    main()