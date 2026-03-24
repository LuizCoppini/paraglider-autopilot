import os
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from parachute_env import ParachuteEnv

# Coordenadas do Alvo (Joinville/SC por exemplo)
TARGET_LAT = -26.2385
TARGET_LON = -48.884


def main():
    # 1. Criação do Ambiente
    # Usamos DummyVecEnv + VecNormalize para ajudar a IA a lidar com números grandes (distância)
    raw_env = ParachuteEnv(TARGET_LAT, TARGET_LON)
    env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Configuração de Checkpoints (Salva o progresso automaticamente)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./models/checkpoints/',
        name_prefix='parachute_model'
    )

    # 3. Definição do Modelo PPO
    model = sb3.PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,  # Valor padrão mais estável para navegação
        n_steps=2048,  # Janela de experiência antes de atualizar a rede
        batch_size=64,
        n_epochs=10,  # Quantas vezes ele revisita a mesma experiência
        ent_coef=0.05,  # Coeficiente de entropia (0.05 a 0.1 força exploração)
        gamma=0.99,  # Fator de desconto para recompensas futuras
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tensorboard/"
    )

    print("--- Iniciando Treinamento de Longo Prazo (1M Steps) ---")

    # 4. Execução do Treino
    try:
        model.learn(
            total_timesteps=1000000,
            callback=checkpoint_callback,
            progress_bar=True  # Barra visual de progresso (SB3 mais recente)
        )
    except KeyboardInterrupt:
        print("Treino interrompido manualmente. Salvando estado atual...")

    # 5. Salvamento Final
    os.makedirs("models", exist_ok=True)
    model.save("models/parachute_model_final_1M")

    # Salva as estatísticas de normalização (necessário para rodar o modelo depois)
    env.save("models/vec_normalize.pkl")

    print("Treino concluído!")
    env.close()


if __name__ == "__main__":
    main()