import os
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime
from parachute_env import ParachuteEnv

# Coordenadas do Alvo
TARGET_LAT = -26.2385
TARGET_LON = -48.884


def main():
    # --- LÓGICA DE ORGANIZAÇÃO DE PASTAS ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_models_path = r"D:\workspace\Pycharm\paraglider-autopilot\models"
    # Pasta específica desta sessão: models/training_20260325_140000
    session_dir = os.path.join(base_models_path, f"training_{timestamp}")

    # Criar a pasta da sessão e a subpasta de checkpoints
    checkpoint_dir = os.path.join(session_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Criação do Ambiente
    raw_env = ParachuteEnv(TARGET_LAT, TARGET_LON)
    env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Configuração de Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix='parachute_model'
    )

    # 3. Definição do Modelo PPO
    model = sb3.PPO(
        "MlpPolicy",  # Rede neural padrão (camadas densas para vetores de dados)
        env,  # O ambiente ParachuteEnv
        verbose=1,  # Nível de log (1 mostra as estatísticas básicas no console)
        learning_rate=3e-4,  # Quão rápido a rede neural tenta aprender (0.0003)
        n_steps=2048,  # Tamanho da "memória" de curto prazo antes de atualizar
        batch_size=64,  # Quantos dados ele processa por vez dentro da atualização
        n_epochs=10,  # Quantas vezes ele revisita os mesmos dados em um treino
        ent_coef=0.05,  # Incentiva a exploração (IA tenta manobras novas)
        gamma=0.99,  # Dá importância a recompensas futuras (visão de longo prazo)
        gae_lambda=0.95,  # Suaviza a estimativa de vantagem (estabilidade matemática)
        clip_range=0.2,  # Limita mudanças bruscas na política (evita colapsos de treino)
        tensorboard_log="./tensorboard/"  # Pasta para visualizar gráficos de performance
    )

    print(f"--- Iniciando Treinamento (Pasta: {session_dir}) ---")

    # 4. Execução do Treino
    try:
        model.learn(
            total_timesteps=1000000,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTreino interrompido manualmente. Salvando estado atual...")

    # 5. Salvamento Final (Tudo centralizado na session_dir)
    model_final_path = os.path.join(session_dir, "parachute_model_final_1M")
    model.save(model_final_path)

    # Salva as estatísticas de normalização no mesmo local
    stats_path = os.path.join(session_dir, "vec_normalize.pkl")
    env.save(stats_path)

    print(f"Treino concluído! Arquivos salvos em: {session_dir}")
    env.close()


if __name__ == "__main__":
    main()