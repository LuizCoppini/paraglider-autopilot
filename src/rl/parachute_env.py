import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim
import math
import csv
import os

# Definição de caminhos do sistema para o motor JSBSim
JSBSIM_ROOT = r"D:\workspace\Pycharm\paraglider-autopilot\jsbsim"
AIRCRAFT_PATH = r"C:\Users\coppi\OneDrive\Documents\FlightGear\Aircraft"

# Arquivos de saída para análise de performance
LOG_FILE = "flight_log2.csv"
DEBUG_FILE = "first_flight_debug2.csv"


def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em metros entre dois pontos geográficos
    usando a fórmula de Haversine (considera a curvatura da Terra).
    """
    R = 6371000  # Raio da Terra em metros
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class ParachuteEnv(gym.Env):
    """
    Ambiente de Aprendizado por Reforço para controle de um paraquedas ram-air.
    Integra o motor de física JSBSim com a biblioteca Gymnasium.
    """

    def __init__(self, target_lat, target_lon):
        super().__init__()
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.episode = 0

        # ESPAÇO DE AÇÃO: Comandos que a IA pode enviar
        # [0] Aileron: -1 (curva total esquerda) a 1 (curva total direita)
        # [1] Elevator: 0 (sem freio) a 1 (freio máximo/estol parcial)
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)

        # ESPAÇO DE OBSERVAÇÃO: O que a IA "enxerga" (Normalizado para facilitar o treino)
        # [Distância, Erro de Proa, Velocidade Solo, Velocidade Vertical, Roll, Pitch]
        self.observation_space = spaces.Box(
            low=np.array([0, -1, 0, -100, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 10, 100, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.fdm = None
        self._init_log_files()

    def _init_log_files(self):
        """Cria o cabeçalho do arquivo CSV se ele ainda não existir."""
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "flight_time_s", "final_dist", "reward", "lat", "lon"])

    def _get_bearing(self, lat1, lon1, lat2, lon2):
        """Calcula o ângulo (azimute) necessário para ir do ponto A ao ponto B."""
        off_x = lon2 - lon1
        off_y = lat2 - lat1
        bearing = 90.0 - math.degrees(math.atan2(off_y, off_x))
        return bearing % 360

    def _create_sim(self):
        """Inicializa uma nova instância do simulador JSBSim com o modelo do paraquedista."""
        self.fdm = jsbsim.FGFDMExec(JSBSIM_ROOT)
        self.fdm.set_aircraft_path(AIRCRAFT_PATH)
        self.fdm.load_model("Parachutist")
        self.fdm.set_dt(1 / 120)  # Passo de tempo da física (120 Hz)

    def reset(self, seed=None, options=None):
        """Reinicia o ambiente para um novo voo (episódio)."""
        super().reset(seed=seed)
        self.episode += 1
        self._create_sim()

        # Configurações de posicionamento inicial (Latitude, Longitude, Altitude e Proa)
        self.fdm["ic/lat-gc-deg"] = -26.288
        self.fdm["ic/long-gc-deg"] = -48.884
        self.fdm["ic/h-sl-ft"] = 8000
        self.fdm["ic/psi-true-deg"] = 0

        # Define velocidades iniciais para evitar que a simulação quebre no início
        self.fdm["ic/u-fps"] = 35.0  # Velocidade frontal
        self.fdm["ic/w-fps"] = 15.0  # Velocidade de queda

        self.fdm.run_ic()  # Executa as condições iniciais

        # Define o peso total do paraquedista (essencial para o planeio correto)
        self.fdm["inertia/pointmass-weight-lbs[0]"] = 211.3

        # Comando para "abrir" o paraquedas (ativa os coeficientes aerodinâmicos do velame)
        self.fdm["systems/chute/chute-cmd-norm"] = 1.0

        # Ciclo de estabilização: Roda 4 segundos de física antes da IA começar a agir
        # Isso garante que o velame já esteja inflado e voando de forma estável.
        for _ in range(480): self.fdm.run()

        # Inicializa variáveis de controle de recompensa e log
        self.last_dist = haversine(self.fdm["position/lat-gc-deg"], self.fdm["position/long-gc-deg"],
                                   self.target_lat, self.target_lon)
        self.flight_time = 0
        self.total_reward = 0.0
        self.debug_data = []

        return self._get_obs(), {}

    def _get_obs(self):
        """Coleta e normaliza os dados do simulador para a rede neural."""
        lat = self.fdm["position/lat-gc-deg"]
        lon = self.fdm["position/long-gc-deg"]
        heading = self.fdm["attitude/psi-deg"]
        dist = haversine(lat, lon, self.target_lat, self.target_lon)

        target_bearing = self._get_bearing(lat, lon, self.target_lat, self.target_lon)
        # Calcula a diferença entre onde o paraquedas aponta e onde está o alvo (-180 a 180)
        bearing_error = (target_bearing - heading + 180) % 360 - 180

        return np.array([
            np.clip(dist / 5500.0, 0, 1),  # Distância (normalizada pelo máx de 5.5km)
            bearing_error / 180.0,  # Erro de direção (-1 a 1)
            self.fdm["velocities/vg-fps"] / 60.0,  # Velocidade de solo
            self.fdm["velocities/h-dot-fps"] / 30.0,  # Razão de subida/descida
            self.fdm["attitude/roll-rad"],  # Inclinação lateral
            self.fdm["attitude/pitch-rad"]  # Inclinação longitudinal
        ], dtype=np.float32)

    def step(self, action):
        """Executa um passo de tempo (1 segundo) baseado na ação da IA."""
        # Aplica os comandos da IA nos controles do simulador (definidos no Controls.xml)
        self.fdm["fcs/aileron-cmd-norm"] = float(action[0])
        self.fdm["fcs/elevator-cmd-norm"] = float(action[1])

        # Roda a física por 1 segundo (120 iterações de 1/120s)
        for _ in range(120):
            if not self.fdm.run(): break

        self.flight_time += 1
        obs = self._get_obs()
        dist = obs[0] * 5500.0  # Desnormaliza a distância para o cálculo de recompensa

        lat, lon = self.fdm["position/lat-gc-deg"], self.fdm["position/long-gc-deg"]
        alt = self.fdm["position/h-sl-ft"]
        vs = self.fdm["velocities/h-dot-fps"]

        # --- LÓGICA DE RECOMPENSA (Coração do Aprendizado) ---
        # 1. Recompensa por progresso: Ganha pontos se a distância diminuiu
        reward = (self.last_dist - dist) * 15.0

        # 2. Bônus por precisão: Ganha bônus se estiver apontando para o alvo
        if abs(obs[1]) < 0.1: reward += 2.0

        # 3. Penalidade por tempo: Estimula a IA a chegar mais rápido
        reward -= 0.1

        self.total_reward += float(reward)
        self.last_dist = dist

        # Condições de término do episódio
        # - Tocou o solo (alt <= 10)
        # - Afastou-se demais do alvo (dist > 10km)
        # - Erro matemático (NaN)
        done = bool(alt <= 10 or dist > 10000 or np.isnan(alt))

        # Coleta dados detalhados apenas do primeiro episódio para análise técnica
        if self.episode == 1:
            heading = self.fdm["attitude/psi-deg"]
            self.debug_data.append([self.flight_time, lat, lon, alt, heading, vs, dist, action[0], action[1]])

        if done:
            self._save_logs(dist, lat, lon)

        return obs, reward, done, False, {}

    def _save_logs(self, dist, lat, lon):
        """Salva os resultados finais do episódio no CSV."""
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([self.episode, self.flight_time, dist, self.total_reward, lat, lon])

        # Se for o episódio de debug, salva a telemetria segundo a segundo
        if self.episode == 1 and self.debug_data:
            with open(DEBUG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["time_s", "lat", "lon", "alt_ft", "heading", "vs_fps", "dist_m", "aileron", "elevator"])
                writer.writerows(self.debug_data)

    def close(self):
        """Finaliza o processo do simulador."""
        if self.fdm: self.fdm = None