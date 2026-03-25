import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim
import math
import csv
import os
from datetime import datetime  # Para gerar o timestamp das pastas

# Definição de caminhos do sistema para o motor JSBSim
JSBSIM_ROOT = r"D:\workspace\Pycharm\paraglider-autopilot\jsbsim"
AIRCRAFT_PATH = r"C:\Users\coppi\OneDrive\Documents\FlightGear\Aircraft"


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

        # --- NOVA LÓGICA DE ORGANIZAÇÃO DE ARQUIVOS ---
        # Gera um timestamp único para esta sessão de treino (Ex: 20231027_153045)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # AJUSTE DE CAMINHO: Agora utilizando o caminho absoluto solicitado dentro da pasta src
        base_records_path = r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records"
        self.run_dir = os.path.join(base_records_path, f"data_records_training_{timestamp}")

        # Cria as pastas necessárias (inclusive a flight_records dentro da src, se não existir)
        os.makedirs(self.run_dir, exist_ok=True)

        # O arquivo principal de log agora reside dentro da pasta da sessão
        self.log_file = os.path.join(self.run_dir, "flight_log.csv")

        # Ranking para salvar apenas os 10 melhores voos (menor distância final)
        self.top_flights = []  # Lista de tuplas: (distancia, numero_episodio, dados_completos)
        self.max_top_flights = 10

        # ESPAÇO DE AÇÃO: Comandos que a IA pode enviar
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)

        # ESPAÇO DE OBSERVAÇÃO: O que a IA "enxerga"
        self.observation_space = spaces.Box(
            low=np.array([0, -1, 0, -100, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 10, 100, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.fdm = None
        self._init_log_files()

    def _init_log_files(self):
        """Cria o cabeçalho do arquivo de log principal na pasta da sessão."""
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "flight_time_s", "final_dist", "reward", "lat", "lon"])

    def _get_bearing(self, lat1, lon1, lat2, lon2):
        """Calcula o ângulo (azimute) necessário para ir do ponto A ao ponto B."""
        off_x = lon2 - lon1
        off_y = lat2 - lat1
        bearing = 90.0 - math.degrees(math.atan2(off_y, off_x))
        return bearing % 360

    def _create_sim(self):
        """Inicializa uma nova instância do simulador JSBSim."""
        self.fdm = jsbsim.FGFDMExec(JSBSIM_ROOT)
        self.fdm.set_aircraft_path(AIRCRAFT_PATH)
        self.fdm.load_model("Parachutist")
        self.fdm.set_dt(1 / 120)

    def reset(self, seed=None, options=None):
        """Reinicia o ambiente para um novo voo."""
        super().reset(seed=seed)
        self.episode += 1
        self._create_sim()

        # Condições iniciais
        self.fdm["ic/lat-gc-deg"] = -26.288
        self.fdm["ic/long-gc-deg"] = -48.884
        self.fdm["ic/h-sl-ft"] = 9850
        self.fdm["ic/psi-true-deg"] = 0

        self.fdm["ic/u-fps"] = 35.0
        self.fdm["ic/w-fps"] = 15.0
        self.fdm.run_ic()

        self.fdm["inertia/pointmass-weight-lbs[0]"] = 211.3
        self.fdm["systems/chute/chute-cmd-norm"] = 1.0

        # Estabilização física de 4 segundos antes do controle da IA
        for _ in range(480): self.fdm.run()

        self.last_dist = haversine(self.fdm["position/lat-gc-deg"], self.fdm["position/long-gc-deg"],
                                   self.target_lat, self.target_lon)
        self.flight_time = 0
        self.total_reward = 0.0

        # Armazena temporariamente os dados deste voo para decidir se salvará no Top 10 depois
        self.current_flight_telemetry = []

        return self._get_obs(), {}

    def _get_obs(self):
        """Coleta e normaliza os dados para a rede neural."""
        lat = self.fdm["position/lat-gc-deg"]
        lon = self.fdm["position/long-gc-deg"]
        heading = self.fdm["attitude/psi-deg"]
        dist = haversine(lat, lon, self.target_lat, self.target_lon)
        target_bearing = self._get_bearing(lat, lon, self.target_lat, self.target_lon)
        bearing_error = (target_bearing - heading + 180) % 360 - 180

        return np.array([
            np.clip(dist / 5500.0, 0, 1),
            bearing_error / 180.0,
            self.fdm["velocities/vg-fps"] / 60.0,
            self.fdm["velocities/h-dot-fps"] / 30.0,
            self.fdm["attitude/roll-rad"],
            self.fdm["attitude/pitch-rad"]
        ], dtype=np.float32)

    def step(self, action):
        """Executa um passo de tempo (1 segundo) baseado na ação da IA."""
        self.fdm["fcs/aileron-cmd-norm"] = float(action[0])
        self.fdm["fcs/elevator-cmd-norm"] = float(action[1])

        for _ in range(120):
            if not self.fdm.run(): break

        self.flight_time += 1
        obs = self._get_obs()
        dist = obs[0] * 5500.0

        lat, lon = self.fdm["position/lat-gc-deg"], self.fdm["position/long-gc-deg"]
        alt = self.fdm["position/h-sl-ft"]
        vs = self.fdm["velocities/h-dot-fps"]
        heading = self.fdm["attitude/psi-deg"]

        # Armazena a telemetria deste segundo de voo
        self.current_flight_telemetry.append([
            self.flight_time, lat, lon, alt, heading, vs, dist, action[0], action[1]
        ])

        # Lógica de recompensa
        reward = (self.last_dist - dist) * 15.0
        if abs(obs[1]) < 0.1: reward += 2.0
        reward -= 0.1

        self.total_reward += float(reward)
        self.last_dist = dist

        done = bool(alt <= 10 or dist > 10000 or np.isnan(alt))

        if done:
            self._save_logs(dist, lat, lon)

        return obs, reward, done, False, {}

    def _save_logs(self, dist, lat, lon):
        """Salva o log geral e decide se este voo merece ser salvo no Top 10."""
        # Salva no arquivo de log geral que mostra todos os episódios
        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([self.episode, self.flight_time, dist, self.total_reward, lat, lon])

        # --- LÓGICA DE RANKING (TOP 10 MELHORES VOOS) ---
        if len(self.top_flights) < self.max_top_flights or dist < self.top_flights[-1][0]:
            self.top_flights.append((dist, self.episode, self.current_flight_telemetry))
            self.top_flights.sort(key=lambda x: x[0])

            if len(self.top_flights) > self.max_top_flights:
                removed_flight = self.top_flights.pop()
                file_to_delete = os.path.join(self.run_dir, f"flight_record_{removed_flight[1]}.csv")
                if os.path.exists(file_to_delete):
                    os.remove(file_to_delete)

            new_file = os.path.join(self.run_dir, f"flight_record_{self.episode}.csv")
            with open(new_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["time_s", "lat", "lon", "alt_ft", "heading", "vs_fps", "dist_m", "aileron", "elevator"])
                writer.writerows(self.current_flight_telemetry)

    def close(self):
        if self.fdm: self.fdm = None


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))