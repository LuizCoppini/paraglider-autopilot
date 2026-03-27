import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim
import math
import csv
import os
import random
from datetime import datetime

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_records_path = r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records"
        self.run_dir = os.path.join(base_records_path, f"data_records_training_{timestamp}")

        os.makedirs(self.run_dir, exist_ok=True)
        self.log_file = os.path.join(self.run_dir, "flight_log.csv")

        self.top_flights = []
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
        """Reinicia o ambiente para um novo voo com localização aleatória."""
        super().reset(seed=seed)
        self.episode += 1
        self._create_sim()

        # --- LÓGICA DE LOCALIZAÇÃO ALEATÓRIA (Raio de 5km) ---
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, 5000)

        delta_lat = (radius * math.cos(angle)) / 111320.0
        delta_lon = (radius * math.sin(angle)) / (111320.0 * math.cos(math.radians(self.target_lat)))

        start_lat = self.target_lat + delta_lat
        start_lon = self.target_lon + delta_lon

        # Condições iniciais
        self.fdm["ic/lat-gc-deg"] = start_lat
        self.fdm["ic/long-gc-deg"] = start_lon
        self.fdm["ic/h-sl-ft"] = 9850
        self.fdm["ic/psi-true-deg"] = random.uniform(0, 360)

        # --- CONFIGURAÇÃO DE VENTO PADRÃO LEVE ---
        self.fdm["atmosphere/wind-north-fps"] = -8.4
        self.fdm["atmosphere/wind-east-fps"] = 0.0

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
        self.current_flight_telemetry = []

        return self._get_obs(), {}

    def _get_obs(self):
        """Coleta e normaliza os dados com proteção contra NaN."""
        try:
            lat = self.fdm["position/lat-gc-deg"]
            lon = self.fdm["position/long-gc-deg"]
            heading = self.fdm["attitude/psi-deg"]
            alt = self.fdm["position/h-sl-ft"]

            dist = haversine(lat, lon, self.target_lat, self.target_lon)
            target_bearing = self._get_bearing(lat, lon, self.target_lat, self.target_lon)
            bearing_error = (target_bearing - heading + 180) % 360 - 180

            # Normalização com clip para evitar explosão de gradiente
            obs = np.array([
                np.clip(dist / 5500.0, 0, 1),
                bearing_error / 180.0,
                np.clip(self.fdm["velocities/vg-fps"] / 60.0, -1, 2),
                np.clip(self.fdm["velocities/h-dot-fps"] / 30.0, -2, 2),
                np.clip(self.fdm["attitude/roll-rad"], -1, 1),
                np.clip(self.fdm["attitude/pitch-rad"], -1, 1)
            ], dtype=np.float32)

            if np.isnan(obs).any():
                return np.zeros(6, dtype=np.float32)
            return obs
        except:
            return np.zeros(6, dtype=np.float32)

    def step(self, action):
        """Executa um passo de tempo com blindagem contra erros físicos."""
        # Clipagem das ações para evitar valores inválidos
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.fdm["fcs/aileron-cmd-norm"] = float(action[0])
        self.fdm["fcs/elevator-cmd-norm"] = float(action[1])

        # Execução da física com verificação de sucesso
        success = True
        for _ in range(120):
            if not self.fdm.run():
                success = False
                break

        alt = self.fdm["position/h-sl-ft"]

        # Se a física explodir (NaN ou falha no run), encerra o episódio com penalidade
        if not success or np.isnan(alt):
            return np.zeros(6, dtype=np.float32), -100.0, True, False, {}

        self.flight_time += 1
        obs = self._get_obs()
        dist = obs[0] * 5500.0

        lat, lon = self.fdm["position/lat-gc-deg"], self.fdm["position/long-gc-deg"]
        vs = self.fdm["velocities/h-dot-fps"]
        heading = self.fdm["attitude/psi-deg"]

        self.current_flight_telemetry.append([
            self.flight_time, lat, lon, alt, heading, vs, dist, action[0], action[1]
        ])

        # Lógica de recompensa
        reward = (self.last_dist - dist) * 15.0
        if abs(obs[1]) < 0.1: reward += 2.0
        reward -= 0.1

        # Condição de término com limite de tempo (1500 segundos)
        done = bool(alt <= 10 or dist > 10000 or np.isnan(alt) or self.flight_time > 1500)

        # Lógica de Flare (Pouso Suave)
        if done and alt <= 15:
            if dist < 50:
                landing_softness = vs
                if landing_softness < -5.0:
                    reward -= abs(landing_softness) * 10.0
                else:
                    reward += 500.0

        self.total_reward += float(reward)
        self.last_dist = dist

        if done:
            self._save_logs(dist, lat, lon)

        return obs, reward, done, False, {}

    def _save_logs(self, dist, lat, lon):
        """Salva o log geral e o Top 10."""
        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([self.episode, self.flight_time, dist, self.total_reward, lat, lon])

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