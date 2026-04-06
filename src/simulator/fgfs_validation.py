import math
import time
import csv
import socket
import numpy as np
from pathlib import Path
from datetime import datetime
from flightgear_python.fg_if import FDMConnection
from stable_baselines3 import PPO

# --- CONFIGURAÇÕES ---
MODEL_PATH = r"D:\workspace\Pycharm\paraglider-autopilot\models\training_20260327_212226\parachute_model_final_4000_jumps.zip"
LOG_FOLDER = Path(r"D:\workspace\Pycharm\paraglider-autopilot\src\flight_records")
TARGET_LAT, TARGET_LON = -26.2385, -48.884

# Globais
csv_writer = None
csv_file_handle = None  # Referência para fechar o arquivo depois
telnet_socket = None
model = None
last_control_time = 0
chute_deployed = False
start_time = None
current_actions = [0.0, 0.0]  # [aileron, elevator]


def connect_telnet():
    global telnet_socket
    if telnet_socket: return
    try:
        telnet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        telnet_socket.connect(("localhost", 5401))
        print("✅ Telnet Connected")
    except:
        pass


def deploy_chute():
    global chute_deployed
    if chute_deployed: return
    try:
        telnet_socket.send("set /fdm/jsbsim/systems/chute/chute-cmd-norm 1\r\n".encode())
        chute_deployed = True
        print("🪂 PARACHUTE DEPLOYED!")
    except:
        pass


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_observation(fdm_data):
    lat = math.degrees(fdm_data.lat_rad)
    lon = math.degrees(fdm_data.lon_rad)
    dist = haversine(lat, lon, TARGET_LAT, TARGET_LON)
    v_ground = math.sqrt(fdm_data.v_north_ft_per_s ** 2 + fdm_data.v_east_ft_per_s ** 2)

    off_x, off_y = TARGET_LON - lon, TARGET_LAT - lat
    target_hdg = (90.0 - math.degrees(math.atan2(off_y, off_x))) % 360
    current_hdg = math.degrees(fdm_data.psi_rad)
    bearing_err = (target_hdg - current_hdg + 180) % 360 - 180
    v_vertical = -fdm_data.v_down_ft_per_s

    return np.array([
        np.clip(dist / 5500, 0, 1),
        bearing_err / 180,
        v_ground / 60,
        v_vertical / 30,
        fdm_data.phi_rad,
        fdm_data.theta_rad
    ], dtype=np.float32)


def fdm_callback(fdm_data, event_pipe):
    global csv_writer, csv_file_handle, model, last_control_time, start_time, chute_deployed, current_actions

    if model is None:
        model = PPO.load(MODEL_PATH)

    if start_time is None: start_time = time.time()
    connect_telnet()

    lat = math.degrees(fdm_data.lat_rad)
    lon = math.degrees(fdm_data.lon_rad)
    current_dist = haversine(lat, lon, TARGET_LAT, TARGET_LON)

    if time.time() - start_time > 4.0:
        deploy_chute()

    # Controle a 1Hz
    if time.time() - last_control_time >= 1.0 and chute_deployed:
        obs = get_observation(fdm_data)
        action, _ = model.predict(obs, deterministic=True)
        current_actions = [float(action[0]), float(action[1])]

        cmd = f"set /controls/flight/aileron {current_actions[0]}\r\nset /controls/flight/elevator {current_actions[1]}\r\n"
        try:
            telnet_socket.send(cmd.encode())
            last_control_time = time.time()
        except:
            pass

    # Registro de Log Expandido
    if csv_writer is None:
        LOG_FOLDER.mkdir(exist_ok=True, parents=True)
        filename = LOG_FOLDER / f"flight_records_fgfs_{datetime.now().strftime('%H%M%S')}.csv"
        csv_file_handle = open(filename, "w", newline="")
        csv_writer = csv.writer(csv_file_handle)
        csv_writer.writerow([
            "timestamp", "lat", "lon", "alt_ft", "dist_m",
            "v_down_fps", "heading_deg", "aileron", "elevator"
        ])

    csv_writer.writerow([
        round(time.time() - start_time, 2),
        round(lat, 6),
        round(lon, 6),
        round(fdm_data.alt_m * 3.28084, 2),
        round(current_dist, 2),
        round(fdm_data.v_down_ft_per_s, 2),
        round(math.degrees(fdm_data.psi_rad), 2),
        round(current_actions[0], 4),
        round(current_actions[1], 4)
    ])
    return fdm_data


def run_validation():
    print("🛰️ Starting FDM Receiver (Port 5501)...")
    conn = FDMConnection(fdm_version=24)
    conn.connect_rx("localhost", 5501, fdm_callback)
    conn.connect_tx("localhost", 5502)
    conn.start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        if csv_file_handle: csv_file_handle.close()
        print("Closing...")


if __name__ == "__main__":
    run_validation()