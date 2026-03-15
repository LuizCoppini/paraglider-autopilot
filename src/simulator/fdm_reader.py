import math
import time
import csv
import socket
from pathlib import Path
from datetime import datetime
from flightgear_python.fg_if import FDMConnection
from config import settings

csv_writer = None
csv_file = None
parachute_deployed = False
telnet_socket = None


def connect_telnet():
    global telnet_socket

    if telnet_socket is not None:
        return

    try:
        telnet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        telnet_socket.connect(("localhost", settings.TELNET_PORT))
        print("Connected to FlightGear telnet")
    except Exception as e:
        print("Telnet connection failed:", e)


def deploy_parachute():
    global parachute_deployed

    if parachute_deployed:
        return

    try:
        command = "set /fdm/jsbsim/systems/chute/chute-cmd-norm 1\r\n"
        telnet_socket.send(command.encode())

        parachute_deployed = True

        print("=================================")
        print("PARACHUTE DEPLOYED")
        print("=================================")

    except Exception as e:
        print("Failed to deploy parachute:", e)


def ensure_csv():

    global csv_writer, csv_file

    if csv_writer is not None:
        return

    folder = Path("flight_records")
    folder.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = folder / f"flight_{timestamp}.csv"

    csv_file = open(filename, "w", newline="")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "timestamp",
        "lat",
        "lon",
        "alt_m",
        "heading_deg",
        "pitch_deg",
        "roll_deg",
        "v_north_ft_s",
        "v_east_ft_s",
        "v_down_ft_s"
    ])

    print(f"Flight log created: {filename}")


def fdm_callback(fdm_data, event_pipe):

    ensure_csv()
    connect_telnet()

    lat = math.degrees(fdm_data.lat_rad)
    lon = math.degrees(fdm_data.lon_rad)
    alt = fdm_data.alt_m
    roll = math.degrees(fdm_data.phi_rad)
    pitch = math.degrees(fdm_data.theta_rad)
    heading = math.degrees(fdm_data.psi_rad)

    v_north = fdm_data.v_north_ft_per_s
    v_east = fdm_data.v_east_ft_per_s
    v_down = fdm_data.v_down_ft_per_s

    timestamp = datetime.now().isoformat()

    csv_writer.writerow([
        timestamp,
        lat,
        lon,
        alt,
        heading,
        pitch,
        roll,
        v_north,
        v_east,
        v_down
    ])

    csv_file.flush()

    print(f"Lat: {lat:.6f} | Lon: {lon:.6f} | Alt: {alt:.1f}")

    # 🚀 Deploy parachute
    if alt <= settings.PARACHUTE_DEPLOY_ALTITUDE:
        deploy_parachute()

    return fdm_data


def start_reader():

    print("Waiting for FDM data...")

    conn = FDMConnection(fdm_version=24)

    conn.connect_rx(
        "localhost",
        5501,
        fdm_callback
    )

    conn.start()

    try:
        while True:
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Telemetry stopped")

        if csv_file:
            csv_file.close()