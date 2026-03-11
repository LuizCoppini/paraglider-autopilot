import math
import time
import csv
from pathlib import Path
from datetime import datetime
from flightgear_python.fg_if import FDMConnection

csv_writer = None
csv_file = None


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
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Telemetry stopped")