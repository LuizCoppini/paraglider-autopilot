# Paraglider Autopilot

Autonomous paraglider control system integrated with the FlightGear simulator for research, development, and testing of navigation, guidance, and control algorithms.

This project provides a framework for reading real-time Flight Dynamics Model (FDM) telemetry from FlightGear, recording flight data, and developing autonomous flight control strategies.

---

## Features

- Real-time telemetry from FlightGear FDM
- Flight data logging to CSV
- Autonomous flight experimentation environment
- Modular architecture for navigation and control algorithms
- Designed for future integration with embedded autopilot systems

---

## Project Architecture

paraglider-autopilot
│
├── src
│ ├── simulator
│ │ └── fdm_reader.py
│ │
│ ├── autopilot (future)
│ │ ├── navigation.py
│ │ ├── controller.py
│ │ └── state.py
│ │
│ └── main.py
│
├── flight_records
│
├── pyproject.toml
└── README.md

---

## Telemetry Data

The system reads FlightGear Flight Dynamics Model (FDM) packets and extracts key flight parameters:

- Latitude
- Longitude
- Altitude
- Heading
- Pitch
- Roll
- North/East/Down velocity

These values are logged in CSV format for analysis and replay.

Example log:
timestamp,lat,lon,alt_m,heading_deg,pitch_deg,roll_deg,v_north_ft_s,v_east_ft_s,v_down_ft_s
2026-03-11T14:50:12,-37.12345,-122.12345,3810,298.2,0.4,0.0,10.2,1.3,-0.4


---

## Requirements

- Python 3.12+
- Poetry
- FlightGear Simulator

Python dependencies are managed with **Poetry**.

---

## Installation

Clone the repository:

git clone https://github.com/LuizCoppini/paraglider-autopilot.git
cd paraglider-autopilot


Install dependencies:

poetry install


---

## Running the Telemetry Reader

Start FlightGear with FDM output enabled, then run:

poetry run python src/main.py


The system will:

1. Connect to the FlightGear FDM stream
2. Read real-time flight telemetry
3. Save flight data to the `flight_records` directory

---

## Roadmap

Future development will include:

- Waypoint navigation
- Autonomous landing algorithms
- PID-based control systems
- Glide path optimization
- Wind estimation
- Flight replay and analysis tools

---

## Applications

This project is intended for:

- UAV research
- Autonomous flight experimentation
- Flight control algorithm development
- Aerospace engineering education

---

## Author

**Luiz Fernando Coppini de Lima**

Software Engineer | Aerospace Engineering | Applied Artificial Intelligence

Experience in autonomous systems, embedded AI, and aerospace telemetry.

---

## License

This project is open-source and available under the MIT License.
