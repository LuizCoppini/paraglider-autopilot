#import time

#from simulator.launcher import FlightGearLauncher
#from simulator.fdm_reader import start_reader


#def main():

    #fg = FlightGearLauncher()
    #fg.start()

    #print("Waiting FlightGear start...")

    #time.sleep(10)

    #print("Starting telemetry reader...")

    #start_reader()


#if __name__ == "__main__":
 #   main()

from simulator.jsbsim_runner import run_simulation

run_simulation()