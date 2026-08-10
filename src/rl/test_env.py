import stable_baselines3 as sb3
from parachute_env import ParachuteEnv

TARGET_LAT = -26.2385
TARGET_LON = -48.884


def main():

    env = ParachuteEnv(TARGET_LAT, TARGET_LON)

    model = sb3.PPO.load("models/parachute_model")

    obs, _ = env.reset()

    done = False

    while not done:

        action, _ = model.predict(obs)

        obs, reward, done, truncated, info = env.step(action)

    print("Flight finished")

    env.close()


if __name__ == "__main__":
    main()