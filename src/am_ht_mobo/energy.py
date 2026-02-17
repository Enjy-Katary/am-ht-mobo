import numpy as np

time_hours = np.array([0.017, 0.025, 0.033, 0.0417, 0.05, 0.0583, 0.0667, 0.075,
                       0.083, 0.0917, 0.1, 0.108, 0.117, 0.125, 0.133, 0.142,
                       0.15, 0.158, 0.167, 0.175, 0.183, 0.191, 0.2, 0.208,
                       0.217, 0.225, 0.233, 0.242, 0.25, 0.258, 0.267, 0.275,
                       0.283, 0.292, 0.3, 0.308, 0.317, 0.325, 0.333, 0.341, 0.35])
temperature_ramp = np.array([63, 67, 70, 84, 95, 99, 103, 124, 136, 150, 170, 211, 272,
                             329, 370, 411, 445, 483, 521, 594, 651, 719, 790, 850,
                             893, 930, 961, 987, 1011, 1033, 1053, 1073, 1091, 1107,
                             1123, 1138, 1152, 1166, 1180, 1192, 1200])

coeffs_ramp = np.polyfit(temperature_ramp, time_hours, 6)


def ramp_time(T):

    return np.polyval(coeffs_ramp, T)


temperature_duty = np.array([570, 650, 800, 850, 1000, 1150, 1200])
duty_cycle_values = np.array([3.41, 5.48, 8.54, 14.23, 23.43, 45.39, 70.67])
coeffs_duty = np.polyfit(temperature_duty, duty_cycle_values, 3)


def duty_cycle_func(T):

    return np.polyval(coeffs_duty, T)

def energy_consumption(T1, T2, t1_min, t2_min):

    t_ramp1 = ramp_time(T1)
    t_ramp2 = ramp_time(T2)

    D1 = duty_cycle_func(T1) / 100.0
    D2 = duty_cycle_func(T2) / 100.0

    t1_hours = t1_min / 60.0
    t2_hours = t2_min / 60.0

    E_total = 2.6 * (t_ramp1 + D1 * t1_hours + t_ramp2 + D2 * t2_hours)

    total_time = t_ramp1 + t1_hours + t_ramp2 + t2_hours

    energy_per_hour = E_total / total_time

    return round(E_total, 2), round(energy_per_hour, 2)


# --------------------------
# Main Routine
# --------------------------
if __name__ == '__main__':
    print("Heat Treatment Energy Consumption Calculator")
    try:
        # Prompt the user for temperatures (°C) and hold times (minutes)
        T1 = float(input("Enter temperature T1 (°C) for stage 1: "))
        T2 = float(input("Enter temperature T2 (°C) for stage 2: "))
        t1 = float(input("Enter hold time t1 (minutes) for stage 1: "))
        t2 = float(input("Enter hold time t2 (minutes) for stage 2: "))
    except ValueError:
        print("Please enter valid numerical values.")
        exit(1)

    # Calculate the energy consumption and average energy per hour
    E_total, energy_per_hour = energy_consumption(T1, T2, t1, t2)

    print("\n------ Results ------")
    print(f"Total energy consumption for the process: {E_total:.3f} kWh")
    print(f"Average energy consumption per hour: {energy_per_hour:.3f} kW")