import airsim
import time
import matplotlib.pyplot as plt

# Настройки
TARGET_ALTITUDE = -35  # Целевая высота (отрицательная, так как AirSim использует z-координату, направленную вниз)
HOLD_TIME = 60  # Время удержания высоты (секунды)
MAX_SPEED = 2 # Максимальная скорость для перемещения в позицию

def altitude_hold_test(target_altitude, hold_time, max_speed):
    """
    Тестирует удержание заданной высоты, отображает график
    и выводит конечную высоту.
    """

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # Взлетаем на целевую высоту
    print("Взлетаем...")
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, target_altitude, max_speed)
    #time.sleep(2) # Даем время стабилизироваться

    # Отслеживаем высоту
    altitudes = []
    times = []
    start_time = time.time()

    while time.time() - start_time < hold_time:
        state = client.getMultirotorState()
        altitude = state.kinematics_estimated.position.z_val
        current_time = time.time() - start_time

        altitudes.append(altitude)
        times.append(current_time)

        print(f"Время: {current_time:.2f} с, Высота: {altitude:.2f} м")
        time.sleep(0.1)  # Небольшая задержка

    end_altitude = altitudes[-1]

    # График
    plt.figure(figsize=(10, 6))
    plt.plot(times, [-a for a in altitudes], label='Высота', color='blue')  # Инвертируем ось Y
    plt.xlabel('Время (с)')
    plt.ylabel('Высота (м)')
    plt.title('Удержание высоты')
    plt.axhline(y=-target_altitude, color='red', linestyle='--', label='Целевая высота')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Вывод
    print(f"\nКонечная высота: {end_altitude:.2f} м")

    # Завершение
    print("Приземляемся...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    altitude_hold_test(TARGET_ALTITUDE, HOLD_TIME, MAX_SPEED)