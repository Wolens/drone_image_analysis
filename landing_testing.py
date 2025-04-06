import airsim
import time
import matplotlib.pyplot as plt

# Настройки
LANDING_ALTITUDE = -0.5  # Высота, на которой считаем, что приземление завершено (почти земля)
MAX_LANDING_SPEED = 1  # Максимальная скорость снижения (м/с)
INITIAL_ALTITUDE = -35 # Начальная высота перед приземлением

def check_landing(landing_altitude, max_landing_speed, initial_altitude):
    """
    Проверяет приземление дрона, отображает графики скорости и выводит статистику.
    """

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # Поднимаемся на начальную высоту
    print("Поднимаемся на начальную высоту...")
    client.moveToZAsync(initial_altitude, 2).join() # Поднимаемся быстро
    time.sleep(2)

    # Начинаем приземление
    print("Начинаем приземление...")
    start_time_total = time.time() # Запомнили общее время старта

    # Отслеживаем высоту и скорость
    altitudes = []
    vertical_speeds = []
    times = []

    start_time = time.time()  # Запомнили время начала отслеживания
    while True:
        state = client.getMultirotorState()
        altitude = state.kinematics_estimated.position.z_val
        vertical_speed = state.kinematics_estimated.linear_velocity.z_val
        current_time = time.time() - start_time_total  # Считаем время относительно самого начала

        altitudes.append(altitude)
        vertical_speeds.append(vertical_speed)
        times.append(current_time)

        print(f"Высота: {altitude:.2f} м, Вертикальная скорость: {vertical_speed:.2f} м/с")

        # Плавное приземление: устанавливаем скорость снижения
        client.moveByVelocityAsync(0, 0, max_landing_speed, 0.1).join() # Двигаемся вниз с заданной скоростью
        time.sleep(0.05) # Делаем паузы, чтобы дать дрону стабилизироваться

        if altitude >= landing_altitude:  # Высота в AirSim увеличивается при снижении
            print("Приземление завершено!")
            break

    end_time = time.time()
    landing_time = end_time - start_time_total  # Считаем полное время приземления
    max_altitude = max(altitudes) # Наибольшая z координата соответствует самой низкой точке
    average_speed = abs(initial_altitude - max_altitude) / landing_time  # Рассчитываем среднюю скорость

    # Графики
    plt.figure(figsize=(12, 6))

    # # График высоты
    # plt.subplot(1, 2, 1)
    # plt.plot(times, [-a for a in altitudes], label='Высота', color='blue')  # Инвертируем ось Y для более понятного отображения
    # plt.xlabel('Время (с)')
    # plt.ylabel('Высота (м)')
    # plt.title('Изменение высоты во времени')
    # plt.axhline(y=-landing_altitude, color='red', linestyle='--', label='Целевая высота')  # Отрицательная, тк AirSim
    # plt.legend()
    # plt.grid(True)

    # # График вертикальной скорости
    # plt.subplot(1, 2, 2)
    # plt.plot(times, [-v for v in vertical_speeds], label='Вертикальная скорость', color='green')  # Инвертируем ось Y
    # plt.xlabel('Время (с)')
    # plt.ylabel('Вертикальная скорость (м/с)')
    # plt.title('Изменение вертикальной скорости во времени')
    # plt.axhline(y=max_landing_speed, color='red', linestyle='--', label='Макс. скорость')  # Отрицательная, тк AirSim
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    # Вывод статистики
    print("\n--- Статистика приземления ---")
    print(f"Фактическая высота приземления: {-max_altitude:.2f} м")  # Инвертируем
    print(f"Время приземления: {landing_time:.2f} с")
    print(f"Средняя скорость снижения: {average_speed:.2f} м/с")

    # Завершение
    print("Выключаем двигатели...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    check_landing(LANDING_ALTITUDE, MAX_LANDING_SPEED, INITIAL_ALTITUDE)