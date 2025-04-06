import airsim
import time
import matplotlib.pyplot as plt

# Настройки
TARGET_ALTITUDE = -35  # Целевая высота (отрицательная, так как AirSim использует z-координату, направленную вниз)
MAX_TAKEOFF_SPEED = 2  # Максимальная скорость подъема (м/с)
INITIAL_ALTITUDE = -2  # Начальная высота (чтобы не сталкиваться с землей)

def check_takeoff(target_altitude, max_takeoff_speed, initial_altitude):
    """
    Проверяет взлет дрона на заданную высоту, отображает графики скорости
    и выводит статистику.
    """

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # Взлетаем
    print("Взлетаем...")
    start_time_total = time.time() #Запомнили общее время старта

    # Поднимаемся на небольшую высоту, чтобы избежать столкновения с землей
    client.moveToZAsync(initial_altitude, MAX_TAKEOFF_SPEED).join()
    time.sleep(1)  # Даем время подняться

    # Теперь перемещаемся на целевую высоту
    client.moveToZAsync(target_altitude, MAX_TAKEOFF_SPEED).join()

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

        if altitude <= target_altitude + 0.5:  # Высота в AirSim уменьшается при подъеме, добавляем погрешность в 0.5 метра
            print("Целевая высота достигнута!")
            break

        time.sleep(0.1) # Небольшая задержка

    end_time = time.time()
    takeoff_time = end_time - start_time_total # Считаем полное время взлета
    max_altitude = min(altitudes) # Наименьшая z координата соответствует максимальной высоте
    average_speed = abs(max_altitude - initial_altitude) / takeoff_time # Рассчитываем среднюю скорость

    # Графики
    plt.figure(figsize=(12, 6))

    # График высоты
    plt.subplot(1, 2, 1)
    plt.plot(times, [-a for a in altitudes], label='Высота', color='blue')  # Инвертируем ось Y для более понятного отображения
    plt.xlabel('Время (с)')
    plt.ylabel('Высота (м)')
    plt.title('Изменение высоты во времени')
    plt.axhline(y=-target_altitude, color='red', linestyle='--', label='Целевая высота') #Отрицательная, тк AirSim
    plt.legend()
    plt.grid(True)

    # График вертикальной скорости
    plt.subplot(1, 2, 2)
    plt.plot(times, [-v for v in vertical_speeds], label='Вертикальная скорость', color='green')  # Инвертируем ось Y
    plt.xlabel('Время (с)')
    plt.ylabel('Вертикальная скорость (м/с)')
    plt.title('Изменение вертикальной скорости во времени')
    plt.axhline(y=max_takeoff_speed, color='red', linestyle='--', label='Макс. скорость') #  Отрицательная, тк AirSim
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Вывод статистики
    print("\n--- Статистика взлета ---")
    print(f"Фактическая набранная высота: {-max_altitude:.2f} м") #Инвертируем
    print(f"Время взлета: {takeoff_time:.2f} с")
    print(f"Средняя скорость подъема: {average_speed:.2f} м/с")

    # Завершение
    print("Приземляемся...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    check_takeoff(TARGET_ALTITUDE, MAX_TAKEOFF_SPEED, INITIAL_ALTITUDE)