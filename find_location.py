import cv2
import numpy as np
import os

def find_location(current_image_path, photo_data, output_dir="matches"):
    """
    Сравнивает текущее изображение с изображениями из photo_data и возвращает наиболее вероятное местоположение.
    Также возвращает количество совпадающих ключевых точек.
    Использует порог расстояния для отбора good_matches.
    """
    # Создаем директорию для отладочных изображений
    os.makedirs(output_dir, exist_ok=True)

    current_img = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
    if current_img is None:
        print(f"Error: Could not read image at {current_image_path}")
        return None, 0

    best_match = None
    best_score = 0
    best_match_count = 0

    # Инициализация ORB детектора
    orb = cv2.ORB_create(nfeatures=2000) #Увеличение количества фич
    #orb = cv2.ORB_create()

    # Находим ключевые точки и дескрипторы для текущего изображения
    keypoints_current, descriptors_current = orb.detectAndCompute(current_img, None)
    if descriptors_current is None or len(keypoints_current) < 10:
        print(f"Warning: Could not find enough keypoints in {current_image_path}")
        return None, 0

    for data in photo_data:
        map_img = cv2.imread(data['filename'], cv2.IMREAD_GRAYSCALE)
        if map_img is None:
            print(f"Error: Could not read image at {data['filename']}")
            continue

        # Находим ключевые точки и дескрипторы для изображения из карты
        keypoints_map, descriptors_map = orb.detectAndCompute(map_img, None)
        if descriptors_map is None or len(keypoints_map) < 10:
            continue

        # Создаем объект BFMatcher (Brute-Force Matcher)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Сопоставляем дескрипторы
        try:
            matches = bf.match(descriptors_current, descriptors_map)
        except cv2.error as e:
            print(f"Error during matching: {e}")
            continue

        # Порог расстояния (экспериментируйте со значением)
        distance_threshold = 50  # Примерное значение.  Подберите подходящее для ваших изображений

        # Отбираем только хорошие совпадения по порогу расстояния
        good_matches = [m for m in matches if m.distance < distance_threshold]

        # Оценка качества сопоставления: количество хороших совпадений
        score = len(good_matches)

        if score > best_score:
            best_score = score
            best_match = data
            best_match_count = len(good_matches)

    if best_match:
        print(f"Best match found: {best_match['filename']} with score {best_score}")
        print(f"Number of good matches: {best_match_count}")

        # Рисуем совпадения (опционально, для визуализации)
        map_img = cv2.imread(best_match['filename'])
        img_matches = cv2.drawMatches(current_img, keypoints_current, map_img, keypoints_map, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.path.join(output_dir, f"matches_{os.path.basename(current_image_path)}_{os.path.basename(best_match['filename'])}.jpg"), img_matches)

        return best_match, best_match_count
    else:
        print("No match found.")
        return None, 0