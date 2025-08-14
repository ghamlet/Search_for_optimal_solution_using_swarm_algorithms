import time
import threading
from typing import List, Tuple
import cv2
import numpy as np

from pion import Pion
from flight_utils import load_flight_coordinates
from camera_sim import CameraSim
from object_detection_controller_sim import ObjectDetectionControllerSim
import math
from camera_real import CameraReal



def display_loop(detector: ObjectDetectionControllerSim, stop_event: threading.Event, detection_event: threading.Event):
    """Отображает последний кадр из контроллера. Ставит detection_event при подтвержденной детекции."""
    cv2.namedWindow('SIM Detection', cv2.WINDOW_NORMAL)
    while not stop_event.is_set():
        detector.show_frame('SIM Detection')
        if detector.is_confirmed():
            detection_event.set()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
        time.sleep(0.01)


class MissionControllerSim:
    def __init__(self, drone: Pion):
        self.drone = drone
        self.mission_points: List[Tuple[float, float, float]] = []
        self.current_point_index = 0
        self.last_detection_time = 0  # Время последней детекции
        self.detection_cooldown = 30  # Время блокировки детекции в секундах (увеличено)
        self.cows_approached = {}  # Словарь коров: {cow_key: (coords, timestamp)}
        self.approach_radius = 2.0  # Радиус в метрах для определения "той же коровы"
        self.detector = None  # Ссылка на детектор для получения координат
        # Новые поля для отслеживания обработанных коров
        self.processed_cows = {}  # {cow_key: (coords, status, timestamp)} - status: 'herded' или 'skipped'
        self.processed_radius = 3.0  # Радиус для определения уже обработанной коровы

    def load_mission(self, points: List[Tuple[float, float, float]]):
        self.mission_points = points

    def set_detector(self, detector):
        """Устанавливает ссылку на детектор для получения координат коровы."""
        self.detector = detector
        # print(f"DEBUG: Детектор установлен: {detector}")
        # if detector is not None:
        #     print(f"DEBUG: Детектор имеет метод get_last_detection: {hasattr(detector, 'get_last_detection')}")
        #     print(f"DEBUG: Детектор имеет метод get_global_coordinates: {hasattr(detector, 'get_global_coordinates')}")

    def _get_current_cow_coordinates(self):
        """Получает текущие координаты коровы из детектора."""
        if COW_TEST_COORDS is not None:
            return COW_TEST_COORDS
        
        if self.detector is not None and hasattr(self.detector, 'get_last_detection'):
            detection = self.detector.get_last_detection()
            if detection and len(detection) >= 3 and detection[0]:
                center = detection[1]
                bbox = detection[2] if len(detection) > 2 else None
                
                if center is not None:
                    # Проверяем качество детекции
                    if bbox is not None:
                        x, y, w, h = bbox
                        area = w * h
                        # Фильтруем слишком маленькие или слишком большие объекты
                        if area < 1000 or area > 50000:  # Минимальная и максимальная площадь
                            print(f"Детекция отфильтрована по размеру: area={area}")
                            return None
                    
                    coords = self.detector.get_global_coordinates(center)
                    print(f"Получены координаты коровы: {coords}")
                    return coords
        
        return None

    def _is_cow_already_approached(self, cow_coords):
        """Проверяет, была ли корова уже обработана."""
        if not cow_coords:
            return False
        
        current_time = time.time()
        cow_x, cow_y = cow_coords
        
        # Проверяем всех обработанных коров
        for cow_key, (stored_coords, timestamp) in self.cows_approached.items():
            stored_x, stored_y = stored_coords
            
            # Вычисляем расстояние между коровами
            distance = ((cow_x - stored_x) ** 2 + (cow_y - stored_y) ** 2) ** 0.5
            
            # Если коровы близко друг к другу (в радиусе approach_radius)
            if distance <= self.approach_radius:
                # Проверяем, не устарели ли данные (больше 5 минут)
                if current_time - timestamp < 300:  # 5 минут
                    print(f"Корова уже была обработана: текущая=({cow_x:.2f}, {cow_y:.2f}), "
                          f"сохраненная=({stored_x:.2f}, {stored_y:.2f}), расстояние={distance:.2f}м")
                    return True
                else:
                    # Удаляем устаревшие данные
                    del self.cows_approached[cow_key]
                    print(f"Удалены устаревшие данные коровы: {cow_key}")
        
        return False

    def _add_cow_to_approached(self, cow_coords):
        """Добавляет корову в список обработанных."""
        if not cow_coords:
            return
        
        current_time = time.time()
        cow_x, cow_y = cow_coords
        
        # Создаем ключ для коровы
        cow_key = f"cow_{len(self.cows_approached) + 1}"
        
        # Добавляем корову в словарь
        self.cows_approached[cow_key] = ((cow_x, cow_y), current_time)
        
        print(f"Добавлена корова в обработанные: {cow_key} = ({cow_x:.2f}, {cow_y:.2f})")
        print(f"Всего обработанных коров: {len(self.cows_approached)}")

    def _is_cow_already_processed(self, cow_coords):
        """Проверяет, была ли корова уже полностью обработана (загнана или пропущена)."""
        if not cow_coords:
            return False
        
        current_time = time.time()
        cow_x, cow_y = cow_coords
        
        # Проверяем всех полностью обработанных коров
        for cow_key, (stored_coords, status, timestamp) in self.processed_cows.items():
            stored_x, stored_y = stored_coords
            
            # Вычисляем расстояние между коровами
            distance = ((cow_x - stored_x) ** 2 + (cow_y - stored_y) ** 2) ** 0.5
            
            # Если коровы близко друг к другу (в радиусе processed_radius)
            if distance <= self.processed_radius:
                # Проверяем, не устарели ли данные (больше 10 минут)
                if current_time - timestamp < 600:  # 10 минут
                    print(f"Корова уже была полностью обработана ({status}): текущая=({cow_x:.2f}, {cow_y:.2f}), "
                          f"сохраненная=({stored_x:.2f}, {stored_y:.2f}), расстояние={distance:.2f}м")
                    return True
                else:
                    # Удаляем устаревшие данные
                    del self.processed_cows[cow_key]
                    print(f"Удалены устаревшие данные обработанной коровы: {cow_key}")
        
        return False

    def _add_cow_to_processed(self, cow_coords, status):
        """Добавляет корову в список полностью обработанных."""
        if not cow_coords:
            return
        
        current_time = time.time()
        cow_x, cow_y = cow_coords
        
        # Создаем ключ для коровы
        cow_key = f"processed_cow_{len(self.processed_cows) + 1}"
        
        # Добавляем корову в словарь
        self.processed_cows[cow_key] = ((cow_x, cow_y), status, current_time)
        
        print(f"Добавлена корова в полностью обработанные: {cow_key} = ({cow_x:.2f}, {cow_y:.2f}), статус: {status}")
        print(f"Всего полностью обработанных коров: {len(self.processed_cows)}")
        
        # Также удаляем из временного списка approached
        self._remove_cow_from_approached(cow_coords)

    def _remove_cow_from_approached(self, cow_coords):
        """Удаляет корову из временного списка approached."""
        if not cow_coords:
            return
        
        cow_x, cow_y = cow_coords
        
        # Ищем и удаляем корову из approached
        keys_to_remove = []
        for cow_key, (stored_coords, timestamp) in self.cows_approached.items():
            stored_x, stored_y = stored_coords
            distance = ((cow_x - stored_x) ** 2 + (cow_y - stored_y) ** 2) ** 0.5
            
            if distance <= self.approach_radius:
                keys_to_remove.append(cow_key)
        
        for key in keys_to_remove:
            del self.cows_approached[key]
            print(f"Удалена корова из временного списка approached: {key}")

    def execute_mission_non_blocking(self, stop_event: threading.Event, detection_event: threading.Event, stop_on_detection: bool):
        if not self.mission_points:
            print("Ошибка: маршрут не загружен!")
            return

        # Сохраняем ссылку на событие детекции для сброса
        self._detection_event = detection_event

        try:
            print(f"Начало миссии ({len(self.mission_points)} точек) [SIM]")
            
            while self.current_point_index < len(self.mission_points) and not stop_event.is_set():
                # Проверяем детекцию перед каждым перелетом с учетом времени блокировки
                current_time = time.time()
                if (stop_on_detection and detection_event.is_set() and 
                    current_time - self.last_detection_time > self.detection_cooldown):
                    
                    # Получаем координаты коровы для проверки, не та же ли это корова
                    cow_coords = self._get_current_cow_coordinates()
                    if cow_coords:
                        # Сначала проверяем, не была ли корова уже полностью обработана
                        if self._is_cow_already_processed(cow_coords):
                            print(f"Корова уже была полностью обработана, игнорируем: {cow_coords}")
                            # Сбрасываем событие детекции и продолжаем
                            detection_event.clear()
                            continue
                        # Затем проверяем, не была ли корова уже в процессе обработки
                        elif self._is_cow_already_approached(cow_coords):
                            print(f"Корова уже в процессе обработки, игнорируем: {cow_coords}")
                            # Сбрасываем событие детекции и продолжаем
                            detection_event.clear()
                            continue
                        else:
                            print(f"Новая корова обнаружена: {cow_coords}")
                            # Добавляем корову в обработанные
                            self._add_cow_to_approached(cow_coords)
                    
                    print("Обнаружена корова - начинаем логику сближения [SIM]")
                    self.last_detection_time = current_time  # Записываем время детекции
                    
                    # ПРИОСТАНАВЛИВАЕМ детекцию во время сближения
                    if self.detector is not None and hasattr(self.detector, 'pause_detection'):
                        print("Приостанавливаем детекцию...")
                        self.detector.pause_detection()
                    
                    self._approach_cow_logic(stop_event)
                    
                    # ВОЗОБНОВЛЯЕМ детекцию после сближения
                    if self.detector is not None and hasattr(self.detector, 'resume_detection'):
                        print("Возобновляем детекцию...")
                        self.detector.resume_detection()
                    
                    # Сбрасываем событие детекции
                    detection_event.clear()
                    
                    # После сближения переходим к следующей точке (независимо от результата)
                    print(f"Переходим к следующей точке маршрута")
                    # Дополнительная проверка после сближения
                    if stop_event.is_set():
                        break
                    # Переходим к следующей точке после сближения
                    self.current_point_index += 1
                    continue

                # Если детекции нет или она заблокирована - летим к точке маршрута
                x, y, z = self.mission_points[self.current_point_index]
                print(f"Перелет {self.current_point_index + 1}/{len(self.mission_points)} -> X:{x:.2f}, Y:{y:.2f}, Z:{z:.2f}")
                
                # Выполняем перелет с проверкой детекции во время полета
                self.drone.goto(x, y, z, yaw=0, wait=True, accuracy=0.2)
                
                # Проверяем детекцию после завершения перелета
                if stop_on_detection and detection_event.is_set():
                    print("Обнаружена корова после перелета - переходим к логике сближения")
                    # Не переходим к следующей точке, а обрабатываем детекцию в следующей итерации
                    continue
                
                # Переходим к следующей точке
                self.current_point_index += 1

                # короткая неблокирующая пауза с проверкой событий
                start_time = time.time()
                while time.time() - start_time < 0.1:
                    if stop_event.is_set() or (stop_on_detection and detection_event.is_set()):
                        break
                    time.sleep(0.01)

            if not detection_event.is_set() and self.current_point_index >= len(self.mission_points):
                print("Миссия успешно завершена [SIM]")
                # Выводим статистику по обработанным коровам
                print(f"\n=== СТАТИСТИКА ПО КОРОВАМ ===")
                print(f"Всего коров в процессе обработки: {len(self.cows_approached)}")
                print(f"Всего полностью обработанных коров: {len(self.processed_cows)}")
                
                if self.processed_cows:
                    print("\nДетали по коровам:")
                    for cow_key, (coords, status, timestamp) in self.processed_cows.items():
                        x, y = coords
                        print(f"  {cow_key}: ({x:.2f}, {y:.2f}) - {status}")
                print("================================")

        except Exception as e:
            print(f"Ошибка выполнения миссии [SIM]: {e}")




    def _approach_cow_logic(self, stop_event: threading.Event):
        """Логика сближения с коровой для направления её в загон."""
        global COW_TEST_COORDS
        try:
            # Определяем высоты полета
           
            # print("Начинаем логику сближения с коровой...")
            
            # Получаем текущую позицию дрона
            drone_x, drone_y, drone_z = self.drone.position[0], self.drone.position[1], self.drone.position[2]
            print(f"Позиция дрона: X:{drone_x:.2f}, Y:{drone_y:.2f}, Z:{drone_z:.2f}")
            
            # Определяем зоны для животных
            cow_zones = [
                (-2.88, -3.32),  # Загон 1 для коров
                (3.55, -2.85),   # Загон 2 для коров
                (-2.83, 1.61)    # Загон 3 для коров
            ]
            wolf_zone = (2.75, 3.02)  # Загон для волка
            
            # Находим ближайший загон
            nearest_zone = None
            min_distance = float('inf')
            for i, (zone_x, zone_y) in enumerate(cow_zones):
                distance = ((drone_x - zone_x) ** 2 + (drone_y - zone_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_zone = (zone_x, zone_y, i + 1)
            
            if nearest_zone is None:
                print("Ошибка: не удалось определить загон")
                return
            
            zone_x, zone_y, zone_num = nearest_zone
            print(f"Выбран загон #{zone_num} в координатах ({zone_x:.2f}, {zone_y:.2f})")
            
            # Функция для проверки, находится ли корова в загон
            def is_cow_in_zone(cow_x, cow_y, zone_x, zone_y):
                """Проверяет, находится ли корова в пределах загона (сторона 1.5 метра)"""
                zone_side = 1.0  # сторона загона в метрах (увеличена для более легкого попадания)
                half_side = zone_side / 2
                return (abs(cow_x - zone_x) <= half_side and abs(cow_y - zone_y) <= half_side)
            
            # Цикл сближения с коровой до попадания в загон
            attempt = 0
            
            while attempt < MAX_SCARE_ATTEMPTS and not stop_event.is_set():
                attempt += 1
                print(f"\n--- Попытка сближения #{attempt}/{MAX_SCARE_ATTEMPTS} ---")
                
                # Получаем текущие координаты коровы (обновляемся в каждой итерации)
                coords = COW_TEST_COORDS if COW_TEST_COORDS is not None else self._get_current_cow_coordinates()
                if not coords:
                    print("Нет координат коровы — прекращаем сближение.")
                    break
                cow_x, cow_y = coords
                print(f"Текущие координаты коровы: ({cow_x:.2f}, {cow_y:.2f})")
                
                # Проверяем, находится ли корова уже в загон
                if is_cow_in_zone(cow_x, cow_y, zone_x, zone_y):
                    print(f"Корова уже в загон #{zone_num}! Завершаем сближение.")
                    break
                
                # Вычисляем вектор от коровы к загону
                dx = zone_x - cow_x
                dy = zone_y - cow_y
                
                # Нормализуем вектор
                distance_to_zone = (dx ** 2 + dy ** 2) ** 0.5
                if distance_to_zone > 0:
                    dx_norm = dx / distance_to_zone
                    dy_norm = dy / distance_to_zone
                else:
                    dx_norm, dy_norm = 0, 0
                
                # Рассчитываем yaw (поворот дрона) для направления к корове
                # Вектор от дрона к корове
                drone_to_cow_dx = cow_x - drone_x
                drone_to_cow_dy = cow_y - drone_y
                
                # atan2 возвращает угол в радианах, переводим в градусы
                yaw_to_cow = np.degrees(np.arctan2(drone_to_cow_dy, drone_to_cow_dx))


                # print(f"Yaw для направления к корове: {yaw_to_cow:.1f}°")
                
                # Позиция для сближения - на одной прямой с коровой и загоном, но сзади коровы
                # Дрон должен быть позади коровы по направлению к загону
                approach_distance = 1.5  # расстояние сзади коровы
                approach_x = cow_x - dx_norm * approach_distance  # сзади коровы по направлению к загону
                approach_y = cow_y - dy_norm * approach_distance  # сзади коровы по направлению к загону
                
                print(f"Корова: ({cow_x:.2f}, {cow_y:.2f})")
                print(f"Загон: ({zone_x:.2f}, {zone_y:.2f})")
                print(f"Позиция для сближения: X:{approach_x:.2f}, Y:{approach_y:.2f}")
                
                # 1. Перелет к позиции сближения на основной высоте
                print(f"1. Перелет к позиции сближения на высоте {MAIN_FLIGHT_HEIGHT}м с yaw={yaw_to_cow:.1f}°...")
                self.drone.goto(approach_x, approach_y, MAIN_FLIGHT_HEIGHT, yaw=yaw_to_cow, wait=True, accuracy=0.2)
                
                if stop_event.is_set():
                    return
                
                # 2. Перелет к корове на высоте для пугания
                print(f"2. Перелет к корове на высоте {SCARE_HEIGHT}м с yaw={yaw_to_cow:.1f}°...")
                self.drone.goto(cow_x, cow_y, SCARE_HEIGHT, yaw=yaw_to_cow, wait=True, accuracy=0.2)
                
                if stop_event.is_set():
                    return
                
                # 3. Подъем вверх для запугивания коровы
                print(f"3. Подъем вверх для запугивания коровы с yaw={yaw_to_cow:.1f}°...")
                self.drone.goto(cow_x, cow_y, MAIN_FLIGHT_HEIGHT, yaw=yaw_to_cow, wait=True, accuracy=0.2)
                
                # Небольшая пауза для того, чтобы корова могла отреагировать
                time.sleep(2)
                
                # Проверяем, попала ли корова в загон
                print(f"Координаты коровы: ({cow_x:.2f}, {cow_y:.2f})")
                
                # Проверяем расстояние до центра загона
                distance_to_zone_center = ((cow_x - zone_x) ** 2 + (cow_y - zone_y) ** 2) ** 0.5
                print(f"Расстояние до центра загона: {distance_to_zone_center:.2f}м")
                
                if is_cow_in_zone(cow_x, cow_y, zone_x, zone_y):
                    print(f"Корова попала в загон #{zone_num}! Завершаем сближение.")
                    break
                else:
                    print(f"Корова еще не в загон. Продолжаем попытки сближения.")
            
            # Получаем финальные координаты коровы для добавления в список обработанных
            final_cow_coords = self._get_current_cow_coordinates()
            
            if attempt >= MAX_SCARE_ATTEMPTS:
                print(f"Достигнуто максимальное количество попыток сближения ({MAX_SCARE_ATTEMPTS})")
                # Добавляем корову в список пропущенных
                if final_cow_coords:
                    self._add_cow_to_processed(final_cow_coords, 'skipped')
                    print(f"Корова добавлена в список пропущенных после {MAX_SCARE_ATTEMPTS} попыток")
            else:
                print(f"Логика сближения завершена за {attempt} попыток - корова направлена в загон")
                # Добавляем корову в список загнанных
                if final_cow_coords:
                    self._add_cow_to_processed(final_cow_coords, 'herded')
                    print(f"Корова добавлена в список загнанных")
            
            # Сбрасываем событие детекции, чтобы продолжить миссию
            if hasattr(self, '_detection_event'):
                self._detection_event.clear()
                print("Детекция сброшена - продолжаем миссию")
            
        except Exception as e:
            print(f"Ошибка в логике сближения: {e}")

def main():
    # Дрон (как в основном файле, но без arm/takeoff/land)
    pioneer = Pion(ip="127.0.0.1", mavlink_port=8004, logger=True, dt=0.0, mass=0.5)

    # Камера-симулятор и контроллер детекции с подтверждением
    camera = CameraSim(timeout=1, ip='127.0.0.1', port=18004, video_buffer_size=65000, log_connection=True)
    # camera = CameraReal("rtsp://10.1.100.160:8554/pioneer_stream")

    # События должны быть созданы до передачи в детектор
    stop_event = threading.Event()
    detection_event = threading.Event()

    detector = ObjectDetectionControllerSim(camera=camera, min_detections=5, buffer_size=10)
    detector.set_external_stop_event(stop_event)
    detector.set_pioneer(pioneer)  # Передаем объект дрона для вычисления координат
    detector.start()

    # Маршрут
    flight_path: List[Tuple[float, float, float]] = load_flight_coordinates("flight_path.json")

    # Запуск отображения/детекции (всегда показывает кадры)
    display_thread = threading.Thread(target=display_loop, args=(detector, stop_event, detection_event), daemon=True)
    display_thread.start()

    try:
        print("Армирование дрона...")
        pioneer.arm()
        print("Взлет...")
        pioneer.takeoff()
        time.sleep(2)

        # Подготовка и старт миссии в отдельном потоке (как в main.py)
        mission = MissionControllerSim(drone=pioneer)
        mission.load_mission(flight_path)
        mission.set_detector(detector)  # Передаем детектор для получения координат коровы
        print(f"DEBUG: Детектор передан в миссию: {detector}")
        print(f"DEBUG: Тип детектора: {type(detector)}")

        mission_thread = threading.Thread(
            target=mission.execute_mission_non_blocking,
            args=(stop_event, detection_event, STOP_ON_DETECTION),
            daemon=True,
        )
        mission_thread.start()



        # Основной цикл ожидания завершения миссии/детекции/остановки
        while mission_thread.is_alive() and not stop_event.is_set():
            time.sleep(0.1)

        # Ждем завершения потока миссии (включая логику сближения)
        if 'mission_thread' in locals():
            mission_thread.join(timeout=30)  # Даем время на выполнение логики сближения

        if detection_event.is_set() and STOP_ON_DETECTION:
            print("Красная корова обнаружена — логика сближения выполнена.")
        elif detection_event.is_set() and not STOP_ON_DETECTION:
            print("Красная корова обнаружена — продолжаем полёт по маршруту.")
        elif stop_event.is_set():
            print("Остановка по запросу пользователя.")
        else:
            print("Маршрут завершён.")
        
        # Выводим финальную статистику по коровам
        if 'mission' in locals():
            print(f"\n=== ФИНАЛЬНАЯ СТАТИСТИКА ПО КОРОВАМ ===")
            print(f"Всего коров в процессе обработки: {len(mission.cows_approached)}")
            print(f"Всего полностью обработанных коров: {len(mission.processed_cows)}")
            
            if mission.processed_cows:
                print("\nДетали по коровам:")
                for cow_key, (coords, status, timestamp) in mission.processed_cows.items():
                    x, y = coords
                    print(f"  {cow_key}: ({x:.2f}, {y:.2f}) - {status}")
            print("=========================================")

    except KeyboardInterrupt:
        print("\nОстановка пользователем")
        pioneer.stop_moving()
        time.sleep(3)
        pioneer.land()

        stop_event.set()

    except Exception as e:
        print(f"\nОшибка: {e}")
        stop_event.set()

        pioneer.stop_moving()
        time.sleep(3)
        pioneer.land()

    finally:
        stop_event.set()
        if 'display_thread' in locals():
            display_thread.join(timeout=1)
        if 'mission_thread' in locals():
            mission_thread.join(timeout=1)
        try:
            pioneer.stop_moving()
            time.sleep(3)
            pioneer.land()
            
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Флаг поведения: останавливать ли полёт при подтверждённой детекции
    STOP_ON_DETECTION = True
    
    # Координаты коровы для тестирования (если детекция работает неправильно)
    # Установите None для автоматического определения или задайте точные координаты
    COW_TEST_COORDS = None  # или например: (-4.0, -3.8)
    # COW_TEST_COORDS = (-2.13, -2.56)

    SCARE_HEIGHT = 0.4  # высота для пугания коровы
    MAIN_FLIGHT_HEIGHT = 1.5  # основная высота полета

    MAX_SCARE_ATTEMPTS = 1  # максимальное количество попыток сближения с коровой по истечении полетит дальше

        

    main()


