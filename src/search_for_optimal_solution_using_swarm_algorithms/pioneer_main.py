from pioneer_sdk import Pioneer
import logging
import cv2
from flight_utils import load_flight_coordinates
from flight_mission_runner import FlightMissionRunner
import time
import math

from camera_real import CameraReal  # класс для работы с камерой rtsp потока или видеофайлом
from camera_sim import CameraSim  # класс для работы с камерой в симуляторе
from object_detection_controller_hybrid import NeuralObjectDetectionController

# Параметры сближения
SCARE_HEIGHT = 0.8       # высота для пугания коровы
MAIN_FLIGHT_HEIGHT = 1.5  # основная высота полета
MAX_SCARE_ATTEMPTS = 3    # максимальное число попыток сближения
REQUIRED_STABLE_FRAMES = 3  # сколько подряд стабильных кадров ждать перед сближением


def _is_cow_in_zone(cow_x: float, cow_y: float, zone_x: float, zone_y: float, zone_side: float = 1.0) -> bool:
    half = zone_side / 2.0
    return (abs(cow_x - zone_x) <= half) and (abs(cow_y - zone_y) <= half)


def _wait_until_reached(pioneer: Pioneer, timeout: float = 10.0, check_dt: float = 0.05) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if pioneer.point_reached():
            return True
        time.sleep(check_dt)
    return False


def approach_cow_once(pioneer: Pioneer,
                     camera,
                     detector: NeuralObjectDetectionController,
                     cow_coords: tuple[float, float],
                     assigned_zone_type: str,
                     assigned_zone_idx: int) -> bool:
    """Выполняет одну попытку сближения по логике _approach_cow_logic.
    Возвращает True, если после манёвра корова оказалась в загоне.
    """
    if assigned_zone_type not in detector.ZONES or assigned_zone_idx < 0:
        return False

    zone_x, zone_y = detector.ZONES[assigned_zone_type][assigned_zone_idx]
    cow_x, cow_y = cow_coords
    
    # Проверяем, не находится ли корова уже в загоне
    if _is_cow_in_zone(cow_x, cow_y, zone_x, zone_y, zone_side=1.0):
        print(f"Корова уже находится в загоне {assigned_zone_type} #{assigned_zone_idx + 1} - сближение не требуется")
        return True  # Возвращаем True, так как корова уже в нужном месте

    # Вектор к зоне и нормализация
    dx = zone_x - cow_x
    dy = zone_y - cow_y
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > 0:
        dx_n = dx / dist
        dy_n = dy / dist
    else:
        dx_n, dy_n = 0.0, 0.0

    # Позиция для сближения: за спиной коровы на прямой с загоном
    approach_distance = 1.5
    approach_x = cow_x - dx_n * approach_distance
    approach_y = cow_y - dy_n * approach_distance
    
    print(f"Координаты коровы: ({cow_x:.3f}, {cow_y:.3f})")
    print(f"Координаты загона: ({zone_x:.3f}, {zone_y:.3f})")
    
    # Рассчитываем yaw для направления дрона головой на корову
    cow_to_approach_dx = cow_x - approach_x
    cow_to_approach_dy = cow_y - approach_y
    yaw_to_cow = math.degrees(math.atan2(-cow_to_approach_dx, cow_to_approach_dy))
    yaw_to_cow = (yaw_to_cow + 180) % 360 - 180
    
    if yaw_to_cow > 90:
        yaw_to_cow = yaw_to_cow - 180
    elif yaw_to_cow < -90:
        yaw_to_cow = yaw_to_cow + 180
    
    yaw_to_cow = -yaw_to_cow
    
    if yaw_to_cow > 0:
        direction = "влево"
    elif yaw_to_cow < 0:
        direction = "вправо"
    else:
        direction = "прямо"
    print(f"Дрон повернется {direction} на {abs(yaw_to_cow):.1f}°")
    
    if abs(yaw_to_cow) > 180:
        print(f"ВНИМАНИЕ: Yaw слишком большой: {yaw_to_cow:.1f}°")
        yaw_to_cow = 180 if yaw_to_cow > 0 else -180
        print(f"Yaw ограничен до: {yaw_to_cow:.1f}°")

    # 1. Перелет к позиции сближения на основной высоте
    pioneer.go_to_local_point(x=approach_x, y=approach_y, z=MAIN_FLIGHT_HEIGHT, yaw=yaw_to_cow)
    _wait_until_reached(pioneer, timeout=10.0)

    # 2. Перелет к корове на высоте пугания
    pioneer.go_to_local_point(x=cow_x, y=cow_y, z=SCARE_HEIGHT, yaw=yaw_to_cow)
    _wait_until_reached(pioneer, timeout=10.0)

    # 3. Подъем вверх для пугания
    pioneer.go_to_local_point(x=cow_x, y=cow_y, z=MAIN_FLIGHT_HEIGHT, yaw=yaw_to_cow)
    _wait_until_reached(pioneer, timeout=10.0)

    time.sleep(2.0)

    # Проверка положения коровы после маневра
    frame = camera.get_cv_frame()
    if frame is None:
        return False

    current_pos = pioneer.get_local_position_lps(get_last_received=True)
    if current_pos is not None:
        drone_x, drone_y = current_pos[0], current_pos[1]
    else:
        drone_x, drone_y = 0.0, 0.0

    result = detector.process_frame(frame, drone_x=drone_x, drone_y=drone_y)
    agc = result.get("averaged_global_coords")
    if agc is None:
        return False

    in_zone = _is_cow_in_zone(agc[0], agc[1], zone_x, zone_y, zone_side=1.0)
    return bool(in_zone)


flight_height = float(1.5)
FULL_MAP_COVERAGE_POINTS = load_flight_coordinates("flight_path.json")



if __name__ == "__main__":
    try:
        mission = FlightMissionRunner(FULL_MAP_COVERAGE_POINTS)
       
        # Выбор между симуляцией и реальной камерой
        USE_SIMULATION = False  # Измените на False для использования реальной камеры
        
        if USE_SIMULATION:
            camera = CameraSim(ip="127.0.0.1", port=18004, log_connection=True, timeout=2)
            print("Используется симуляция камеры")
        else:
            camera = CameraReal("/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/videos/output_pascal_line2.mp4")
            print("Используется реальная камера")

        # Инициализация
        detector = NeuralObjectDetectionController(
            model_path='/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/weights/best_last_sana.pt',
            conf_threshold=0.5,
            min_detections=5,
            buffer_size=10,
            use_yolo=True,
            device='cpu'
        )

        pioneer = Pioneer(
            name="pioneer", 
            ip="127.0.0.1", 
            mavlink_port=8004, 
            connection_method="udpout", 
            device="dev/serial0", 
            baud=115200, log_connection=True, logger=True
        )

        pioneer.arm()
        pioneer.takeoff()

        first_point = mission.get_next_point()
        x, y, z = first_point
        pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)

        # Текущая назначенная корова
        current_assigned_type = None
        current_assigned_idx = None
        current_assignment_handled = False
        waiting_for_stable = False
        stable_frames_count = 0
        
        while not mission.is_complete():
            frame = camera.get_cv_frame()
            if frame is None:
                cv2.waitKey(1)
                continue

            # Текущее положение дрона
            current_pos = pioneer.get_local_position_lps(get_last_received=True)
            if current_pos is not None:
                drone_x, drone_y = current_pos[0], current_pos[1]
            else:
                drone_x, drone_y = 0.0, 0.0

            # Детекция + распределение по зонам
            result = detector.process_frame(frame, drone_x, drone_y)
            annotated = result.get("annotated", frame)

            # Обработка событий
            if result.get("coords_event") or result.get("assignment_event"):
                agc = result.get("averaged_global_coords")
                zt, zi = result.get("zone_type"), result.get("zone_index")
                
                if result.get("coords_event") and agc is not None:
                    print(f"Обнаружен объект: глобальные координаты ({agc[0]:.2f}, {agc[1]:.2f})")
                
                if result.get("assignment_event"):
                    azt, azi = result.get("assigned_zone_type"), result.get("assigned_zone_index")
                   
                    if azt != "unknown" and azi is not None and azi >= 0:
                        print(f"Назначение в ближайший свободный загон: {azt} #{azi + 1}")

                    # Обновляем текущую назначенную цель
                    current_assigned_type = azt if azt != "unknown" else None
                    current_assigned_idx = azi if isinstance(azi, int) and azi >= 0 else None
                    current_assignment_handled = False
                    waiting_for_stable = True
                    stable_frames_count = 0
                    print(f"Ожидаем стабилизации координат ({REQUIRED_STABLE_FRAMES} кадра)...")

            # Запуск сближения после стабилизации координат
            if (current_assigned_type is not None and current_assigned_idx is not None 
                and not current_assignment_handled and waiting_for_stable):
                
                agc = result.get("averaged_global_coords")
                if agc is not None:
                    if not result.get("coords_event"):
                        stable_frames_count += 1
                    else:
                        stable_frames_count = 0

                    if stable_frames_count >= REQUIRED_STABLE_FRAMES:
                        print("Координаты стабилизированы — начинаем сближение")
                        success = False
                        for attempt in range(MAX_SCARE_ATTEMPTS):
                            print(f"Старт сближения, попытка {attempt+1}/{MAX_SCARE_ATTEMPTS}...")
                            success = approach_cow_once(
                                pioneer=pioneer,
                                camera=camera,
                                detector=detector,
                                cow_coords=(agc[0], agc[1]),
                                assigned_zone_type=current_assigned_type,
                                assigned_zone_idx=current_assigned_idx,
                            )
                            if success:
                                print(f"Корова направлена в загон: {current_assigned_type} #{current_assigned_idx + 1}")
                                break
                            else:
                                print("Сближение не удалось — повторная попытка")
                        
                        current_assignment_handled = True
                        waiting_for_stable = False
                        stable_frames_count = 0
                        
                        # Продолжаем маршрут
                        next_point = mission.get_next_point()
                        if next_point:
                            nx, ny, nz = next_point
                            pioneer.go_to_local_point(x=nx, y=ny, z=flight_height, yaw=0)
                        
                        current_assigned_type = None
                        current_assigned_idx = None

            # Движение по маршруту
            if pioneer.point_reached():
                next_point = mission.get_next_point()
                if next_point:
                    x, y, z = next_point
                    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)

            # Показ кадра
            cv2.imshow("frame", annotated if annotated is not None else frame)
            cv2.waitKey(1)

        pioneer.land()

    except KeyboardInterrupt:
        print("\nОстановка пользователем")
        pioneer.land()
        time.sleep(3)
        pioneer.close_connection()

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}", exc_info=True)
        raise
    finally:
        print("Завершение работы")
        pioneer.disarm()
        pioneer.close_connection()
        del pioneer