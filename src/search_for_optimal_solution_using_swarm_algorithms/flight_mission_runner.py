class FlightMissionRunner:
    def __init__(self, points=None):
        """Инициализация миссии с параметрами поля и обзора дрона"""
        self.is_complete_var = False

        self.points = points if points else self._generate_flight_path()
        self.current_index = 0
    

    
    def has_more_points(self):
        """Проверяет, остались ли точки для посещения"""
        if self.current_index < len(self.points):
            return True
        else:
            self.is_complete_var = True
            return False
        
    
    def get_next_point(self):
        """
        Возвращает следующую точку маршрута
        Returns:
            tuple|None: (x, y) или None если маршрут завершён
        """
        if self.has_more_points():
            point = self.points[self.current_index]
            self.current_index += 1
            return point 
        
        return None
    
    
    def get_total_points(self):
        """Возвращает общее количество точек в маршруте"""
        return len(self.points)
    
    def get_current_progress(self):
        """Возвращает прогресс выполнения миссии в процентах"""
        return (self.current_index / len(self.points)) * 100 if self.points else 0
    

    def is_complete(self):
        return self.is_complete_var
