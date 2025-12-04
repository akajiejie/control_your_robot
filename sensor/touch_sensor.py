import sys
sys.path.append("./")

from sensor.sensor import Sensor

class TouchSensor(Sensor):
    def __init__(self):
        super().__init__()
        self.name = "touch_sensor"
        self.type = "touch_sensor"
        self.collect_info = None

    def get_information(self):
        touch = self.get_touch()
        # 只遍历一次 collect_info，直接从 touch 中提取对应的数据
        return {key: touch[key] for key in self.collect_info if key in touch}

    
    

