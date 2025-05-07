class Sensor:
    def __init__(self):
        self.name = "sensor"
        self.type = "sensor"
    
    def set_collect_info(self, collect_info:list[str]):
       self.collect_info = collect_info
    
    def get(self):
        if self.collect_info is None:
            print("collect_info is not set")
            return None
        info = self.get_information()
        return {collect_info: info[collect_info] for collect_info in self.collect_info}

    def __repr__(self):
        return f"Base Sensor, can't be used directly \n \
                name: {self.name} \n \
                type: {self.type}"
    
        
        