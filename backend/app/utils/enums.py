from enum import Enum 

class StorageType(str, Enum):
    TEMP = "temp"
    PERMANENT = "permanent"