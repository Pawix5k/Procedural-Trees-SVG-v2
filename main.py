from plant import PlantConfig, Plant
from utils import Vector

plant = Plant(PlantConfig())
plant.generate_plant()

plant.draw_plant(r"plant.svg", 10.0)
