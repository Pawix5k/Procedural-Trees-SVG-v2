import random

from plant import PlantConfig, Plant
from utils import Vector


random.seed(343)

plant = Plant(PlantConfig())
plant.generate_plant()

plant.draw_plant(r"plant.svg", 6.7)
