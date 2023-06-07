from plant import Plant
from utils import Vector

plant = Plant()
plant.generate_plant()
plant.draw_plant(r"plant.svg")

# def print_plant(node):
#     if not node:
#         return
#     print(node.get_svg_outline(Vector(0.0, 0.0)))
#     for child in node.children:
#         print_plant(child)

# print_plant(plant.root_)


