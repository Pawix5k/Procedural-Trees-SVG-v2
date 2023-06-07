import random
from enum import Enum, auto
from itertools import accumulate

from utils import Angle, Vector


class Partition(Enum):
    SPLIT = auto()
    OFFSHOOT = auto()
    CONTINUE = auto()


class Branch:
    def __init__(self, length, width_initial, width_final, angle, bulbousness, parent):
        self.length = length
        self.width_initial = width_initial
        self.width_final = width_final
        self.angle = angle
        self.bulbousness = bulbousness

        self.parent = parent
        self.children = []

    def __repr__(self):
        return f"{self.__class__.__name__}({self.length}, {self.width_initial}, {self.angle.value})"

    def get_svg_paths(self, start):
        bottom = Vector(0, 0).rotate(self.angle) + start
        top = Vector(0, -self.length).rotate(self.angle) + start
        offset_left_bottom = (0.5 * Vector(-self.width_initial, 0)).rotate(self.angle)
        offset_left_top = (0.5 * Vector(-self.width_final, 0)).rotate(self.angle)

        self.start_point_: Vector = bottom
        self.end_point_: Vector = top

        p1 = bottom + offset_left_bottom
        p2 = top + offset_left_top
        p3 = top + offset_left_top.rotate(90.0)
        p4 = top + offset_left_top.rotate(180.0)
        p5 = bottom + offset_left_bottom.rotate(180.0)
        p6 = bottom + offset_left_bottom.rotate(-90.0)
        c1 = 0.5 * (top + bottom) + 0.5 * (1 + self.bulbousness) * (
            offset_left_bottom + offset_left_top
        )
        c2 = p2 + offset_left_top.rotate(90.0)
        c3 = p3 + offset_left_top.rotate(180.0)
        c4 = 0.5 * (top + bottom) - 0.5 * (1 + self.bulbousness) * (
            offset_left_bottom + offset_left_top
        )
        c5 = p6 - offset_left_bottom
        c6 = p6 + offset_left_bottom

        points = [p1, p2, p3, p4, p5, p6, c1, c2, c3, c4, c5, c6]
        stringified = [p.to_str() for p in points]

        p1, p2, p3, p4, p5, p6, c1, c2, c3, c4, c5, c6 = stringified

        path_d = (
            f"M{p1} Q{c1} {p2} Q{c2} {p3} Q{c3} {p4} Q{c4} {p5} Q{c5} {p6} Q{c6} {p1} Z"
        )
        sides_path_d = f"M{p1} Q{c1} {p2} M{p4} Q{c4} {p5}"

        beg = '<path d="'
        end_outline = '" stroke-opacity="1" stroke="black"\
            fill="none" stroke-width="16" stroke-linecap="square"/>'
        end_fill = (
            '" stroke="none" stroke-width="2" stroke-linecap="square" fill="#702f03"/>'
        )
        end_sides_outline = (
            '" stroke="#381700" stroke-width="4" stroke-linecap="square" fill="none"/>'
        )
        outline = beg + path_d + end_outline

        fill = beg + sides_path_d + end_sides_outline + beg + path_d + end_fill
        return outline, fill


class PlantConfig:
    def __init__(self):
        self.length = 160.0
        self.length_delta = 0.7

        self.width = 40.0
        self.bulbousness = 1.0

        self.n_splits = 31
        self.n_offshoots = 0
        self.n_continues = 0

        self.left_angle = Angle(-24.0)
        self.right_angle = Angle(24.0)


class Plant:
    def __init__(self, params: PlantConfig) -> None:
        self.params = params
        self.root_ = None

    def generate_plant(self):
        self.root_ = self.create_branch(Angle(0.0), None)
        self.update_params()
        current_leaves = [self.root_]

        while self.params.n_splits + self.params.n_offshoots + self.params.n_continues >= len(
            current_leaves
        ):
            new_leaves = []
            for leaf in current_leaves:
                self.choose_and_resolve_partition(leaf)
                new_leaves.extend(leaf.children)
            current_leaves = new_leaves
            self.update_params()

    def draw_plant(self, file_path):
        outlines = []
        fills = []
        root = self.root_
        bounding_box = [0.0, 0.0, 0.0, 0.0]

        def add_path(node: Branch, start_point):
            outline, fill = node.get_svg_paths(start_point)
            bounding_box[0] = min(
                bounding_box[0], node.start_point_.x, node.end_point_.x
            )
            bounding_box[1] = min(
                bounding_box[1], node.start_point_.y, node.end_point_.y
            )
            bounding_box[2] = max(
                bounding_box[2], node.start_point_.x, node.end_point_.x
            )
            bounding_box[3] = max(
                bounding_box[3], node.start_point_.y, node.end_point_.y
            )
            outlines.append(outline)
            fills.append(fill)
            for child in node.children:
                add_path(child, node.end_point_)

        add_path(root, Vector(0.0, 0.0))

        print(bounding_box)

        viewbox = [
            str(bounding_box[0] - 100),
            str(bounding_box[1] - 100),
            str(bounding_box[2] - bounding_box[0] + 200),
            str(bounding_box[3] - bounding_box[1] + 200),
        ]

        beg = f'<svg version="1.1" viewBox="{" ".join(viewbox)}" xmlns="http://www.w3.org/2000/svg">'
        end = "</svg>"

        content = beg + "\n".join(outlines) + "\n".join(fills) + end

        with open(file_path, "w", encoding="utf8") as f:
            f.writelines(content)

    def create_branch(self, angle, parent):
        return Branch(
            self.params.length,
            self.params.width,
            self.params.width * self.params.length_delta,
            angle,
            self.params.bulbousness,
            parent,
        )

    def update_params(self):
        self.params.length *= self.params.length_delta
        self.params.width *= self.params.length_delta

    def choose_and_resolve_partition(self, parent):
        partition = self.choose_partition()
        self.resolve_partition(parent, partition)

    # TODO: refactor that shit
    def choose_partition(self):
        partitions = [self.params.n_splits, self.params.n_offshoots, self.params.n_continues]
        accumulated_partitions = list(accumulate(partitions))
        i = random.randint(0, sum(partitions) - 1)
        if i < accumulated_partitions[0]:
            self.params.n_splits -= 1
            return Partition.SPLIT
        if i < accumulated_partitions[1]:
            self.params.n_offshoots -= 1
            return Partition.OFFSHOOT
        if i < accumulated_partitions[2]:
            self.params.n_continues -= 1
            return Partition.CONTINUE

    def resolve_partition(self, parent: Branch, partition):
        if partition == Partition.SPLIT:
            left_child = self.create_branch(parent.angle + self.params.left_angle, parent)
            right_child = self.create_branch(parent.angle + self.params.right_angle, parent)
            parent.children.extend([left_child, right_child])
        if partition == Partition.OFFSHOOT:
            if random.random() < 0.5:
                left_child = self.create_branch(parent.angle + self.params.left_angle, parent)
                right_child = self.create_branch(parent.angle, parent)
                parent.children.extend([left_child, right_child])
            else:
                left_child = self.create_branch(parent.angle, parent)
                right_child = self.create_branch(
                    parent.angle + self.params.right_angle, parent
                )
                parent.children.extend([left_child, right_child])
        if partition == Partition.CONTINUE:
            child = self.create_branch(parent.angle, parent)
            parent.children.append(child)
