import random
from enum import Enum, auto
from itertools import accumulate

import numpy as np

from utils import Angle, point_to_str, get_raw_leaf_points, get_rotation_matrix, get_point_at_quad_bez


class Partition(Enum):
    SPLIT = auto()
    OFFSHOOT = auto()
    CONTINUE = auto()


class PlantConfig:
    def __init__(self):
        self.length = 160.0
        self.length_delta = 0.7

        self.width = 30.0
        self.bulbousness = 1.0

        self.n_splits = 7
        self.n_offshoots = 0
        self.n_continues = 0

        self.left_angle = Angle(-24.0)
        self.right_angle = Angle(44.0)

        self.growth_time = 2.0

        self.leaf_shape = 0.0
        self.leaf_size = 20.0
        self.leaf_angles = (Angle(-70.0), Angle(70.0))
        self.leaf_start_at_x_branch = 0.7
        self.leaf_growth_time = 0.5
        self.leaf_placement = (
            ("l", 0.3),
            ("r", 0.3),
            ("l", 0.5),
            ("r", 0.5),
            ("l", 0.75),
            ("r", 0.75),
        )


class Branch:
    def __init__(
        self,
        length,
        width_initial,
        width_final,
        angle,
        bulbousness,
        t_start,
        growth_time,
        parent,
    ):
        self.length = length
        self.width_initial = width_initial
        self.width_final = width_final
        self.angle = angle
        self.bulbousness = bulbousness

        self.t_start = t_start
        self.t_end = t_start + growth_time

        self.parent = parent
        self.children = []

        self.is_leafed = False
        self.is_leaf = False

    def __repr__(self):
        return f"{self.__class__.__name__}({self.length}, {self.width_initial}, {self.angle.value})"

    def get_points(self, params: PlantConfig, start, t):
        cur_length = self.get_cur_length(t)
        cur_ini_width = self.get_cur_ini_width(t)
        cur_fin_width = self.get_cur_fin_width(t)

        # TODO: move
        rotation_matrix = np.array(
            [
                [
                    np.cos(np.radians(self.angle.value)),
                    np.sin(np.radians(self.angle.value)),
                ],
                [
                    -np.sin(np.radians(self.angle.value)),
                    np.cos(np.radians(self.angle.value)),
                ],
            ]
        )

        # TODO: move
        points = np.array(
            [
                [-cur_ini_width / 2.0, 0.0],
                [
                    -(cur_ini_width + cur_fin_width) / 4 * (1.0 + self.bulbousness),
                    -cur_length / 2,
                ],  #
                [-cur_fin_width / 2.0, -cur_length],
                [-cur_fin_width / 2.0, -cur_length - cur_fin_width / 2],
                [0.0, -cur_length - cur_fin_width / 2.0],
                [cur_fin_width / 2.0, -cur_length - cur_fin_width / 2],
                [cur_fin_width / 2.0, -cur_length],
                [
                    (cur_ini_width + cur_fin_width) / 4 * (1.0 + self.bulbousness),
                    -cur_length / 2,
                ],  #
                [cur_ini_width / 2.0, 0.0],
                [cur_fin_width / 2.0, cur_ini_width / 2],
                [0.0, cur_ini_width / 2],
                [-cur_fin_width / 2.0, cur_ini_width / 2],
                [-cur_ini_width / 2.0, 0.0],
            ]
        )

        points = points @ rotation_matrix
        points += start

        self.start_point_ = np.array([0.0, 0.0]) + start
        self.end_point_ = np.array([0.0, -cur_length]) @ rotation_matrix + start

        shapes = [("branch", points)]

        # TODO: move
        for leaf in params.leaf_placement:
            side, fin_placement = leaf
            leaf_alive_for = t - (
                self.t_start
                + self.get_t_in_x_length(params.leaf_start_at_x_branch * fin_placement)
            )
            if leaf_alive_for <= 0:
                break
            leaf_cur_size = self.get_cur_leaf_size(
                params.leaf_size, params.leaf_growth_time, leaf_alive_for
            )
            ini_placement = fin_placement * params.leaf_start_at_x_branch
            branch_grown = cur_length / self.length
            cur_leaf_placement = (
                ini_placement
                + (fin_placement - ini_placement)
                * (branch_grown - ini_placement)
                / (1.0 - ini_placement)
            ) / branch_grown

            p1 = points[0]
            c = points[1]
            p2 = points[2]
            if side == "r":
                p1 = points[8]
                c = points[7]
                p2 = points[6]

            start_point = get_point_at_quad_bez(p1, c, p2, cur_leaf_placement)

            shapes.append(
                (
                    "leaf",
                    self.get_leaf_points(
                        params.leaf_shape,
                        side,
                        leaf_cur_size,
                        start_point,
                        params.leaf_angles,
                        self.angle,
                    ),
                )
            )

        return shapes

    def get_leaf_points(
        self, shape, side, cur_size, start_point, leaf_angles, branch_angle
    ):
        points = get_raw_leaf_points(shape)
        points *= cur_size
        if side == "l":
            angle = branch_angle + leaf_angles[0]
        else:
            angle = branch_angle + leaf_angles[1]
        points = points @ get_rotation_matrix(angle)
        points += start_point
        return points

    def get_cur_length(self, t):
        if self.t_end <= t:
            return self.length
        length = (t - self.t_start) / (self.t_end - self.t_start) * self.length
        return length

    def get_t_in_x_length(self, x):
        return x * (self.t_end - self.t_start)

    def get_cur_ini_width(self, t):
        if self.t_end <= t:
            return self.width_initial
        ini_width = (
            (t - self.t_start) / (self.t_end - self.t_start) * self.width_initial
        )
        return ini_width

    def get_cur_fin_width(self, t):
        if self.t_end <= t:
            return self.width_final
        fin_width = (t - self.t_start) / (self.t_end - self.t_start) * self.width_final
        return fin_width

    def get_cur_leaf_size(self, leaf_max_size, leaf_growth_time, alive_for):
        if alive_for >= leaf_growth_time:
            return leaf_max_size
        return (alive_for / leaf_growth_time) ** 0.5 * leaf_max_size


class Plant:
    def __init__(self, params: PlantConfig) -> None:
        self.params = params
        self.root_ = None

    def generate_plant(self):
        self.root_ = self.create_branch(Angle(0.0), 0.0, None)
        self.update_params()
        current_leaves = [self.root_]

        while (
            self.params.n_splits + self.params.n_offshoots + self.params.n_continues
            >= len(current_leaves)
        ):
            new_leaves = []
            for leaf in current_leaves:
                self.choose_and_resolve_partition(leaf)
                new_leaves.extend(leaf.children)
            current_leaves = new_leaves
            self.update_params()

    def draw_plant(self, file_path, t):
        shapes = []
        root = self.root_

        def add_points(node: Branch, start_point):
            if node.t_start > t:
                return
            shapes.extend(node.get_points(self.params, start_point, t))
            for child in node.children:
                add_points(child, node.end_point_)

        add_points(root, np.array([0.0, 0.0]))

        strokes = []
        fills = []
        bounding_box = [0.0, 0.0, 0.0, 0.0]
        for shape in shapes:
            _, points = shape
            bounding_box[0] = min(bounding_box[0], np.min(points[:, 0]))
            bounding_box[1] = min(bounding_box[1], np.min(points[:, 1]))
            bounding_box[2] = max(bounding_box[2], np.max(points[:, 0]))
            bounding_box[3] = max(bounding_box[3], np.max(points[:, 1]))
            outline, fill = self.get_svg_paths(shape)
            strokes.append(outline)
            fills.append(fill)
        viewbox = [
            bounding_box[0] - 100.0,
            bounding_box[1] - 100.0,
            bounding_box[2] - bounding_box[0] + 200,
            bounding_box[3] - bounding_box[1] + 200,
        ]

        prefix = f'<svg version="1.1" viewBox="{" ".join([str(p) for p in viewbox])}" xmlns="http://www.w3.org/2000/svg">\n\n'
        paths = "\n".join(strokes) + "\n".join(fills) + "\n\n"
        suffix = "</svg>"
        content = prefix + paths + suffix

        with open(file_path, "w") as f:
            f.writelines(content)

    def get_svg_paths(self, shape):
        shape_type, points = shape
        if shape_type == "branch":
            path_d = "M%s Q%s %s Q%s %s Q%s %s Q%s %s Q%s %s Q%s %s Z" % tuple(
                point_to_str(point) for point in points
            )
            stroke = (
                '<path d="'
                + path_d
                + '" stroke-opacity="1" stroke="black" fill="none" stroke-width="16" stroke-linecap="square"/>'
            )
            fill = (
                '<path d="'
                + path_d
                + '" stroke="none" stroke-width="2" stroke-linecap="square" fill="#702f03"/>'
            )
        elif shape_type == "leaf":
            path_d = path_d = " ".join(point_to_str(p, ",") for p in points)
            stroke = (
                '<polygon points="'
                + path_d
                + '" stroke-opacity="1" stroke="black" fill="none" stroke-width="16" stroke-linejoin="round"/>'
            )
            fill = (
                '<polygon points="'
                + path_d
                + '" stroke="#095721" stroke-width="2" stroke-linecap="square" fill="green"/>'
            )

        return stroke, fill

    def create_branch(self, angle, t_start, parent):
        return Branch(
            self.params.length,
            self.params.width,
            self.params.width * self.params.length_delta,
            angle,
            self.params.bulbousness,
            t_start,
            self.params.growth_time,
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
        partitions = [
            self.params.n_splits,
            self.params.n_offshoots,
            self.params.n_continues,
        ]
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
            left_child = self.create_branch(
                parent.angle + self.params.left_angle,
                parent.t_start + self.params.growth_time,
                parent,
            )
            right_child = self.create_branch(
                parent.angle + self.params.right_angle,
                parent.t_start + self.params.growth_time,
                parent,
            )
            parent.children.extend([left_child, right_child])
        if partition == Partition.OFFSHOOT:
            if random.random() < 0.5:
                left_child = self.create_branch(
                    parent.angle + self.params.left_angle,
                    parent.t_start + self.params.growth_time,
                    parent,
                )
                right_child = self.create_branch(
                    parent.angle, parent.t_start + self.params.growth_time, parent
                )
                parent.children.extend([left_child, right_child])
            else:
                left_child = self.create_branch(
                    parent.angle, parent.t_start + self.params.growth_time, parent
                )
                right_child = self.create_branch(
                    parent.angle + self.params.right_angle,
                    parent.t_start + self.params.growth_time,
                    parent,
                )
                parent.children.extend([left_child, right_child])
        if partition == Partition.CONTINUE:
            child = self.create_branch(
                parent.angle, parent.t_start + self.params.growth_time, parent
            )
            parent.children.append(child)
