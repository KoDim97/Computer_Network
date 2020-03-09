from enum import Enum
from src.ospf_net import Net
import math

class ActionType(Enum):
    None_ = 0
    Add = 1
    Scenario = 2
    Exit = 3


class Action:
    def __init__(self):
        self.actionType: ActionType = ActionType.None_

    def start(self, net: Net):
        pass


class AddAction(Action):
    def __init__(self, x: float, y: float):
        super().__init__()
        self.actionType = ActionType.Add
        self.x = x
        self.y = y

    def start(self, net: Net):
        net.add_router(self.x, self.y)

class ScenarioCircle(Action):
    POINT_COUNT = 6
    def __init__(self):
        super().__init__()
        self.actionType = ActionType.Scenario
        self.points = []
        r = 0.2
        for p_count in range(self.POINT_COUNT):
            x = 0.5 + r * math.sin(2 * math.pi / self.POINT_COUNT * p_count)
            y = 0.5 + r * math.cos(2 * math.pi / self.POINT_COUNT * p_count)
            self.points.insert(p_count, [x,y])

    def start(self, net: Net):
        for p_count in range(self.POINT_COUNT):
            net.add_router(self.points[p_count][0], self.points[p_count][1])

class ScenarioPolygon(Action):
    POINT_COUNT = 5
    def __init__(self):
        super().__init__()
        self.actionType = ActionType.Scenario
        self.points = []
        r = 0.25
        for p_count in range(self.POINT_COUNT - 1):
            x = 0.5 + r * math.sin(2 * math.pi / (self.POINT_COUNT - 1) * p_count)
            y = 0.5 + r * math.cos(2 * math.pi / (self.POINT_COUNT - 1) * p_count)
            self.points.insert(p_count, [x,y])

        self.points.insert(self.POINT_COUNT - 1, [0.5, 0.5])

    def start(self, net: Net):
        for p_count in range(self.POINT_COUNT):
            net.add_router(self.points[p_count][0], self.points[p_count][1])


class ScenarioMill(Action):
    POINT_COUNT = 7
    def __init__(self):
        super().__init__()
        self.actionType = ActionType.Scenario
        self.points = []
        r = 0.2
        for p_count in range(self.POINT_COUNT - 1):
            x = 0.5 + r * math.sin(2 * math.pi / (self.POINT_COUNT - 1) * p_count)
            y = 0.5 + r * math.cos(2 * math.pi / (self.POINT_COUNT - 1) * p_count)
            self.points.insert(p_count, [x,y])

        self.points.insert(self.POINT_COUNT - 1, [0.5, 0.5])

    def start(self, net: Net):
        for p_count in range(self.POINT_COUNT):
            net.add_router(self.points[p_count][0], self.points[p_count][1])

class ExitAction(Action):
    def __init__(self):
        super().__init__()
        self.actionType = ActionType.Exit
