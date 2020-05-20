from enum import Enum
from src.render import Renderer, pygame
from src.system.periscope import Periscope, MirrorLocation, Target
from src.parser import parse, get_project_root
from src.geometry import *
import multiprocessing as mp
from src.algorithms.direct import DirectAlgorithm, Triangle, Point3d
import sys
import datetime


class SolveAlgorithm(Enum):
    DIRECT = 1
    NEURAL_NET = 2


class TargetMoveMode(Enum):
    KEYBOARD = 1
    RANDOM_MOVE = 2


class PeriscopeApplication:
    def __init__(
            self,
            input_model: str = '2d',
            algorithm: SolveAlgorithm = SolveAlgorithm.DIRECT,
            mode: str = '',
    ):
        self.log_list = []
        pygame.init()
        self.input_model = input_model
        self.mode = mode
        config = parse(input_model)

        self.periscope: Periscope = Periscope(config)
        p_target = self.periscope.ray_to_aim().intersect_plane(
            Triangle(Point3d(0.4, 0.7, 0.2),
                     Point3d(0.5, 0.4, 0.1),
                     Point3d(0.2, 0.6, 0.5)
                     ))
        p_target = Point3d(0.62, 0.66, 0)
        tee = Target(p_target, config["target_radius"])
        self.periscope.set_target(tee)

        self.renderer = Renderer(self.periscope)

        # Shared memory
        self.down_plane_points = mp.Array('d', 6)
        self.up_plane_points = mp.Array('d', 6)
        self.plane_3_points = mp.Array('d', 6)
        self.__init_share_memory()

        self.up_plane_queue = mp.Queue()
        self.down_plane_queue = mp.Queue()
        self.plane_3_queue = mp.Queue()

        if algorithm == SolveAlgorithm.DIRECT:
            self.up_plane_process: mp.Process = mp.Process(target=DirectAlgorithm.plane_direct_process,
                                                           args=(
                                                           self.up_plane_queue, self.up_plane_points, self.periscope,
                                                           MirrorLocation.UP))
            self.down_plane_process: mp.Process = mp.Process(target=DirectAlgorithm.plane_direct_process,
                                                             args=(self.down_plane_queue, self.down_plane_points,
                                                                   self.periscope, MirrorLocation.DOWN))
            self.plane_3_process: mp.Process = mp.Process(target=DirectAlgorithm.plane_direct_process,
                                                          args=(self.plane_3_queue, self.plane_3_points, self.periscope,
                                                                MirrorLocation.THIRD))
        else:  # algorithm == SolveAlgorithm.NEURAL_NET:
            from keras.models import load_model
            from src.algorithms.net import NeuralNetAlgorithm
            model = load_model(str(get_project_root()) + '\\src\\neuralnet\\' + self.input_model + '_model.h5')
            self.up_plane_process: mp.Process = mp.Process(target=NeuralNetAlgorithm.run,
                                                           args=(
                                                           self.up_plane_queue, self.up_plane_points, self.periscope,
                                                           MirrorLocation.UP, model))
            self.down_plane_process: mp.Process = mp.Process(target=NeuralNetAlgorithm.run,
                                                             args=(self.down_plane_queue, self.down_plane_points,
                                                                   self.periscope, MirrorLocation.DOWN, model))
            self.plane_3_process: mp.Process = mp.Process(target=NeuralNetAlgorithm.run,
                                                          args=(self.plane_3_queue, self.plane_3_points, self.periscope,
                                                                MirrorLocation.THIRD, model))

    def __init_share_memory(self):
        self.down_plane_points[0] = self.periscope.mirror_down.triangle.point_b.x
        self.down_plane_points[1] = self.periscope.mirror_down.triangle.point_b.y
        self.down_plane_points[2] = self.periscope.mirror_down.triangle.point_b.z
        self.down_plane_points[3] = self.periscope.mirror_down.triangle.point_c.x
        self.down_plane_points[4] = self.periscope.mirror_down.triangle.point_c.y
        self.down_plane_points[5] = self.periscope.mirror_down.triangle.point_c.z

        self.up_plane_points[0] = self.periscope.mirror_up.triangle.point_b.x
        self.up_plane_points[1] = self.periscope.mirror_up.triangle.point_b.y
        self.up_plane_points[2] = self.periscope.mirror_up.triangle.point_b.z
        self.up_plane_points[3] = self.periscope.mirror_up.triangle.point_c.x
        self.up_plane_points[4] = self.periscope.mirror_up.triangle.point_c.y
        self.up_plane_points[5] = self.periscope.mirror_up.triangle.point_c.z

        self.plane_3_points[0] = self.periscope.mirror_3.triangle.point_b.x
        self.plane_3_points[1] = self.periscope.mirror_3.triangle.point_b.y
        self.plane_3_points[2] = self.periscope.mirror_3.triangle.point_b.z
        self.plane_3_points[3] = self.periscope.mirror_3.triangle.point_c.x
        self.plane_3_points[4] = self.periscope.mirror_3.triangle.point_c.y
        self.plane_3_points[5] = self.periscope.mirror_3.triangle.point_c.z

    def backray(self, p3_intersect, p_aim, delta):
        new_taget = Point3d(p_aim.x + delta, p_aim.y, p_aim.z)
        ret_ray = Ray(new_taget, Vector(p3_intersect, p_aim))

        p3_new = ret_ray.intersect_plane(self.periscope.mirror_3.triangle)

        p2_new = ret_ray.reflect_plane(self.periscope.mirror_3.triangle). \
            intersect_plane(self.periscope.mirror_up.triangle)

        p1_new = ret_ray.reflect_plane(self.periscope.mirror_3.triangle). \
            reflect_plane(self.periscope.mirror_up.triangle).intersect_plane(self.periscope.mirror_down.triangle)

        target = ret_ray.reflect_plane(self.periscope.mirror_3.triangle). \
            reflect_plane(self.periscope.mirror_up.triangle).reflect_plane(self.periscope.mirror_down.triangle).dir

        p5 = Point3d(p1_new.x + 0.25 * target.x, p1_new.y + 0.25 * target.y, p1_new.z + 0.25 * target.z)

        return (p5.get_point(), p1_new.get_point(), p2_new.get_point(), p3_new.get_point(), new_taget.get_point(),
                   p_aim.get_point())

    def __move_target(self) -> (bool, bool):
        exit_app = False
        need_rebuild = False
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                exit_app = True
            elif i.type == pygame.KEYDOWN:
                need_rebuild = True
                if i.key == pygame.K_UP:
                    self.periscope.target.location.y += 0.01
                elif i.key == pygame.K_DOWN:
                    self.periscope.target.location.y -= 0.01
                elif i.key == pygame.K_RIGHT:
                    self.periscope.target.location.x += 0.01
                elif i.key == pygame.K_LEFT:
                    self.periscope.target.location.x -= 0.01
                elif i.key == pygame.K_KP1:
                    self.periscope.target.location.z -= 0.01
                elif i.key == pygame.K_KP2:
                    self.periscope.target.location.z += 0.01

        return exit_app, need_rebuild

    def run(self):
        self.up_plane_process.start()
        self.down_plane_process.start()
        self.plane_3_process.start()
        tee = self.periscope.target

        exit_app = False
        iteration = 0
        while not exit_app:
            p1_intersect = self.periscope.laser.intersect_plane(self.periscope.mirror_down.triangle)
            p2_intersect = self.periscope.laser.reflect_plane(self.periscope.mirror_down.triangle). \
                intersect_plane(self.periscope.mirror_up.triangle)
            p3_intersect = self.periscope.laser.reflect_plane(self.periscope.mirror_down.triangle). \
                reflect_plane(self.periscope.mirror_up.triangle).intersect_plane(self.periscope.mirror_3.triangle)
            p_aim = self.periscope.ray_to_aim().intersect_plane(
                Triangle(Point3d(tee.location.x, 0.5, 0.2),
                         Point3d(tee.location.x, 0.4, 0.1),
                         Point3d(tee.location.x, 0.3, 0.5)
                         ))

            ray_ret = None
            if mode == 'setup':
                ray_ret = self.backray(p3_intersect, p_aim, 0.01)

            self.renderer.render(p1_intersect, p2_intersect, p3_intersect, tee, p_aim, ray_ret)

            exit_app, need_rebuild = self.__move_target()

            if need_rebuild:
                self.down_plane_queue.put(self.periscope.target)
                self.up_plane_queue.put(self.periscope.target)
                self.plane_3_queue.put(self.periscope.target)

                self.update_log(iteration, p_aim)
                iteration += 1

            self.apply_changes(self.periscope.mirror_down.triangle, self.down_plane_points)
            self.apply_changes(self.periscope.mirror_up.triangle, self.up_plane_points)
            self.apply_changes(self.periscope.mirror_3.triangle, self.plane_3_points)
            # update log

        self.up_plane_process.terminate()
        self.down_plane_process.terminate()
        self.plane_3_process.terminate()
        self.write_log()
        exit()

    def update_log(self, iteration, p_aim):
        tee = self.periscope.target.location
        up = self.periscope.mirror_up.triangle
        down = self.periscope.mirror_down.triangle
        m3 = self.periscope.mirror_3.triangle

        output_iteration_list = []
        output_iteration_list.append('-------------iteration: ' + str(iteration) + '-------------\n')
        output_iteration_list.append(' '.join(['target: ', str(tee.x), str(tee.y), str(tee.z), '\n']))
        output_iteration_list.append(' '.join(['difference: ', str(p_aim.distance_to_point(tee)), '\n']))
        output_iteration_list.append(
            ' '.join(['up b: ', str(up.point_b.x), str(up.point_b.y), str(up.point_b.z), '\n']))
        output_iteration_list.append(
            ' '.join(['up c: ', str(up.point_c.x), str(up.point_c.y), str(up.point_c.z), '\n']))
        output_iteration_list.append(
            ' '.join(['down b: ', str(down.point_b.x), str(down.point_b.y), str(down.point_b.z), '\n']))
        output_iteration_list.append(
            ' '.join(['down c: ', str(down.point_c.x), str(down.point_c.y), str(down.point_c.z), '\n']))
        output_iteration_list.append(
            ' '.join(['3 b: ', str(m3.point_b.x), str(m3.point_b.y), str(m3.point_b.z), '\n']))
        output_iteration_list.append(
            ' '.join(['3 c: ', str(m3.point_c.x), str(m3.point_c.y), str(m3.point_c.z), '\n']))

        self.log_list.append(''.join(output_iteration_list))

    def write_log(self):
        time_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
        f = open(str(get_project_root()) + '\\logs\\' + time_str + '_log.txt', 'w')
        f.writelines(self.log_list)
        f.close()

    @staticmethod
    def apply_changes(plane: Triangle, arr: mp.Array):
        plane.point_b = Point3d(arr[0], arr[1], arr[2])
        plane.point_c = Point3d(arr[3], arr[4], arr[5])


if __name__ == '__main__':
    n = len(sys.argv)
    if n > 1:
        input_model = sys.argv[1]

    if n > 2 and sys.argv[2] == 'net':
        algorithm = SolveAlgorithm.NEURAL_NET

    if n > 3 and sys.argv[3] == 'setup':
        mode = 'setup'
    else:
        mode = ''

    input_model: str = 'my_conf'
    algorithm: SolveAlgorithm = SolveAlgorithm.NEURAL_NET
    app = PeriscopeApplication(input_model, algorithm, mode)
    app.run()
