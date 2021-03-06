from src.parser import parse
from src.system.periscope import Periscope, Target, MirrorLocation
import random
from src.geometry import *
from src.render import Renderer, pygame
import math

class Generator:
    def __init__(self, conf: str):
        self.conf = conf
        config = parse(conf)
        self.periscope: Periscope = Periscope(config)

        p_target = self.periscope.ray_to_aim().intersect_plane(
            Triangle(Point3d(0.5, 0.6, 0.2),
                     Point3d(0.2, 0.5, 0.1),
                     Point3d(0.2, 0.30, 0.5)
                     ))
        # p_target = Point3d(0.01, 0.4, 0.2)
        tee = Target(p_target, 0.02)
        self.periscope.set_target(tee)

        plane_up = self.periscope.mirror_up.triangle
        plane_down = self.periscope.mirror_down.triangle
        plane_3 = self.periscope.mirror_3.triangle

        self.down_plane = Triangle(plane_down.point_a, plane_down.point_b, plane_down.point_c)
        self.up_plane = Triangle(plane_up.point_a, plane_up.point_b, plane_up.point_c)
        self.plane_3 = Triangle(plane_3.point_a, plane_3.point_b, plane_3.point_c)

        self.base_length = {
            'up': Vector(plane_up.point_c, plane_up.point_b).length(),
            'down': Vector(plane_down.point_c, plane_down.point_b).length(),
            '3': Vector(plane_3.point_c, plane_3.point_b).length()
        }

        self.height_length = {
            'up':  Vector(plane_up.point_a, (plane_up.point_b + plane_up.point_c) / 2).length(),
            'down':  Vector(plane_down.point_a, (plane_down.point_b + plane_down.point_c) / 2).length(),
            '3': Vector(plane_3.point_a, (plane_3.point_b + plane_3.point_c) / 2).length()
        }

        self.side_length = {
            'up':  Vector(plane_up.point_a, plane_up.point_b).length(),
            'down': Vector(plane_down.point_a, plane_down.point_b).length(),
            '3': Vector(plane_3.point_a, plane_3.point_b).length(),
        }



    def generate(self, step = 0):
        max_pitch = 0.1
        max_roll = 0.3
        pitch = random.random() * max_pitch - max_pitch / 2
        roll = random.random() * max_roll - max_roll / 2
        # pitch = 0.0055
        # roll = 0.07673
        print(pitch)
        print(roll)
        if step % 2 == 0:
            self.periscope.mirror_down.triangle = self.down_plane.rotate_plane(roll, Angle.ROLL)
            self.periscope.mirror_down.triangle = self.periscope.mirror_down.triangle.rotate_plane(pitch, Angle.PITCH)
        else:
            self.periscope.mirror_down.triangle = self.down_plane.rotate_plane(pitch, Angle.PITCH)
            self.periscope.mirror_down.triangle = self.periscope.mirror_down.triangle.rotate_plane(roll, Angle.ROLL)

        #build plane up
        point_a_up = self.periscope.mirror_up.triangle.point_a
        n = self.periscope.mirror_down.triangle.n

        d = -(point_a_up.x * n.x + point_a_up.y * n.y + point_a_up.z * n.z)
        up_plane = Plane(n.x, n.y, n.z, d)

        # find projection
        m_p = up_plane.point_projection(self.periscope.laser.intersect_plane(self.periscope.mirror_down.triangle))

        up_x: Vector = Vector(m_p, point_a_up)
        up_y = up_x.vector_prod(n)

        point_b_up = point_a_up + (up_x / up_x.length() * self.height_length['up']) + \
                     up_y / up_y.length() * (self.base_length['up'] / 2)

        point_c_up = point_a_up + (up_x / up_x.length() * self.height_length['up']) - \
                     up_y / up_y.length() * (self.base_length['up'] / 2)


        self.periscope.mirror_up.triangle = Triangle(point_a_up, point_b_up, point_c_up)

        # build plane 3rd
        point_a_3 = self.periscope.mirror_3.triangle.point_a
        n = self.periscope.mirror_up.triangle.n

        d = -(point_a_3.x * n.x + point_a_3.y * n.y + point_a_3.z * n.z)
        plane_3 = Plane(n.x, n.y, n.z, d)

        # find projection
        m_p = plane_3.point_projection(self.periscope.laser.reflect_plane(self.periscope.mirror_down.triangle).intersect_plane(self.periscope.mirror_up.triangle))

        up_x: Vector = Vector(m_p, point_a_3)
        up_y = up_x.vector_prod(n)

        point_b_3 = point_a_3 + (up_x / up_x.length() * self.height_length['3']) + \
                     up_y / up_y.length() * (self.base_length['3'] / 2)

        point_c_3 = point_a_3 + (up_x / up_x.length() * self.height_length['3']) - \
                     up_y / up_y.length() * (self.base_length['3'] / 2)

        self.periscope.mirror_3.triangle = Triangle(point_a_3, point_b_3, point_c_3)
        return


    def __find_up_rotate_angels(self, loc: MirrorLocation):
        if loc == MirrorLocation.UP:
            original_plane = self.periscope.mirror_up.triangle
            rotated_plane = self.up_plane
            str_loc = 'up'
            coef = 1
        elif loc == MirrorLocation.THIRD:
            original_plane = self.periscope.mirror_3.triangle
            rotated_plane = self.plane_3
            str_loc = '3'
            coef = -1
        else:
            original_plane = self.periscope.mirror_down.triangle
            rotated_plane = self.down_plane
            str_loc = 'down'
            coef = 1

        # 'm' point is (c + b) / 2 - middle of the triangle base
        m: Point3d = (rotated_plane.point_b + rotated_plane.point_c) / 2

        # 'm' after rotations
        m_r3: Point3d = (original_plane.point_b + original_plane.point_c) / 2

        # m_r1: 'm' after one (yaw) rotation
        m_r1_proj = rotated_plane.point_projection(m_r3)
        v_mr1 = Vector(m_r1_proj, rotated_plane.point_a)
        v_mr1.normalize()
        m_r1 = rotated_plane.point_a +  v_mr1 * self.height_length[str_loc]

        m_mr1_middle: Point3d = (m + m_r1) / 2
        mr1_mr3_middle:  Point3d = (m_r1 + m_r3) / 2

        yaw = 2 * math.asin(m_mr1_middle.distance_to_point(m) / self.height_length[str_loc])
        if yaw > 1e-9:
            n = Vector(m, rotated_plane.point_a).vector_prod(Vector(m_r1, rotated_plane.point_a))
            n *= coef
            n.normalize()
            if (n + rotated_plane.get_axe('y')).length() < 0.1:  # actually it might be 0 or 2
                yaw *= -1


        pitch = 2 * math.asin(mr1_mr3_middle.distance_to_point(m_r3) / self.height_length[str_loc])
        if pitch > 1e-9:
            n = Vector(m_r3, rotated_plane.point_a).vector_prod(Vector(m_r1, rotated_plane.point_a))
            # n *= coef
            n.normalize()
            if (n + rotated_plane.get_axe('z')).length() > 0.1:  # actually it might be 0 or 2
                pitch *= -1

        r1_plane = rotated_plane.rotate_plane(yaw, Angle.YAW)
        r2_plane = r1_plane.rotate_plane(pitch, Angle.PITCH)

        b_proj: Point3d = original_plane.point_projection(r2_plane.point_b)
        roll = math.asin(b_proj.distance_to_point(r2_plane.point_b) / (self.base_length[str_loc] / 2))
        if roll > 1e-9:
            n = Vector(m_r3, r2_plane.point_b).vector_prod(Vector(m_r3, original_plane.point_b))
            n *= coef
            n.normalize()
            if (n + original_plane.get_axe('x')).length() > 0.1:  # actually it might be 0 or 2
                roll *= -1

        if loc == MirrorLocation.DOWN:
            roll = -roll
        else:
            roll = -roll
        return yaw, pitch, roll


    def render_shell(self):
        self.generate()

        periscope = self.periscope
        tee = periscope.target
        step = 0
        renderer = Renderer(periscope)
        while True:
            mirror_down = periscope.mirror_down
            mirror_up = periscope.mirror_up
            mirror_3 = periscope.mirror_3

            p1_intersect = periscope.laser.intersect_plane(mirror_down.triangle)
            p2_intersect = periscope.laser.reflect_plane(mirror_down.triangle).intersect_plane(mirror_up.triangle)
            p3_intersect = periscope.laser.reflect_plane(mirror_down.triangle).reflect_plane(mirror_up.triangle).intersect_plane(mirror_3.triangle)

            p_aim = periscope.ray_to_aim().intersect_plane(
                Triangle(Point3d(tee.location.x, 0.5, 0.2),
                         Point3d(tee.location.x, 0.4, 0.1),
                         Point3d(tee.location.x, 0.3, 0.5)
                         ))


            renderer.render(p1_intersect, p2_intersect, p3_intersect, tee, p_aim)

            pygame.time.delay(100)
            for i in pygame.event.get():
                if i.type == pygame.QUIT:
                    exit()
                elif i.type == pygame.KEYDOWN:
                    print(p_aim.y, p_aim.z)
                    step += 1
                    self.generate(step)
                    self.__find_up_rotate_angels(MirrorLocation.UP)
                    self.__find_up_rotate_angels(MirrorLocation.DOWN)
                    self.__find_up_rotate_angels(MirrorLocation.THIRD)
                    print('aim: ', p_aim.x, p_aim.y, p_aim.z)

    @staticmethod
    def add_angles_to_list(str_list: [], angles):
        str_list.append(str(angles[0]))
        str_list.append(str(angles[1]))
        str_list.append(str(angles[2]))

    def writer_shell(self):
        output_list = []
        #
        # renderer = Renderer(self.periscope)
        # tee = self.periscope.target
        #
        for i in range(10000):

            self.generate(i)

            p_aim = self.periscope.ray_to_aim().intersect_plane(
                Triangle(Point3d(self.periscope.target.location.x, 0.5, 0.2),
                         Point3d(self.periscope.target.location.x, 0.4, 0.1),
                         Point3d(self.periscope.target.location.x, 0.3, 0.5)
                         ))
            #jnc.lf

            # mirror_down = self.periscope.mirror_down
            # mirror_up = self.periscope.mirror_up
            # mirror_3 = self.periscope.mirror_3
            #
            # p1_intersect = self.periscope.laser.intersect_plane(mirror_down.triangle)
            # p2_intersect = self.periscope.laser.reflect_plane(mirror_down.triangle).intersect_plane(mirror_up.triangle)
            # p3_intersect = self.periscope.laser.reflect_plane(mirror_down.triangle).reflect_plane(
            #     mirror_up.triangle).intersect_plane(mirror_3.triangle)
            #
            #
            # renderer.render(p1_intersect, p2_intersect, p3_intersect, tee, p_aim)
            # jnc.lf

            up_angles = self.__find_up_rotate_angels(MirrorLocation.UP)
            down_angles = self.__find_up_rotate_angels(MirrorLocation.DOWN)
            angles_3 = self.__find_up_rotate_angels(MirrorLocation.THIRD)

            str_list = [str(p_aim.y), str(p_aim.z)]

            self.add_angles_to_list(str_list, down_angles)
            self.add_angles_to_list(str_list, up_angles)
            self.add_angles_to_list(str_list, angles_3)
            output_list.append('\n')
            output_list.append(','.join(str_list))


        f = open('./neuralnet/' + self.conf + '_data_set.csv', 'w')
        f.writelines(output_list)
        f.close()

    def test_shell(self):
        self.periscope.mirror_up.triangle = self.periscope.mirror_up.triangle.rotate_plane(0.2, Angle.ROLL)
        self.__find_up_rotate_angels(MirrorLocation.UP)


if __name__ == '__main__':
    gen = Generator('my_conf')
    # gen.render_shell()
    gen.writer_shell()
    gen.test_shell()