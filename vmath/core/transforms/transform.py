from vmath.core import geometry_utils
from vmath.core.matrices import Mat4
from vmath.core.vectors import Vec3
import math


class Transform:

    __slots__ = "__transform_m"

    def __init__(self):
        self.__transform_m = Mat4(1.0, 0.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0, 0.0,
                                  0.0, 0.0, 0.0, 1.0)

    def __str__(self) -> str:
        return f"{{\n\t\"unique_id\"   :{self.unique_id},\n" \
                   f"\t\"origin\"      :{self.origin},\n" \
                   f"\t\"scale\"       :{self.scale},\n" \
                   f"\t\"rotate\"      :{self.angles / math.pi * 180},\n" \
                   f"\t\"transform_m\" :\n{self.__transform_m}\n}}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Transform):
            return False
        if not(self.__transform_m == other.__transform_m):
            return False
        return True

    def __hash__(self) -> int:
        return hash(self.__transform_m)

    def __build_basis(self, ex: Vec3, ey: Vec3, ez: Vec3) -> None:
        self.__transform_m.m00 = ex.x
        self.__transform_m.m10 = ex.y
        self.__transform_m.m20 = ex.z

        self.__transform_m.m01 = ey.x
        self.__transform_m.m11 = ey.y
        self.__transform_m.m21 = ey.z

        self.__transform_m.m02 = ez.x
        self.__transform_m.m12 = ez.y
        self.__transform_m.m22 = ez.z

        # self.eulerAngles = mathUtils.rot_m_to_euler_angles(self.rotation_mat())
    @property
    def unique_id(self) -> int:
        return id(self)

    @property
    def transform_matrix(self) -> Mat4:
        return self.__transform_m

    @transform_matrix.setter
    def transform_matrix(self, t: Mat4) -> None:
        self.__transform_m.m00 = t.m00
        self.__transform_m.m10 = t.m10
        self.__transform_m.m20 = t.m20

        self.__transform_m.m01 = t.m01
        self.__transform_m.m11 = t.m11
        self.__transform_m.m21 = t.m21

        self.__transform_m.m02 = t.m02
        self.__transform_m.m12 = t.m12
        self.__transform_m.m22 = t.m22

        self.__transform_m.m03 = t.m03
        self.__transform_m.m13 = t.m13
        self.__transform_m.m23 = t.m23

    @property
    def front(self) -> Vec3:
        return Vec3(self.__transform_m.m02,
                    self.__transform_m.m12,
                    self.__transform_m.m22).normalize()

    @front.setter
    def front(self, front_: Vec3) -> None:
        length_ = front_.magnitude
        if length_ < 1e-9:
            raise ArithmeticError("Error transform front set")
        front_dir_ = front_ / length_
        right_ = Vec3.cross(front_dir_, Vec3(0, 1, 0)).normalized()
        up_ = Vec3.cross(right_, front_dir_).normalized()
        self.__build_basis(right_ * self.sx, up_ * self.sy, front_)

    @property
    def up(self) -> Vec3:
        return Vec3(self.__transform_m.m01,
                    self.__transform_m.m11,
                    self.__transform_m.m21).normalize()

    @up.setter
    def up(self, up_: Vec3) -> None:
        length_ = up_.magnitude
        if length_ < 1e-9:
            raise ArithmeticError("Error transform up set")
        up_dir_ = up_ / length_
        front_ = Vec3.cross(up_dir_, Vec3(1, 0, 0)).normalized()
        right_ = Vec3.cross(up_dir_, front_).normalized()
        self.__build_basis(right_ * self.sx, up_, front_ * self.sz)

    @property
    def right(self) -> Vec3:
        return Vec3(self.__transform_m.m00,
                    self.__transform_m.m10,
                    self.__transform_m.m20).normalize()

    @right.setter
    def right(self, right_: Vec3) -> None:
        length_ = right_.magnitude
        if length_ < 1e-9:
            raise ArithmeticError("Error transform up set")
        right_dir_ = right_ / length_
        front_ = Vec3.cross(right_dir_, Vec3(0, 1, 0)).normalized()
        up_ = Vec3.cross(front_, right_dir_).normalized()
        self.__build_basis(right_, up_ * self.sy, front_ * self.sz)

    # масштаб по Х
    @property
    def sx(self) -> float:
        x = self.__transform_m.m00
        y = self.__transform_m.m10
        z = self.__transform_m.m20
        return math.sqrt(x * x + y * y + z * z)

    # масштаб по Y
    @property
    def sy(self) -> float:
        x = self.__transform_m.m01
        y = self.__transform_m.m11
        z = self.__transform_m.m21
        return math.sqrt(x * x + y * y + z * z)
        # масштаб по Z

    @property
    def sz(self) -> float:
        x = self.__transform_m.m02
        y = self.__transform_m.m12
        z = self.__transform_m.m22
        return math.sqrt(x * x + y * y + z * z)
        # установить масштаб по Х

    @sx.setter
    def sx(self, s_x: float) -> None:
        if s_x == 0:
            return
        scl = self.sx
        self.__transform_m.m00 *= s_x / scl
        self.__transform_m.m10 *= s_x / scl
        self.__transform_m.m20 *= s_x / scl

    # установить масштаб по Y
    @sy.setter
    def sy(self, s_y: float) -> None:
        if s_y == 0:
            return
        scl = self.sy
        self.__transform_m.m01 *= s_y / scl
        self.__transform_m.m11 *= s_y / scl
        self.__transform_m.m21 *= s_y / scl

    # установить масштаб по Z
    @sz.setter
    def sz(self, s_z: float) -> None:
        if s_z == 0:
            return
        scl = self.sz
        self.__transform_m.m02 *= s_z / scl
        self.__transform_m.m12 *= s_z / scl
        self.__transform_m.m22 *= s_z / scl

    @property
    def scale(self) -> Vec3:
        return Vec3(self.sx, self.sy, self.sz)

    @scale.setter
    def scale(self, xyz: Vec3):
        self.sx = xyz.x
        self.sy = xyz.y
        self.sz = xyz.z

    @property
    def x(self) -> float:
        return self.__transform_m.m03

    @property
    def y(self) -> float:
        return self.__transform_m.m13

    @property
    def z(self) -> float:
        return self.__transform_m.m23

    @x.setter
    def x(self, x: float) -> None:
        self.__transform_m.m03 = x

    @y.setter
    def y(self, y: float) -> None:
        self.__transform_m.m13 = y

    @z.setter
    def z(self, z: float) -> None:
        self.__transform_m.m23 = z

    @property
    def origin(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)

    @origin.setter
    def origin(self, xyz: Vec3) -> None:
        self.x = xyz.x
        self.y = xyz.y
        self.z = xyz.z

    @property
    def angles(self) -> Vec3:
        # TODO check rot_m_to_euler_angles
        return geometry_utils.rot_m_to_euler_angles(self.rotation_mat())

    @angles.setter
    def angles(self, xyz: Vec3) -> None:
        i = geometry_utils.rotate_x(xyz.x)
        i = geometry_utils.rotate_y(xyz.y) * i
        i = geometry_utils.rotate_z(xyz.z) * i
        scl = self.scale
        orig = self.origin
        self.__transform_m = i
        self.scale = scl
        self.origin = orig

    @property
    def ax(self) -> float:
        return geometry_utils.rot_m_to_euler_angles(self.rotation_mat()).x

    @property
    def ay(self) -> float:
        return geometry_utils.rot_m_to_euler_angles(self.rotation_mat()).y

    @property
    def az(self) -> float:
        return geometry_utils.rot_m_to_euler_angles(self.rotation_mat()).z

    @ax.setter
    def ax(self, x: float) -> None:
        _angles = self.angles
        self.angles = Vec3(geometry_utils.deg_to_rad(x), _angles.y, _angles.z)

    @ay.setter
    def ay(self, y: float) -> None:
        _angles = self.angles
        self.angles = Vec3(_angles.x, geometry_utils.deg_to_rad(y), _angles.z)

    @az.setter
    def az(self, z: float) -> None:
        _angles = self.angles
        self.angles = Vec3(_angles.x, _angles.y, geometry_utils.deg_to_rad(z))

    def rotation_mat(self) -> Mat4:
        scl = self.scale
        return Mat4(self.__transform_m.m00 / scl.x, self.__transform_m.m01 / scl.y, self.__transform_m.m02 / scl.z, 0,
                    self.__transform_m.m10 / scl.x, self.__transform_m.m11 / scl.y, self.__transform_m.m12 / scl.z, 0,
                    self.__transform_m.m20 / scl.x, self.__transform_m.m21 / scl.y, self.__transform_m.m22 / scl.z, 0,
                    0, 0, 0, 1)

    def look_at(self, target: Vec3, eye: Vec3, up: Vec3 = Vec3(0, 1, 0)) -> None:
        self.__transform_m = geometry_utils.look_at(target, eye, up)

    # переводит вектор в собственное пространство координат
    def transform_vect(self, vec: Vec3, w=1.0) -> Vec3:
        if w == 0:
            return Vec3(self.__transform_m.m00 * vec.x + self.__transform_m.m01 * vec.y + self.__transform_m.m02 * vec.z,
                        self.__transform_m.m10 * vec.x + self.__transform_m.m11 * vec.y + self.__transform_m.m12 * vec.z,
                        self.__transform_m.m20 * vec.x + self.__transform_m.m21 * vec.y + self.__transform_m.m22 * vec.z)

        return Vec3(
            self.__transform_m.m00 * vec.x + self.__transform_m.m01 * vec.y + self.__transform_m.m02 * vec.z + self.__transform_m.m03,
            self.__transform_m.m10 * vec.x + self.__transform_m.m11 * vec.y + self.__transform_m.m12 * vec.z + self.__transform_m.m13,
            self.__transform_m.m20 * vec.x + self.__transform_m.m21 * vec.y + self.__transform_m.m22 * vec.z + self.__transform_m.m23)

    # не переводит вектор в собственное пространство координат =)
    def inv_transform_vect(self, vec: Vec3, w=1.0) -> Vec3:
        scl: Vec3 = self.scale
        if w == 0:
            return Vec3((self.__transform_m.m00 * vec.x + self.__transform_m.m10 * vec.y + self.__transform_m.m20 * vec.z) / scl.x / scl.x,
                        (self.__transform_m.m01 * vec.x + self.__transform_m.m11 * vec.y + self.__transform_m.m21 * vec.z) / scl.y / scl.y,
                        (self.__transform_m.m02 * vec.x + self.__transform_m.m12 * vec.y + self.__transform_m.m22 * vec.z) / scl.z / scl.z)

        vec_ = Vec3(vec.x - self.x, vec.y - self.y, vec.z - self.z)
        return Vec3((self.__transform_m.m00 * vec_.x + self.__transform_m.m10 * vec_.y + self.__transform_m.m20 * vec_.z) / scl.x / scl.x,
                    (self.__transform_m.m01 * vec_.x + self.__transform_m.m11 * vec_.y + self.__transform_m.m21 * vec_.z) / scl.y / scl.y,
                    (self.__transform_m.m02 * vec_.x + self.__transform_m.m12 * vec_.y + self.__transform_m.m22 * vec_.z) / scl.z / scl.z)

