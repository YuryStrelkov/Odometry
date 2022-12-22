from Odometry.devices.sensors_utils.real_time_filter import RealTimeFilter
from Odometry.vmath.core.vectors import Vec3
import matplotlib.pyplot as plt
from typing import List, Tuple
import dataclasses
import numpy as np
import json


@dataclasses.dataclass
class WayPoint:
    acceleration: Vec3
    velocity: Vec3
    position: Vec3
    orientation: Vec3
    time_value: float
    time_delta: float


@dataclasses.dataclass
class AccelerometerLog:
    device_name: str
    log_time_start: str
    way_points: List[WayPoint]


def read_accelerometer_log(path_2_json: str) -> AccelerometerLog:
    with open(path_2_json) as input_json:
        raw_json = json.load(input_json)
        way_points = [] * len(raw_json['way_points'])
        v = Vec3(0.0)
        p = Vec3(0.0)

        _filter_x = RealTimeFilter()
        _filter_x.mode = 2
        _filter_y = RealTimeFilter()
        _filter_y.mode = 2
        _filter_z = RealTimeFilter()
        _filter_z.mode = 2

        _filter_o_x = RealTimeFilter()
        _filter_o_x.mode = 2
        _filter_o_y = RealTimeFilter()
        _filter_o_y.mode = 2
        _filter_o_z = RealTimeFilter()
        _filter_o_z.mode = 2

        for item in raw_json['way_points']:
            a = Vec3(_filter_x.filter(float(item["acceleration"]["x"])),
                     _filter_y.filter(float(item["acceleration"]["y"]) - 9.80665),
                     _filter_z.filter(float(item["acceleration"]["z"])))
            o = Vec3(_filter_o_x.filter(float(item["orientation"]["x"])),
                     _filter_o_y.filter(float(item["orientation"]["y"])),
                     _filter_o_z.filter(float(item["orientation"]["z"])))
            v = Vec3(_filter_o_x.filter(float(item["velocity"]["x"])),
                     _filter_o_y.filter(float(item["velocity"]["y"])),
                     _filter_o_z.filter(float(item["velocity"]["z"])))
            p = Vec3(_filter_o_x.filter(float(item["position"]["x"])),
                     _filter_o_y.filter(float(item["position"]["y"])),
                     _filter_o_z.filter(float(item["position"]["z"])))
            t = float(item["curr_time"])

            dt = float(item["delta_time"])

            if len(way_points) == 0:
                way_points.append(WayPoint(a, Vec3(0.0), Vec3(0.0), o, t, dt))
                continue

            way_points.append(WayPoint(a, v, p, o, t, dt))

        return AccelerometerLog(raw_json["device_name"], raw_json["log_time_start"], way_points)


def accelerations(way_points: List[WayPoint]) -> Tuple[List[float], List[float], List[float]]:
    a_0_x = -0.09122
    a_0_y = -0.1166
    a_0_z = 0.511
    return [v.acceleration.x + a_0_x for v in way_points],\
           [v.acceleration.y + a_0_y for v in way_points],\
           [v.acceleration.z + a_0_z for v in way_points]


def integrate(ax: List[float], ay: List[float], az: List[float], dt: List[float]) ->\
        Tuple[List[float], List[float], List[float],
              List[float], List[float], List[float]]:

    v_0 = 0.0
    vx = []
    for _dt, _ax in zip(dt, ax):
        v_0 += _dt * _ax
        vx.append(v_0)

    v_0 = 0.0
    vy = []
    for _dt, _ay in zip(dt, ay):
        v_0 += _dt * _ay
        vy.append(v_0)

    v_0 = 0.0
    vz = []
    for _dt, _az in zip(dt, az):
        v_0 += _dt * _az
        vz.append(v_0)
    ##################################
    ##################################
    ##################################
    s_0 = 0.0
    sx = []
    for _dt, _vx in zip(dt, vx):
        s_0 += _dt * _vx
        sx.append(s_0)

    s_0 = 0.0
    sy = []
    for _dt, _vy in zip(dt, vy):
        s_0 += _dt * _vy
        sy.append(s_0)

    s_0 = 0.0
    sz = []
    for _dt, _vz in zip(dt, vz):
        s_0 += _dt * _vz
        sz.append(s_0)

    return vx, vy, vz, sx, sy, sz


def draw_acceleration(log_info: AccelerometerLog):
    print(f"way points number: {len(log_info.way_points)}")

    _x = [v.acceleration.x for v in log_info.way_points]
    _y = [v.acceleration.y for v in log_info.way_points]
    _z = [v.acceleration.z for v in log_info.way_points]

    print(f"x:[{np.amin(_x):.12}, {np.amax(_x):.12}]")
    print(f"y:[{np.amin(_y):.12}, {np.amax(_y):.12}]")
    print(f"z:[{np.amin(_z):.12}, {np.amax(_z):.12}]")

    fig = plt.figure()

    ax = plt.axes(projection='3d')

    ax.plot3D(_x, _y, _z, 'r')

    plt.show()


def draw_acceleration_2d(log_info: AccelerometerLog):
    print(f"way points number: {len(log_info.way_points)}")

    ax, ay, az = accelerations(log_info.way_points)

    dt = [v.time_delta for v in log_info.way_points]

    t_0 = 0.0
    time_values = []
    for _dt in dt:
        time_values.append(t_0)
        t_0 += _dt

    vx, vy, vz, sx, sy, sz = integrate(ax, ay, az, dt)

    fig_1 = plt.figure()

    axes = plt.axes()  # (projection='3d')

    axes.plot(time_values, ax, 'r')
    axes.plot(time_values, ay, 'g')
    axes.plot(time_values, az, 'b')
    axes.legend(['$a_{x}$', '$a_{y}$', '$a_{z}$'])
    axes.set_xlabel("$x,[sec]$")
    axes.set_ylabel("$a(t),[m/sec^{2}]$")
    plt.show()

    fig_1 = plt.figure()

    axes = plt.axes() # (projection='3d')

    # axes.plot3D(ax, ay, az, 'r')
    axes.plot(time_values, vx, 'r')
    axes.plot(time_values, vy, 'g')
    axes.plot(time_values, vz, 'b')
    axes.legend(['$v_{x}$', '$v_{y}$', '$v_{z}$'])
    axes.set_xlabel("$x,[sec]$")
    axes.set_ylabel("$v(t),[m/sec]$")
    plt.show()

    fig_1 = plt.figure()

    axes = plt.axes()
    axes.plot(time_values, sx, 'r')
    axes.plot(time_values, sy, 'g')
    axes.plot(time_values, sz, 'b')
    axes.legend(['$s_{x}$', '$s_{y}$', '$s_{z}$'])
    axes.set_xlabel("$x,[sec]$")
    axes.set_ylabel("$S(t),[m]$")
    plt.show()

    fig_1 = plt.figure()
    axes = plt.axes()
    axes.plot(sx, sz, 'r')
    axes.plot(sx, sy, 'g')
    axes.plot(sy, sz, 'b')
    axes.legend(['$S_{x,z}$', '$S_{x,y}$', '$S_{y,z}$'])
    axes.set_ylabel("$x, [m]$")
    axes.set_ylabel("$y, [m]$")
    plt.show()

if __name__ == "__main__":
    log = read_accelerometer_log("accelerometer_log_coridor.json")
    draw_acceleration_2d(log)
    # draw_acceleration(log)

