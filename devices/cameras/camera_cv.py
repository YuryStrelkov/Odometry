import datetime as datetime
from os import mkdir, listdir, path, makedirs
import numpy as np
import cv2 as cv
import time


class CameraCV:  # (Camera):

    def __init__(self):
        self.__camera_stream: cv.VideoCapture
        self.__buffer_depth = 8
        self.__buffer: [np.ndarray] = []
        try:
            self.__camera_stream = cv.VideoCapture(0, cv.CAP_DSHOW)
            self.__camera_stream.set(cv.CAP_PROP_FPS, 60)
        except RuntimeError("CV Camera instantiate error") as ex:
            print(ex.args)

        if not self.__camera_stream.isOpened():
            raise RuntimeError("device init function call error")
        # super().__init__()

    @property
    def next_frame(self) -> np.ndarray:
        if not self.is_open:
            raise RuntimeError("Can't receive frame (stream end?). Exiting ...")

        has_frame, cam_frame = self.__camera_stream.read()

        if not has_frame:
            raise RuntimeError("Can't receive frame (stream end?). Exiting ...")
        self.__append_frames_buffer(cam_frame)
        return cam_frame

    def __del__(self):
        try:
            self.__camera_stream.release()
        except RuntimeError() as ex:
            print(f"dispose error {ex.args}")
            return False
        return True

    def __str__(self):
        res: str = "CV Camera: \n" \
                   f"CV_CAP_PROP_FRAME_WIDTH  : {self.__camera_stream.get(cv.CAP_PROP_FRAME_WIDTH)}\n" \
                   f"CV_CAP_PROP_FRAME_HEIGHT : {self.__camera_stream.get(cv.CAP_PROP_FRAME_HEIGHT)}\n" \
                   f"CAP_PROP_FPS             : {self.__camera_stream.get(cv.CAP_PROP_FPS)}\n" \
                   f"CAP_PROP_EXPOSUREPROGRAM : {self.__camera_stream.get(cv.CAP_PROP_EXPOSUREPROGRAM)}\n" \
                   f"CAP_PROP_POS_MSEC        : {self.__camera_stream.get(cv.CAP_PROP_POS_MSEC)}\n" \
                   f"CAP_PROP_FRAME_COUNT     : {self.__camera_stream.get(cv.CAP_PROP_FRAME_COUNT)}\n" \
                   f"CAP_PROP_BRIGHTNESS      : {self.__camera_stream.get(cv.CAP_PROP_BRIGHTNESS)}\n" \
                   f"CAP_PROP_CONTRAST        : {self.__camera_stream.get(cv.CAP_PROP_CONTRAST)}\n" \
                   f"CAP_PROP_SATURATION      : {self.__camera_stream.get(cv.CAP_PROP_SATURATION)}\n" \
                   f"CAP_PROP_HUE             : {self.__camera_stream.get(cv.CAP_PROP_HUE)}\n" \
                   f"CAP_PROP_GAIN            : {self.__camera_stream.get(cv.CAP_PROP_GAIN)}\n" \
                   f"CAP_PROP_CONVERT_RGB     : {self.__camera_stream.get(cv.CAP_PROP_CONVERT_RGB)}\n"
        return res

    __repr__ = __str__

    def __append_frames_buffer(self, frame: np.ndarray):
        if len(self.__buffer) == self.__buffer_depth:
            del self.__buffer[0]
        self.__buffer.append(frame)

    @property
    def is_buffer_empty(self) -> bool:
        return len(self.__buffer) == 0

    @property
    def is_open(self) -> bool:
        return self.__camera_stream.isOpened()

    @property
    def pixel_format(self) -> str:
        return self.__camera_stream.get(cv.CAP_PROP_FORMAT)

    @pixel_format.setter
    def pixel_format(self, pixel_format: str) -> None:
        raise RuntimeError("pixel format setter is unsupported for this camera")

    @property
    def camera_width(self) -> int:
        return int(self.__camera_stream.get(cv.CAP_PROP_FRAME_WIDTH))

    @camera_width.setter
    def camera_width(self, w: int) -> None:
        if not self.__camera_stream.set(cv.CAP_PROP_FRAME_WIDTH, w):
            print(f"incorrect devices width {w}")
            return
        self.aspect = self.camera_width / float(self.camera_height)

    @property
    def camera_height(self) -> int:
        return int(self.__camera_stream.get(cv.CAP_PROP_FRAME_HEIGHT))

    @camera_height.setter
    def camera_height(self, h: int) -> None:
        if not self.__camera_stream.set(cv.CAP_PROP_FRAME_HEIGHT, h):
            print(f"incorrect devices height {h}")
            return
        self.aspect = self.camera_width / float(self.camera_height)

    @property
    def offset_x(self) -> int:
        return int(self.__camera_stream.get(cv.CAP_PROP_XI_OFFSET_X))

    @offset_x.setter
    def offset_x(self, value) -> None:
        if not self.__camera_stream.set(cv.CAP_PROP_XI_OFFSET_X,
                                        min(max(-self.camera_width, value), self.camera_width)):
            print(f"incorrect devices x - offset {value}")

    @property
    def offset_y(self) -> int:
        return int(self.__camera_stream.get(cv.CAP_PROP_XI_OFFSET_Y))

    @offset_y.setter
    def offset_y(self, value) -> None:
        if not self.__camera_stream.set(cv.CAP_PROP_XI_OFFSET_Y,
                                        min(max(-self.camera_height, value), self.camera_height)):
            print(f"incorrect devices y - offset {value}")

    @property
    def exposure_mode(self) -> str:
        return str(self.__camera_stream.get(cv.CAP_PROP_EXPOSUREPROGRAM))

    @exposure_mode.setter
    def exposure_mode(self, value: str):
        raise RuntimeError("exposure mode setter is unsupported for this camera")

    @property
    def exposure(self) -> float:
        return float(self.__camera_stream.get(cv.CAP_PROP_EXPOSURE))

    @exposure.setter
    def exposure(self, value) -> None:
        if not self.__camera_stream.set(cv.CAP_PROP_EXPOSURE, min(max(-12, value), 12)):
            print(f"incorrect devices y - offset {value}")

    @property
    def frame_time(self) -> float:
        return 1.0 / float(self.__camera_stream.get(cv.CAP_PROP_FPS))

    @property
    def camera_fps(self) -> int:
        return self.__camera_stream.get(cv.CAP_PROP_FPS)

    @camera_fps.setter
    def camera_fps(self, fps: int) -> None:
        if not self.__camera_stream.set(cv.CAP_PROP_FPS, min(max(1, fps), 120)):
            print(f"incorrect devices fps {fps}")

    @property
    def last_frame(self) -> np.ndarray:
        if len(self.__buffer) == 0:
            return np.zeros((self.camera_width, self.camera_height), dtype=np.uint8)
        return self.__buffer[len(self.__buffer) - 1]

    def __keyboard_input(self) -> bool:
        key = cv.waitKey(2)
        if key == 27:
            # cv.destroyWindow("video")
            return False

        if key == ord('s') or key == 251:
            try:
                now = datetime.datetime.now()
                cv.imwrite(f'frame_at_time_{now.hour}_{now.minute}_{now.second}_{now.microsecond}.png', self.last_frame)
                return True
            except RuntimeError as ex:
                print(f"{ex.args}")
                return True
        if key == ord('e') or key == 243:
            return True
        if key == ord('p'):
            cv.waitKey(-1)
        if key == ord('k'):
            print("Cameras properties:")
            print(f"\tframe time: {self.frame_time}")
            print(f"\tFPS: {self.camera_fps}")
            print(f"\twidth: {self.camera_width}")
            print(f"\theight: {self.camera_height}")
            print(f"\texposure: {self.exposure}")
            print(f"\texposure mode: {self.exposure_mode}")
            print(f"\toffset_x: {self.offset_x}")
            print(f"\toffset_y: {self.offset_y}")
            print(f"\tpixel format: {self.pixel_format}")
            print('\n')
        if key == ord('+'):
            self.camera_fps = self.camera_fps + 5
            print(f"New FPS: {self.camera_fps}")
        if key == ord('-'):
            self.camera_fps = self.camera_fps - 5
            print(f"New FPS: {self.camera_fps}")
        if key == ord('>'):
            self.exposure = self.exposure + 1
            print(f"New exposure: {self.exposure}")
        if key == ord('<'):
            self.exposure = self.exposure - 1
            print(f"New exposure: {self.exposure}")
        if key == ord('['):
            self.offset_x = self.offset_x - 1
            print(f"New offset_x: {self.offset_x}")
        if key == ord(']'):
            self.offset_x = self.offset_x + 1
            print(f"New offset_x: {self.offset_x}")
        if key == ord('l'):
            self.offset_y = self.offset_y - 1
            print(f"New offset_y: {self.offset_y}")
        if key == ord(';'):
            self.offset_y = self.offset_y + 1
            print(f"New offset_y: {self.offset_y}")

        return True

    def show_video(self):
        cv.namedWindow("video", cv.WINDOW_NORMAL)
        dt: float
        t0: float
        # th.setDaemon(True)
        while True:
            if not self.is_open:
                cv.destroyWindow("video")
                break
            if not self.__keyboard_input():
                cv.destroyWindow("video")
                break
            try:
                t0 = time.time()
                frame = self.next_frame
                dt = time.time() - t0
                self.put_cam_param_on_frame(frame)
                cv.imshow("video", frame)

                if dt > self.frame_time:
                    continue
                time.sleep(self.frame_time - dt)
            except RuntimeError as ex:
                print(f"{ex.args}")

    def record_frames(self, folder: str = None):
        while True:
            if folder is None:
                folder = f"camera record {datetime.datetime.now().strftime('%H; %M; %S')}"
                break
            if len(folder) == 0:
                folder = f"camera record {datetime.datetime.now().strftime('%H; %M; %S')}"
                break
            break
        try:
            mkdir(folder)
        except FileExistsError:
            print(f"directory {folder} exists already")

        cv.namedWindow("video", cv.WINDOW_NORMAL)
        dt: float
        t0: float
        while True:
            if not self.is_open:
                cv.destroyWindow("video")
                break
            # esc to exit
            if not self.__keyboard_input():
                cv.destroyWindow("video")
                break
            try:
                t0 = time.time()
                cv.imshow("video", self.next_frame)
                now = datetime.datetime.now()
                cv.imwrite(folder + f'/frame_at_time_{now.hour}_{now.minute}_{now.second}_{now.microsecond}.png',
                           self.last_frame)
                dt = time.time() - t0
                if dt > self.frame_time:
                    continue
                time.sleep(self.frame_time - dt)
            except RuntimeError as ex:
                print(f"{ex.args}")

    def put_cam_param_on_frame(self, frame):
        font = cv.FONT_HERSHEY_SIMPLEX
        line_type = cv.LINE_AA
        thickness = 2
        color = (244, 192, 85)
        cv.putText(frame, 'play / stop: "p"', (5, 20), font, 0.6, color, thickness, line_type, False)
        cv.putText(frame, 'save frame: "s"', (5, 40), font, 0.6, color, thickness, line_type, False)
        cv.putText(frame, f'FPS: {self.camera_fps}', (5, 60), font, 0.6, color, thickness, line_type, False)
        cv.putText(frame, f'exposure: {self.exposure}', (5, 80), font, 0.6, color, thickness, line_type, False)
        cv.putText(frame, f'exposure mode: {self.exposure_mode}', (5, 100), font, 0.6, color, thickness, line_type,
                   False)
        cv.putText(frame, f'offset_x: {self.offset_x}', (5, 120), font, 0.6, color, thickness, line_type, False)
        cv.putText(frame, f'offset_y: {self.offset_y}', (5, 140), font, 0.6, color, thickness, line_type, False)
        cv.putText(frame, f'pixel_format: {self.pixel_format}', (5, 160), font, 0.6, color, thickness, line_type, False)
        cv.putText(frame, f'height: {self.camera_height}', (5, 180), font, 0.6, color, thickness, line_type, False)
        cv.putText(frame, f'width: {self.camera_width}', (5, 200), font, 0.6, color, thickness, line_type, False)

    @staticmethod
    def monocular_camera_calibration(chess_board_img_dir: str = None, chess_board_size: tuple = None):
        # getting all the images names
        img_names = listdir(chess_board_img_dir)
        if len(img_names) == 0:
            raise RuntimeError("chess board images folder are empty!")
        if chess_board_size is None:
            raise RuntimeError("chess board suze is None!")

        # array that contains abs paths of all the images
        images = [path.join(chess_board_img_dir, img_name) for img_name in img_names]
        # getting images width and height
        img = cv.imread(images[0])
        frame_size = (img.shape[1], img.shape[0])
        print(f'{chess_board_size=}')
        print(f'{frame_size=}')

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
        obj_points = np.zeros((chess_board_size[0] * chess_board_size[1], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:chess_board_size[0], 0:chess_board_size[1]].T.reshape(-1, 2)
        # arrays to store object points and image points from all the images
        obj_points_arr = []
        img_points_arr = []

        #### FIND CORNERS ####

        for image in images:
            print(image)
            img = cv.imread(image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, chess_board_size, None)
            # if found, add objects points, image points (after refining them)
            if ret is True:
                obj_points_arr.append(obj_points)
                corners_2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points_arr.append(corners)
                # draw adn displat the corners
                cv.drawChessboardCorners(img, chess_board_size, corners_2, ret)
                cv.imshow('img', img)
                cv.waitKey(1000)
        cv.destroyAllWindows()

        #### CALIBRATION ####

        ret, camera_matrix, dist, r_vecs, t_vecs = cv.calibrateCamera(obj_points_arr, img_points_arr, frame_size,
                                                                      None, None)
        print("Camera Calibrated: ", ret)
        print("\nCamera Matrix:\n", camera_matrix)
        print("\nDistortion Parameters:\n", dist)
        print("\nRotation Vectors:\n", r_vecs)
        print("\nTranslation Vectors:\n", t_vecs)

        #### UNDISTORTION ####

        img = cv.imread(images[0])
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv.undistort(img, camera_matrix, dist, None, new_camera_matrix)
        # crop the image
        x, y, w, h = roi
        dst = dst[y: y + h, x: x + w]
        cv.imwrite(path.join(chess_board_img_dir, 'undistorted_img.png'), dst)

        # undistort with remapping
        map_x, map_y = cv.initUndistortRectifyMap(camera_matrix, dist, None, new_camera_matrix, (w, h), 5)
        dst = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        dst = dst[y: y + h, x: x + w]
        cv.imwrite(path.join(chess_board_img_dir, 'undistorted_img_with_remapping.png'), dst)

        # Reprojection Error
        mean_error = 0.0
        for i in range(len(obj_points_arr)):
            img_points_arr_2, _ = cv.projectPoints(obj_points_arr[i], r_vecs[i], t_vecs[i], camera_matrix, dist)
            error = cv.norm(img_points_arr[i], img_points_arr_2, cv.NORM_L2) / len(img_points_arr_2)
            mean_error += error
        mean_error /= len(obj_points_arr)
        print(f"\ntotal error: {mean_error}")
        print('\n\n')

    @staticmethod
    def get_stereo_images_for_calibration(stereo_images_dir: str = None):
        stereo_images_dir = stereo_images_dir if stereo_images_dir is not None else "./binocular_calibration"
        image_left_dir = path.join(stereo_images_dir, "stereo_left/image_left")
        image_right_dir = path.join(stereo_images_dir, "stereo_right/image_right")
        try:
            for directory in [stereo_images_dir, image_left_dir, image_right_dir]:
                if not path.exists(directory):
                    makedirs(directory)

            cap1 = cv.VideoCapture(0)
            cap2 = cv.VideoCapture(1)

            num = 0
            while cap1.isOpened() and cap2.isOpened():
                ret1, img1 = cap1.read()
                ret2, img2 = cap2.read()

                k = cv.waitKey(5)
                if k == 27:
                    break
                elif k == ord('s'):
                    cv.imwrite(path.join(image_left_dir, f"{num}.png"), img1)
                    cv.imwrite(path.join(image_right_dir, f"{num}.png"), img2)
                    print('images saved!')
                    num += 1
                cv.imshow('Img 1', img1)
                cv.imshow('Img 2', img2)
        except Exception as ex:
            print(ex)
            exit(-1)

    @staticmethod
    def binocular_calibration(stereo_images_dir: str = None, chess_board_size: tuple = None):
        stereo_images_dir = stereo_images_dir if stereo_images_dir is not None else "./binocular_calibration"
        image_left_dir = path.join(stereo_images_dir, "stereo_left/image_left")
        image_right_dir = path.join(stereo_images_dir, "stereo_right/image_right")

        for directory in [stereo_images_dir, image_left_dir, image_right_dir]:
            if not path.exists(directory):
                raise RuntimeError(f"{directory} doesn't exist!")

        if chess_board_size is None:
            raise RuntimeError(f"chess board size is None!")

        # getting all the images names
        img_left_names = listdir(image_left_dir)
        img_right_names = listdir(image_right_dir)
        # array that contains abs paths of all the images
        images_left = [path.join(image_left_dir, img_name) for img_name in img_left_names]
        images_right = [path.join(image_right_dir, img_name) for img_name in img_right_names]

        # getting images width and height
        img_l = cv.imread(images_left[0])
        frame_size_l = (img_l.shape[1], img_l.shape[0])
        img_r = cv.imread(images_right[0])
        frame_size_r = (img_r.shape[1], img_r.shape[0])
        if frame_size_l != frame_size_r:
            raise RuntimeError(f"size of images are not the same!")
        frame_size = frame_size_l

        print(f'{chess_board_size=}')
        print(f'{frame_size=}')

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
        obj_points = np.zeros((chess_board_size[0] * chess_board_size[1], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:chess_board_size[0], 0:chess_board_size[1]].T.reshape(-1, 2)

        # arrays to store object points and image points from all the images
        obj_points_arr = []  # 3d point in real world space
        img_points_arr_l = []  # 2d point in image plane
        img_points_arr_r = []  # 2d point in image plane

        for img_left, img_right in zip(images_left, images_right):
            img_l = cv.imread(img_left)
            img_r = cv.imread(img_right)
            gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
            gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

            # cv.imshow("gray l", gray_l)
            # cv.imshow("gray r", gray_r)
            # cv.waitKey(1000)

            # find the chess board corners
            ret_l, corners_l = cv.findChessboardCorners(gray_l, chess_board_size, None)
            ret_r, corners_r = cv.findChessboardCorners(gray_r, chess_board_size, None)

            # if found, add object points, image point (after refining them)
            if ret_l is True and ret_r is True:
                obj_points_arr.append(obj_points)

                corners_l = cv.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                img_points_arr_l.append(corners_l)
                corners_r = cv.cornerSubPix(gray_l, corners_r, (11, 11), (-1, -1), criteria)
                img_points_arr_r.append(corners_r)

                # draw and display the corners
                cv.drawChessboardCorners(img_l, chess_board_size, corners_l, ret_l)
                cv.imshow('img left', img_l)
                cv.drawChessboardCorners(img_r, chess_board_size, corners_r, ret_r)
                cv.imshow('img left', img_r)
                cv.waitKey(1000)
            cv.destroyAllWindows()

            #### CALIBRATION ####

            ret_l, camera_matrix_l, dist_l, r_vecs_l, t_vecs_l = cv.calibrateCamera(obj_points_arr, img_points_arr_l,
                                                                                    frame_size, None, None)
            h_l, w_l, channels_l = img_l.shape
            new_camera_matrix_l, roi_l = cv.getOptimalNewCameraMatrix(camera_matrix_l, dist_l, (w_l, h_l), 1, (w_l, h_l))

            ret_r, camera_matrix_r, dist_r, r_vecs_r, t_vecs_r = cv.calibrateCamera(obj_points_arr, img_points_arr_r,
                                                                                    frame_size, None, None)
            h_r, w_r, channels_r = img_r.shape
            new_camera_matrix_r, roi_r = cv.getOptimalNewCameraMatrix(camera_matrix_r, dist_r, (w_r, h_r), 1, (w_r, h_r))

            #### Stereo Vision Calibration

            flags = 0
            flags |= cv.CALIB_FIX_INTRINSIC

            criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            ret_stereo, new_camera_matrix_l, dist_l, new_camera_matrix_r, dist_r, rot, trans, essential_matrix, \
            fundamental_matrix = cv.stereoCalibrate(
                obj_points_arr, img_points_arr_l, img_points_arr_r,
                new_camera_matrix_l, dist_l,
                new_camera_matrix_r, dist_r,
                gray_l.shape[::-1], criteria_stereo, flags)

            ##### Stereo Rectification #####

            rectify_scale = 1
            rect_l, rect_r, proj_matrix_l, proj_matrix_r, Q, roi_l, roi_r = cv.stereoRectify(new_camera_matrix_l,
                                                                                             dist_l,
                                                                                             new_camera_matrix_r,
                                                                                             dist_r,
                                                                                             gray_l.shape[::-1], rot,
                                                                                             trans,
                                                                                             rectify_scale, (0, 0))

            print(f'{dist_l=}\n{dist_r=}\n')
            print(f'{rot=}\n{trans=}\n')

            stereo_map_l = cv.initUndistortRectifyMap(new_camera_matrix_l, dist_l, rect_l, proj_matrix_l,
                                                      gray_l.shape[::-1],
                                                      cv.CV_16SC2)
            stereo_map_r = cv.initUndistortRectifyMap(new_camera_matrix_l, dist_r, rect_r, proj_matrix_r,
                                                      gray_r.shape[::-1],
                                                      cv.CV_16SC2)

            print("Saving parameters!")
            cv_file = cv.FileStorage('stereo_map.xml', cv.FILE_STORAGE_WRITE)

            cv_file.write('stereo_map_l_x', stereo_map_l[0])
            cv_file.write('stereo_map_l_y', stereo_map_l[1])
            cv_file.write('stereo_map_r_x', stereo_map_r[0])
            cv_file.write('stereo_map_r_y', stereo_map_r[1])

            cv_file.release()


def camera_cv_test():
    # cam = CameraCV()
    # cam.show_video()
    # cam.record_frames()

    # img_folder = 'C:/Users/daniil/PycharmProjects/Odometry/devices/cameras/chess_board_images'
    chess_board_size = (15, 9)
    # CameraCV.monocular_camera_calibration(chess_board_img_dir=img_folder, chess_board_size=chess_board_size)

    stereo_img_folder = 'C:/Users/dakh0322/PycharmProjects/Odometry/devices/cameras/binocular_calibration'
    # CameraCV.get_stereo_images_for_calibration(stereo_img_folder)
    CameraCV.binocular_calibration(stereo_img_folder, chess_board_size)
    # CameraCV.monocular_camera_calibration(stereo_img_folder + "/stereo_right/image_right", chess_board_size)


if __name__ == "__main__":
    camera_cv_test()
