#检测人脸及关键点
import cv2
import math
import numpy as np

#精确的面部定位网络库，从Python中检测面部地标，该网络能够检测二维和三维坐标中的点
import face_alignment


class FaceDetect:
    def __init__(self, device, detector):
        #指定在CPU/GPU上运行
        #面部检测器
        # landmarks will be detected by face_alignment library. Set device = 'cuda' if use GPU.
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector=detector)

    def align(self, image):
        #关键点检测，得到标记点人脸
        landmarks = self.__get_max_face_landmarks(image)

        if landmarks is None:
            return None

        else:
            #根据标记点旋转矫正人脸
            return self.__rotate(image, landmarks)
        
    #将关键点边界框按固定的比例扩张并裁剪出人脸区域
    def __get_max_face_landmarks(self, image):
        preds = self.fa.get_landmarks(image)
        if preds is None:
            return None

        elif len(preds) == 1:
            return preds[0]

        else:
            # find max face：查找最大面孔
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            return preds[max_face_index]

    @staticmethod
    def __rotate(image, landmarks):
        # rotation angle：旋转角度
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        #弧度：
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

        # image size after rotating：旋转后的图像大小
        height, width, _ = image.shape
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))

        # translation
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        # affine matrix：仿射矩阵
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

        #图像旋转
        #cv2.warpAffine()：仿射变换
        #borderValue：边界值填充（为白色）
        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

        #np.concatenate()：用来对数列或矩阵进行合并
        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T
        return image_rotate, landmarks_rotate


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('3989161_1.jpg'), cv2.COLOR_BGR2RGB)
    fd = FaceDetect(device='cpu')
    face_info = fd.align(img)
    if face_info is not None:
        image_align, landmarks_align = face_info

        for i in range(landmarks_align.shape[0]):
            cv2.circle(image_align, (int(landmarks_align[i][0]), int(landmarks_align[i][1])), 2, (255, 0, 0), -1)

        cv2.imwrite('image_align.png', cv2.cvtColor(image_align, cv2.COLOR_RGB2BGR))
