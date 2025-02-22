import pyzed.sl as sl
import cv2
import random

from ultralytics import YOLO
import numpy as np


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def main():

    model_weight_path = "models/yolov8n.pt"
    model = YOLO(model_weight_path)
    img_size = 640
    conf_threshold = 0.2
    iou_threshold = 0.45

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()

    # Variables to store depth and image from the zed camera.
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    runtime_parameters = sl.RuntimeParameters()

    # camera parameters need for depth calculation
    camera_info = zed.get_camera_information()
    calibration_params = camera_info.camera_configuration.calibration_parameters

    f_x = calibration_params.left_cam.fx
    f_y = calibration_params.left_cam.fy
    c_x = calibration_params.left_cam.cx
    c_y = calibration_params.left_cam.cy

    while True:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            # Since zed_sdk uses its own sl.MAT class to store images, we use the provided `get_data()`
            # function the convert the matrix into a numpy array.
            image_np = image_mat.get_data()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

            # Get the depth map of the left camera.
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

            # Run the model and get the predictions
            results = model.predict(
                image_np,
                save=False,
                imgsz=img_size,
                conf=conf_threshold,
                iou=iou_threshold,
            )

            for result in results:
                # Get the bounding box
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    # Center of the bounding box
                    center_point_x = int((box[0] + box[2]) / 2)
                    center_point_y = int((box[1] + box[3]) / 2)

                    # find the depth at the center of the target
                    err, depth_value = depth_mat.get_value(
                        center_point_x, center_point_y
                    )

                    # convert to 3d coordinates
                    u, v = center_point_x, center_point_y

                    Z = depth_value
                    X = ((u - c_x) * Z) / (f_x)
                    Y = ((v - c_y) * Z) / (f_y)

                    label_text = f"x:{X:.2f}m y:{Y:.2f}m z:{Z:.2f}m"

                    # Plot the bounding box and depth information on the label
                    plot_one_box(box, image_np, (100, 100, 100), label_text, 2)

            # Display the left image from the numpy array
            cv2.imshow("Image", image_np)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    # Close the camera
    cv2.destroyAllWindows()
    zed.close()


if __name__ == "__main__":
    main()