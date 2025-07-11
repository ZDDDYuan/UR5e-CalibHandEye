import numpy as np
import pyrealsense2 as rs
import cv2
import time

class RealSenseD435iCamera:
    def __init__(self, height: int = 480, width: int = 640, fps: int = 30):
        self.im_height = height
        self.im_width = width
        self.fps = fps
        self.intrinsics = None
        self.coeffs = None
        self.scale = None
        self.pipeline = None
        self.connect()

    def connect(self):
        # Configure depth and color streams
        # Get device and reset it first
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)

        # OPTIONAL: reset before starting stream
        ctx = rs.context()
        devices = ctx.query_devices()
        if devices.size() > 0:
            devices[0].hardware_reset()
            time.sleep(2)  # waiting for hardware reset
        else:
            raise RuntimeError("No RealSense device found")

        # Now start streaming
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics, self.coeffs = self.get_intrinsics(rgb_profile)
        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        print("[INFO] camera depth scale:",self.scale)
        print("[INFO] RealSense D435i have connected ...")
        
    def get_frames(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()

        # align
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # no align
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.float32)
        # depth_image *= self.scale
        depth_image = np.expand_dims(depth_image, axis=2)
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def plot_image(self):
        color_image, depth_image = self.get_frames()
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # cv2.imwrite('color_image.png', color_image)
        # cv2.waitKey(5000)
        
    def get_intrinsics(self, rgb_profile):
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        # camera intrinsics form is as follows.
        #[[fx,0,ppx],
        # [0,fy,ppy],
        # [0,0,1]]
        intrinsics = np.array([raw_intrinsics.fx, 0, raw_intrinsics.ppx, 0, raw_intrinsics.fy, raw_intrinsics.ppy, 0, 0, 1]).reshape(3, 3)
        coeffs = np.array(raw_intrinsics.coeffs)
        return intrinsics, coeffs
    
    def release(self):
        self.pipeline.stop()

if __name__ == "__main__":
    camera = RealSenseD435iCamera(480, 640, 30)
    # Show color and depth frames
    cv2.namedWindow('color')
    cv2.namedWindow('depth')

    while True:
        camera_color_img, camera_depth_img = camera.get_frames()
        rgb_data = camera_color_img
        cv2.imshow('color', rgb_data)
        cv2.imshow('depth', camera_depth_img)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.release()