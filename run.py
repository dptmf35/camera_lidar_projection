import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
import cv2, numpy as np
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import json

bridge = CvBridge()

# open calibration configuration file
with open('calibration_configs.json', 'r') as f:
    data = json.load(f)

camera_matrix = np.array(data['camera_matrix'])
extrinsic_matrix = np.array(data['extrinsic_matrix'])

class CameraLidarProjection : 

    """
    Run Camera-Lidar Projection. It will publish an output topic.
    
    init parameters:
    - image_sub : camera image topic message 
    - pcl_sub : lidar pc2 topic message
    - output_pub : output topic publisher
    - ts : camera-lidar message time synchronizer
    - voxel_size [optional] : if not None, filter pcs using voxel grid with defined size
    """

    def __init__(self, voxel_size) :
        rospy.init_node('CameraLidarProjection', anonymous=True)

        image_sub = message_filters.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image)
        pcl_sub = message_filters.Subscriber('/hesai/pandar_points', PointCloud2)    
        self.output_pub = rospy.Publisher('/output', Image, queue_size=10)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, pcl_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.callback, voxel_size)

    def callback(self, image_data, pcl_data, voxel_size):
        # Convert ROS Image message to OpenCV image
        camera_data = bridge.imgmsg_to_cv2(image_data, "bgr8")

        # Read x, y, z, and intensity fields from LiDAR point cloud data
        lidar_data = np.array(list(pc2.read_points(pcl_data, skip_nans=True, field_names=('x', 'y', 'z'))))
        lidar_data = lidar_data[(lidar_data[:, 0] != 0) | (lidar_data[:, 1] != 0) | (lidar_data[:, 2] != 0)] # nonzero points

        if voxel_size : 
            filtered_data = self.voxel_downsampling(lidar_data[:,:3], voxel_size)
        else : 
            filtered_data = lidar_data[:, :3]

        lidar_points_cam =  np.dot(extrinsic_matrix, np.transpose((np.hstack((filtered_data, np.ones((filtered_data.shape[0], 1)))))))

        # LiDAR points to image coordinates
        projected_points = np.transpose((np.dot(camera_matrix, lidar_points_cam[:3, :])))

        # normalization
        projected_points = projected_points[:, :2] / projected_points[:, 2:] 

        height, width = camera_data.shape[:2]

        # drawing points on image coords
        for i, point in enumerate(projected_points.astype(int)) :
                x, y = point
                if 0 <= x < width and 0 <= y < height :
                    camera_data = cv2.circle(camera_data, (x, y), 2, (0, 255, 0), -1)

        ros_image = bridge.cv2_to_imgmsg(camera_data, "bgr8")
        self.output_pub.publish(ros_image)

    def voxel_downsampling(self, lidar_data, voxel_size):
        voxel_indices = np.floor(lidar_data[:, :3] / voxel_size).astype(int)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        filtered_data = lidar_data[unique_indices]
        
        return filtered_data




if __name__ == '__main__':
    try:
        projection = CameraLidarProjection(voxel_size=None)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass