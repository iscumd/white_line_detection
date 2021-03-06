# white_line_detection
From package '[White Line Detection](https://github.com/iscumd/white_line_detection/tree/main)'
# File
`./src/white_line_detection.cpp`

## Summary 
 Generic white line detection node. Detects white lines found in an image and performs some action on them. This node has configurable 
frontends and backends, which allow for configuring the line detection algorithm and action performed on the lines respectively.
This node should function with any rectilinear monocular camera, which has a transform to base.

## Topics

### Publishes
- `(pointcloud2 only) /camera/camera_points`: The points from the white line detection. These will be in the base_frame as set by param.
- `/camera/test_img`: Image topic publishing the final transformed image before being passed to line detection. Used for debugging in lieu of imshow.

### Subscribes
- `/camera/image_raw`: The raw image from the camera node/gazebo to be processed.
- `/camera/camera_info`: The standard camera info topic, used to find things like resolution. Must be from the same camera that outputs the image topic.

## Params
- `thresholder`: The thresholding strategy to use. This is the algorithm that will actually detect the white lines. Default: isc.dyn_gauss

    **Options**:

    1. 'basic': A simple global threshold. Uses `lower_bound_white` as its threshold value.

    2. 'isc.dyn_gauss': An ISC made algorithm that finds the mean of the image, then sets the threshold value to be 3 standard deviations away. 
works well for environments that do not have major shadows, including rain with puddles and reflections.

- `backend`: The backend in use. This is the action performed on the detected lines. Default: pointcloud2

    **Options**:

    1. 'pointcloud2': Publishes lines as pointclouds by raytracing a path from the pixel to ground plane.

- `lower_bound_white`: The lower threshold value for what is considered white. Only used with `basic` thresholder
- `sample_nth_pixel`: How many pixels to sample from the white lines to be used in the pointcloud.
- `kernel_size`: The size of the erosion kernel used, when appropriate.
- `camera_frame`: The frame the camera is in. Default: "camera_link"
- `base_frame`: The base frame the camera_frame will be relative to. Default: "base_footprint"
- `debug_only`: Does not use backends if set. Use for testing when you only want to run this node and a rosbag, for example. Default: false

## Potential Improvements
Add shadow removal to dyn_gauss. Look into using ML as an alternitive thresholder. 

