# white_line_detection
From package '[White Line Detection](https://github.com/iscumd/white_line_detection/tree/main)'
# File
`./src/white_line_detection.cpp`

## Summary 
 Generic white line detection node. Converts any white lines detected in an image message to pointclouds relative to the camera.
This node should function with any rectilinear monocular camera, which has a frame relative to base set in tf somehow.

## Topics

### Publishes
- `/camera/camera_points`: The points from the white line detection. These will be in the base_frame as set by param.
- `/camera/test_img`: Image topic publishing the final transformed image before being passed to line detection. Used for debugging in lieu of imshow.

### Subscribes
- `/camera/image_raw`: The raw image from the camera node/gazebo to be processed.
- `/camera/camera_info`: The standard camera info topic, used to find things like resolution. Must be from the same camera that outputs the image topic.

## Params
- `lower_bound_white`: The lower threshold value for what is considered white.
- `kernel_size`: The size of the erosion kernel used for threasholding. Larger values remove more white.
- `sample_nth_pixel`: How many pixels to sample from the white lines to be used in the pointcloud.
- `camera_frame`: The frame the camera is in. Default: "camera_link"
- `base_frame`: The base frame the camera_frame will be relative to. Default: "base_footprint"

## Potential Improvements
Add dynamic thresholding. Use ML to detect the white lines, allowing us to use this in any weather condition. 

# Misc 
 This node is primarily useful in controlled environments where the lighting is relatively static. It currently has issues otherwise. 
