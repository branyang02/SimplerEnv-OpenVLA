import numpy as np


def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    image = obs["image"][camera_name]["Color"]
    image = image[..., :3]  # Remove alpha channel if it exists
    # Convert float32 image to uint8
    if image.dtype == np.float32:
        # Scale the image if the values are between 0 and 1
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        # Directly convert to uint8 if it's not float32
        image = image.astype(np.uint8)

    return image
