import cv2
import subprocess
import os
from django.conf import settings

from rembg import remove
from PIL import Image

from .image_mask import make_body_mask
from .openpose_json import generate_pose_keypoints
from .cloth_mask import cloth_masking
#from GMM.checkpoints.openpose import for_django_views

def overlay_images(original_image_path, output_path):
    # Load the original image
    original_image = Image.open(original_image_path)

    # Change the background to white
    white_background_image = Image.new("RGBA", original_image.size, (255, 255, 255, 255))
    white_background_image.paste(original_image, (0, 0), original_image)
    white_background_image.save(output_path)

def dataset_preparation(filename_person,filename_cloth):
    filepath_person = os.path.join(settings.BASE_DIR, "GMM/data/test/image/", filename_person)
    filepath_cloth = os.path.join(settings.BASE_DIR, "GMM/data/test/cloth/", filename_cloth)
    print(filepath_person, filepath_cloth)

    # ... adjust backgrounds:...#
    input_path = filepath_person
    output_path = os.path.join(settings.BASE_DIR, "GMM/data/test/image_bg_removed/", filename_person)

    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)
    overlay_images(output_path, input_path)
    

    # ..... Resize/Crop Images 192 x 256 (width x height) ..... # 
    img_p = cv2.imread(filepath_person)
    person_resize = cv2.resize(img_p, (192, 256))
    # save resized person image
    cv2.imwrite(filepath_person, person_resize) 
    
    img_c = cv2.imread(filepath_cloth)
    cloth_resize = cv2.resize(img_c, (192, 256)) 
    # save resized cloth image
    cv2.imwrite(filepath_cloth, cloth_resize)

    
    # ..... Cloth Masking ..... #
    res_path = os.path.join(settings.BASE_DIR, "GMM/data/test/cloth-mask/", filename_cloth)
    clothmask = cloth_masking(filepath_cloth, res_path)
    print("Cloth Masking done !!!")
    
    
    # ..... Image parser ..... # 
    data_root = os.path.join(settings.BASE_DIR, "GMM/data/")
    checkpoint = os.path.join(settings.BASE_DIR, "GMM/checkpoints/graphonomy_inference.pth")
    cmd_parse = "python GMM/utils/graphonomy_inference.py --loadmodel "+ checkpoint + " --data_root "+ data_root+ " --img_path " + filename_person + " --output_name "+ filename_person
    subprocess.call(cmd_parse, shell=True)
    print("Image Parsing done !!!")
    
    
    # ..... Person Image Masking ..... #
    #img_file = "000010_0.jpg", seg_file = "000010_0.png" 
    seg_file = filename_person.replace(".jpg", ".png")
    img_mask = make_body_mask(filename_person, seg_file, data_root)
    print("Person Image masking done !!!")
    
    
    # ..... Generate Pose Keypoints .....# 
    base_root = os.path.join(settings.BASE_DIR)
    # for_django_views.main()
    # # cmd_keypoint = "python GMM/checkpoints/openpose/python/openpose_python.py"
    # # subprocess.call(cmd_keypoint, shell=True)
    
    # # cmd_change_dir = "cd "+ base_dir +"/GMM/checkpoints/openpose"
    # # subprocess.call(cmd_change_dir, shell=True)
    # # cmd_parse = "bin/OpenPoseDemo.exe --image_dir "+ base_dir +"/GMM/data/test/image --write_json " + data_root +"test/pose"
    # # print(cmd_parse)
    # # subprocess.call(cmd_parse, shell=True)

    pose_keypoints = generate_pose_keypoints(filename_person, base_root)
    print("Person Image Keypoints done!!!")
