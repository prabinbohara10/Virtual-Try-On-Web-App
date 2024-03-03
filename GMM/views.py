from django.shortcuts import render
from django.http import request, HttpResponse

# Create your views here.
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import os.path as osp

import argparse
import os
import time
from .cp_dataset import CPDataset, CPDataLoader
from .networks import GMM, UnetGenerator, load_checkpoint

from tensorboardX import SummaryWriter
from .visualization import board_add_image, board_add_images, save_images

from django.core.files import File
from django.core.files.storage import default_storage
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
import argparse
from django.views.decorators.csrf import csrf_exempt


from .utils.custom_dataset import dataset_preparation
from .utils.google_drive_apis_python_apps.g_drive_service import GoogleDriveService

main_stage = "GMM"

def get_file_list_from_gdrive():
    g_drive_service = GoogleDriveService()
    return {"files": g_drive_service.list_files()}

def upload_file_to_gdrive(res, bool_delete):

    parent_folder_id = '1DEzVAPGogBIhOmcTY-HBmYDUQlT9epL3'  # Provide the ID of the parent folder where you want to upload the folders
    folders_to_upload = [res]  # List of folder paths to upload
    g_drive_service = GoogleDriveService()
    g_drive_service.main_upload_folders(folders_to_upload, parent_folder_id, bool_delete)
    return "File uploaded successfully"

def get_opt():
    module_dir = os.path.dirname(__file__)  
    dataroot = os.path.join(module_dir, 'data')
    checkpoint = os.path.join(module_dir,'checkpoints/GMM/gmm_50k.pth')
    result  =  os.path.join(module_dir,"static/result")
    
    args = {
        "name": "GMM",
        "name2": "GMM",
        "gpu_ids": "",
        "workers": 1,
        "batch_size": 4,
        "dataroot": dataroot,
        "datamode": "test",
        "stage": "GMM",
        "data_list": "test_pairs.txt",
        "fine_width": 192,
        "fine_height": 256,
        "radius": 5,
        "grid_size": 5,
        "tensorboard_dir": "tensorboard",
        "result_dir": result,
        
        "checkpoint": checkpoint,
        
        "display_count": 1,
        "shuffle": False
    }


    opt = argparse.Namespace(**args)
    return opt


def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    name = opt.name
    save_dir = os.path.join(opt.result_dir, name, opt.datamode)
    
    # Check if the "result" folder exists
    if os.path.exists(save_dir):
        # If it exists, remove it and all its contents
        # Use caution with this operation as it will permanently delete the folder and its contents
        # Make sure to use the correct path and double-check before proceeding
        try:
            # Remove the directory and its contents
            os.system(f'rmdir /s /q "{save_dir}"')  # For Windows
            # For Linux or macOS, use the following line instead:
            # os.system(f'rm -rf "{save_dir}"')
            print("Existing 'result' folder and its contents deleted.")
        except Exception as e:
            print("Error deleting the 'result' folder:", str(e))

    # Create a new "result" folder
    try:
        os.makedirs(save_dir)
        print("New 'result' folder created.")
    except Exception as e:
        print("Error creating the 'result' folder:", str(e))



    ### if warp-cloth is already present:
    data_warp_cloth_dir = os.path.join(opt.dataroot,'test/warp-cloth')
    if os.path.exists(data_warp_cloth_dir):
        # If it exists, remove it and all its contents
        # Use caution with this operation as it will permanently delete the folder and its contents
        # Make sure to use the correct path and double-check before proceeding
        try:
            # Remove the directory and its contents
            os.system(f'rmdir /s /q "{data_warp_cloth_dir}"')  # For Windows
            # For Linux or macOS, use the following line instead:
            # os.system(f'rm -rf "{save_dir}"')
            print("Existing 'warp-cloth' folder and its contents deleted.")
        except Exception as e:
            print("Error deleting the 'warp-cloth' folder:", str(e))

    # Create a new "result" folder
    try:
        os.makedirs(data_warp_cloth_dir)
        print("New 'warp-cloth' folder created.")
    except Exception as e:
        print("Error creating the 'warp-cloth' folder:", str(e))

#################################################################
    ### if warp-mask is already present:
    data_warp_mask_dir = os.path.join(opt.dataroot,'test/warp-mask')
    if os.path.exists(data_warp_mask_dir):
        # If it exists, remove it and all its contents
        # Use caution with this operation as it will permanently delete the folder and its contents
        # Make sure to use the correct path and double-check before proceeding
        try:
            # Remove the directory and its contents
            os.system(f'rmdir /s /q "{data_warp_mask_dir}"')  # For Windows
            # For Linux or macOS, use the following line instead:
            # os.system(f'rm -rf "{save_dir}"')
            print("Existing 'warp-mask' folder and its contents deleted.")
        except Exception as e:
            print("Error deleting the 'warp-mask' folder:", str(e))

    # Create a new "result" folder
    try:
        os.makedirs(data_warp_mask_dir)
        print("New 'warp-mask' folder created.")
    except Exception as e:
        print("Error creating the 'warp-mask' folder:", str(e))


    

    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    result_dir1 = os.path.join(save_dir, 'result_dir')
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    overlayed_TPS_dir = os.path.join(save_dir, 'overlayed_TPS')
    if not os.path.exists(overlayed_TPS_dir):
        os.makedirs(overlayed_TPS_dir)
    warped_grid_dir = os.path.join(save_dir, 'warped_grid')
    if not os.path.exists(warped_grid_dir):
        os.makedirs(warped_grid_dir)

    image_dir = os.path.join(save_dir, 'image')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    cloth_dir = os.path.join(save_dir, 'cloth')
    if not os.path.exists(cloth_dir):
        os.makedirs(cloth_dir)
    
    


    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']
        im_names = inputs['im_name']
        print(im_names, "  ", type(im_names))
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        shape_ori = inputs['shape_ori']  # original body shape without blurring

        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        overlay = 0.7 * warped_cloth + 0.3 * im

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]

        # save_images(warped_cloth, c_names, warp_cloth_dir)
        # save_images(warped_mask*2-1, c_names, warp_mask_dir)
        im_names = []
        im_names.append("latest.jpg")
        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
        save_images(shape_ori.cuda() * 0.2 + warped_cloth *
                    0.8, im_names, result_dir1)
        save_images(warped_grid, im_names, warped_grid_dir)
        save_images(overlay, im_names, overlayed_TPS_dir)

        save_images(im, im_names, image_dir)
        save_images(c, im_names, cloth_dir)

        print(opt.file2)
        im_names = []
        im_names.append(opt.file2)
        save_images(warped_cloth, im_names, data_warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, data_warp_mask_dir)



        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)


def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    # save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    save_dir = os.path.join(opt.result_dir, opt.name, opt.datamode)
    
    
    # Check if the "result" folder exists
    if os.path.exists(save_dir):
        # If it exists, remove it and all its contents
        # Use caution with this operation as it will permanently delete the folder and its contents
        # Make sure to use the correct path and double-check before proceeding
        try:
            # Remove the directory and its contents
            os.system(f'rmdir /s /q "{save_dir}"')  # For Windows
            # For Linux or macOS, use the following line instead:
            # os.system(f'rm -rf "{save_dir}"')
            print("Existing 'result' folder and its contents deleted.")
        except Exception as e:
            print("Error deleting the 'result' folder:", str(e))

    # Create a new "result" folder
    try:
        os.makedirs(save_dir)
        print("New 'result' folder created.")
    except Exception as e:
        print("Error creating the 'result' folder:", str(e))

    
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    p_rendered_dir = os.path.join(save_dir, 'p_rendered')
    if not os.path.exists(p_rendered_dir):
        os.makedirs(p_rendered_dir)
    m_composite_dir = os.path.join(save_dir, 'm_composite')
    if not os.path.exists(m_composite_dir):
        os.makedirs(m_composite_dir)
    im_pose_dir = os.path.join(save_dir, 'im_pose')
    if not os.path.exists(im_pose_dir):
        os.makedirs(im_pose_dir)
    shape_dir = os.path.join(save_dir, 'shape')
    if not os.path.exists(shape_dir):
        os.makedirs(shape_dir)
    im_h_dir = os.path.join(save_dir, 'im_h')
    if not os.path.exists(im_h_dir):
        os.makedirs(im_h_dir)  # for test data

    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
        outputs = model(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, 2*cm-1, m_composite],
                   [p_rendered, p_tryon, im]]
        
        im_names = []
        im_names.append("latest.jpg")
        save_images(p_tryon, im_names, try_on_dir)
        save_images(im_h, im_names, im_h_dir)
        save_images(shape, im_names, shape_dir)
        save_images(im_pose, im_names, im_pose_dir)
        save_images(m_composite, im_names, m_composite_dir)
        save_images(p_rendered, im_names, p_rendered_dir)  # For test data

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)



def main(opt):
    
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    test_dataset = CPDataset(opt)

    # create dataloader
    test_loader = CPDataLoader(opt, test_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & test
    if opt.stage == 'GMM':
    # if main_stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, test_loader, model, board)
    elif opt.stage == 'TOM':
    # elif main_stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, test_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished test %s, named: %s!' % (opt.stage, opt.name))


@csrf_exempt
def index(request):
    context = {'condition_met': False}
    if(request.method == "POST"):
        file1 = request.FILES['sentFile1'] # person
        file2 = request.FILES['sentFile2'] # cloth

        print("person file, cloth file:", file1, file2)

        # save images to folder:
        person_file_path = default_storage.save("image/" + file1.name, file1)
        cloth_file_path = default_storage.save("cloth/" + file2.name, file2)
        
        print("Person file saved at:", person_file_path)
        print("Cloth file saved at:", cloth_file_path)

        file1_name = str(person_file_path).split("/")[1]
        file2_name = str(cloth_file_path).split("/")[1]

        dataset_preparation(file1_name, file2_name)
        
        #... write
        module_dir = os.path.dirname(__file__)  
        file_path = os.path.join(module_dir, 'data/test_pairs.txt')
        with open(file_path, 'w') as file:
            file.write(file1_name + " " + file2_name)


        #f.write(file1.name," ", file2.name)
        #f.close()

        opt = get_opt()
        opt.file1 = file1_name
        opt.file2 = file2_name
        main(opt)
        
        ##########setup for TOM#####
        main_stage = "TOM"
        opt.stage = "TOM"
        opt.name = "TOM"
        checkpoint = os.path.join(module_dir,'checkpoints/TOM/tom_final.pth')
        result =  os.path.join(module_dir,"static/result2")
        opt.checkpoint = checkpoint
        opt.result_dir = result



        main(opt)
        print("doneewee")
        ##original = load_img(file_url, target_size=(224, 224))
        #numpy_image = img_to_array(original)
        context = {'condition_met': True}


        ###################################
        #uploading to drive part:
        module_dir = os.path.dirname(__file__)  
        result1 =  os.path.join(module_dir,"static/result")
        result2  =  os.path.join(module_dir,"static/result2")
        upload_file_to_gdrive(result1, bool_delete =True)
        upload_file_to_gdrive(result2,  bool_delete =False)

        #... Making response ready ...#
        #try_on_image = Image.open(osp.join(settings.BASE_DIR))
        #return HttpResponse(, content_type="image/png")


    
    return render(request, 'home.html', context)