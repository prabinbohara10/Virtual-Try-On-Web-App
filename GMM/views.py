from django.shortcuts import render
from django.http import request

# Create your views here.
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def get_opt():
    module_dir = os.path.dirname(__file__)  
    dataroot = os.path.join(module_dir, 'data')
    checkpoint = os.path.join(module_dir,'checkpoints/GMM/gmm_50k.pth')
    result  =  os.path.join(module_dir,"static/result")
    args = {
        "name": "GMM",
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

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)




def main():
    opt = get_opt()
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
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, test_loader, model, board)
    elif opt.stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, test_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished test %s, named: %s!' % (opt.stage, opt.name))


def index(request):
    context = {'condition_met': False}
    if(request.method == "POST"):
        file1 = request.FILES['sentFile1']
        file2 = request.FILES['sentFile2']

        print(file1.name)
        
        module_dir = os.path.dirname(__file__)  
        file_path = os.path.join(module_dir, 'data/test_pairs.txt')

        #f = open("/data/test_paris.txt", "w")
       
        with open(file_path, 'w') as file:
            file.write(file1.name + " " + file2.name)


        #f.write(file1.name," ", file2.name)
        #f.close()

        main()
        ##original = load_img(file_url, target_size=(224, 224))
        #numpy_image = img_to_array(original)
        context = {'condition_met': True}

    
    return render(request, 'home.html', context)