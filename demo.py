import torch
import os
from imageio import imread, imsave
from skimage.transform import resize
import numpy as np
import cv2
from path import Path
import argparse
from tqdm import tqdm

from models import DispNetS
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='./demo/source/', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='./demo/output/', type=str, help="Output directory")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    o_dir = Path(args.output_dir)
    o_dir.makedirs_p()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    disp_net = DispNetS().to(device)
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    vid_list = [i for i in list(Path(args.dataset_dir).walkdirs()) if i[-4:] == 'data']
    vid_list.sort()
    N = len(vid_list)
    print('{} videos to demo'.format(N))

    for ii, vid_path in enumerate(vid_list):
        
        vid_name = vid_path.split('/')[-2]

        dataset_dir = Path(vid_path)
        output_dir = Path(args.output_dir + vid_name)
        output_dir.makedirs_p()
        
        test_files = sum([list(dataset_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
        vid_save_name = str(args.output_dir)+'{}.mp4'.format(vid_name)

        print('{}/{} - {} - {} files to test::video saved to \'{}\''.format(ii, N, vid_name, len(test_files), vid_save_name))

        for file in tqdm(test_files):

            img = imread(file)

            h,w,_ = img.shape

            if (not args.no_resize) and (h != args.img_height or w != args.img_width):
                img = cv2.resize(img, (args.img_width, args.img_height))
            
            img = np.transpose(img, (2, 0, 1))

            tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
            tensor_img = ((tensor_img/255 - 0.5)/0.5).to(device)

            output = disp_net(tensor_img)[0]

            file_path, file_ext = file.relpath(args.dataset_dir).splitext()
            # print(file_path)
            # print(file_path.splitall())
            file_name = '-'.join(file_path.splitall()[1:])
            # print(file_name)

            if args.output_disp:
                disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
                imsave(output_dir/'{}_disp{}'.format(file_name, file_ext), np.transpose(disp, (1,2,0)))
            if args.output_depth:
                depth = 1/output
                depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
                imsave(output_dir/'{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1,2,0)))
        

        #make vid
        img_dir = output_dir

        test_files.sort()

        video = cv2.VideoWriter(vid_save_name, 0x7634706d, 10, (args.img_width, args.img_height*2))

        for file in test_files:
            file_core_name, file_ext = str(file).split('.')[-2:]
            file_core_name = '-'.join(file_core_name.split('/')[-3:])
            input_img = cv2.imread(file)
            h,w,_ = input_img.shape
            if (not args.no_resize) and (h != args.img_height or w != args.img_width):
                input_img = cv2.resize(input_img, (args.img_width, args.img_height))
            disp_img = cv2.imread(output_dir + '/{}_disp.{}'.format(file_core_name, file_ext))

            video.write(np.concatenate((input_img, disp_img)))

        cv2.destroyAllWindows()
        video.release()

if __name__ == '__main__':
    main()
