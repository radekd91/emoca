from abc import abstractmethod, ABC
import numpy as np
import torch
import pickle as pkl
from gdl.utils.FaceDetector import FaceDetector, MTCNN
import os, sys
from gdl.utils.other import get_path_to_externals 
from pathlib import Path
from torchvision import transforms as tf
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from face_alignment.utils import get_preds_fromhm, crop
from collections import OrderedDict
import torch.nn.functional as F
from munch import Munch

path_to_sbr = (Path(get_path_to_externals()) / ".." / ".." / "landmark-detection" / "SBR").absolute()
# path_to_hrnet = (Path(get_path_to_externals())  / "landmark-detection"/ "SBR").absolute()

if str(path_to_sbr) not in sys.path:
    sys.path.insert(0, str(path_to_sbr))

from lib.models   import obtain_model, remove_module_dict
# from lib.config_utils import obtain_lk_args as obtain_args

INPUT_SIZE = 256


class SBR(FaceDetector):

    def __init__(self, device = 'cuda', instantiate_detector='sfd', threshold=0.5):
        # args = obtain_args()
        # cfg = path_to_sbr / "experiments/300w/face_alignment_300w_hrnet_w18.yaml"
        # model_file = path_to_sbr / "hrnetv2_pretrained" / "HR18-300W.pth"

        # args = Munch()
        # args.train_lists                    #type=str,   nargs='+',      help='The list file path to the video training dataset.')
        # args.eval_vlists                    #type=str,   nargs='+',      help='The list file path to the video testing dataset.')
        # args.eval_ilists                    #type=str,   nargs='+',      help='The list file path to the image testing dataset.')
        # args.num_pts                        #type=int,                   help='Number of point.')
        # args.model_config                   #type=str,                   help='The path to the model configuration')
        # args.opt_config                     #type=str,                   help='The path to the optimizer configuration')
        # args.lk_config                      #type=str,                   help='The path to the LK configuration')
        # # args.Generation               
        # args.heatmap_type                   #type=str,   choices=['gaussian','laplacian'], help='The method for generating the heatmap.')
        # args.data_indicator                 #type=str, default='300W-68',help='The dataset indicator.')
        # args.video_parser                   #type=str,                   help='The video-parser indicator.')
        # # args.nsform               
        # args.pre_crop_expand                #type=float,                 help='parameters for pre-crop expand ratio')
        # args.sigma                          #type=float,                 help='sigma distance for CPM.')
        # args.scale_prob                     #type=float,                 help='argument scale probability.')
        # args.scale_min                      #type=float,                 help='argument scale : minimum scale factor.')
        # args.scale_max                      #type=float,                 help='argument scale : maximum scale factor.')
        # args.scale_eval                     #type=float,                 help='argument scale : maximum scale factor.')
        # args.rotate_max                     #type=int,                   help='argument rotate : maximum rotate degree.')
        # args.crop_height                    #type=int,   default=256,    help='argument crop : crop height.')
        # args.crop_width                     #type=int,   default=256,    help='argument crop : crop width.')
        # args.crop_perturb_max               #type=int,                   help='argument crop : center of maximum perturb distance.')
        # args.arg_flip                       #action='store_true',        help='Using flip data argumentation or not ')
        # # args.tion options             
        # args.eval_once                      #action='store_true',        help='evaluation only once for evaluation ')
        # args.error_bar                      #type=float,                 help='For drawing the image with large distance error.')
        # args.batch_size                     #type=int,   default=2,      help='Batch size for training.')
        # # args.points               
        # args.print_freq                     #type=int,   default=100,    help='print frequency (default: 200)')
        # args.init_model                     #type=str,                   help='The detector model to be initalized.')
        # args.save_path                      #type=str,                   help='Folder to save checkpoints and log.')
        # args.eration                
        # args.workers                        #type=int,   default=8,      help='number of data loading workers (default: 2)')
        # # args.andom Seed               
        # args.rand_seed                      #type=int,                   help='manual seed')

        # cfg = path_to_hrnet / "experiments/aflw/face_alignment_aflw_hrnet_w18.yaml"
        # model_file = path_to_hrnet / "hrnetv2_pretrained" / "HR18-AFLW.pth"

        snapshot =  path_to_sbr / "snapshots/300W-CPM-DET/checkpoint/cpm_vgg16-epoch-049-050.pth"

        snapshot = torch.load(snapshot)

        # model_config = load_configure(args.model_config, logger)
        # lk_config = load_configure(args.lk_config, logger)
        self.net = obtain_model(model_config, lk_config, args.num_pts + 1)
        # self.model = models.get_face_alignment_net(config)
        # # self.num_landmarks = 68
        # self.num_landmarks = args.num_pts
        # self.args = args
        try:
            weights = remove_module_dict(snapshot['detector'])
        except:
            weights = remove_module_dict(snapshot['state_dict'])
        self.net.load_state_dict(weights)
        
        # if hasattr(net, 'specify_parameter'):
        #     net_param_dict = net.specify_parameter(opt_config.LR, opt_config.Decay)
        # else:
        #     net_param_dict = net.parameters()

        # optimizer, scheduler, criterion = obtain_optimizer(net_param_dict, opt_config, logger)
        
        # last_info = torch.load(last_info)
        # checkpoint = torch.load(last_info['last_checkpoint'])
        # assert last_info['epoch'] == checkpoint['epoch'], 'Last-Info is not right {:} vs {:}'.format(last_info, checkpoint['epoch'])
        # self.net.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])


        self.detector = None
        if instantiate_detector == 'mtcnn':
            self.detector = MTCNN()
        elif instantiate_detector == 'sfd': 
            # Get the face detector

            face_detector_kwargs =  {
                "filter_threshold": threshold
            }
            self.detector = SFDDetector(device=device, verbose=False, **face_detector_kwargs)

        elif instantiate_detector is not None: 
            raise ValueError("Invalid value for instantiate_detector: {}".format(instantiate_detector))
        
        # # self.transforms = [utils.transforms.CenterCrop(INPUT_SIZE)]
        self.transforms = [tf.ToTensor()]
        # self.transforms += [utils.transforms.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]
        self.crop_to_tensor = tf.Compose(self.transforms)

    
    # @profile
    @torch.no_grad()
    def run(self, image, with_landmarks=False, detected_faces=None):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        if detected_faces is None: 
            bboxes = self.detector.detect_from_image(image)
        else:
            print("Image size: {}".format(image.shape)) 
            bboxes = [np.array([0, 0, image.shape[1], image.shape[0]])]

        final_boxes = []
        final_kpts = []

        for bbox in bboxes:
            center = torch.tensor(
                [bbox[2] - (bbox[2] - bbox[0]) / 2.0, bbox[3] - (bbox[3] - bbox[1]) / 2.0])
            # center[1] = center[1] - (bbox[3] - bbox[1]) * 0.12 # this might result in clipped chin
            center[1] = center[1] + (bbox[3] - bbox[1])  * 0.00 # this appears to be a good value
            # scale = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / self.detector.reference_scale
            # scale = 1.2
            # scale = 1.3
            # scale = 1.4
            scale = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / self.detector.reference_scale * 0.65 # this appears to be a good value
            # print("Scale: {}".format(scale))
            # print("Bbox: {}".format(bbox))
            # print("Width: {}".format(bbox[2] - bbox[0]))
            # print("Height: {}".format(bbox[3] - bbox[1]))
            # scale = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / 256
            # scale = ((bbox[2] - bbox[0] + bbox[3] - bbox[1]) / image.shape[0] ) * 0.85
            images_ = crop(image, center, scale, resolution=256.0)
            images = self.crop_to_tensor(images_)
            if images.ndimension() == 3:
                images = images.unsqueeze(0)
            # images = nn.atleast4d(images).cuda()
            
            # X_recon, lms, X_lm_hm = self.detect_in_crop(images)
            pts_img, X_lm_hm = self.detect_in_crop(images, center.unsqueeze(0), torch.tensor([scale]))
            # pts, pts_img = get_preds_fromhm(X_lm_hm, center.numpy(), scale)
            # torch.cuda.empty_cache()
            if pts_img is None:
                del pts_img
                if with_landmarks:
                    return [],  f'kpt{self.num_landmarks}', []
                else:
                    return [],  f'kpt{self.num_landmarks}'
            else:
                import matplotlib.pyplot as plt
                # # image to numpy array
                # images_np = images.cpu().numpy()[0].transpose((1, 2, 0))
                # images_np = images_ / 255.
                # print("images_np.shape: {}".format(images_np.shape))
                # plt.figure(1)
                # plt.imshow((images_np * 255.).clip(0, 255).astype(np.uint8))
                # plt.figure(2)
                # plt.imshow(image)
                # for i in range(len(lms)):
                for i in range(len(pts_img)):
                    kpt = pts_img[i][:68].squeeze().detach().cpu().numpy()
                    left = np.min(kpt[:, 0])
                    right = np.max(kpt[:, 0])
                    top = np.min(kpt[:, 1])
                    bottom = np.max(kpt[:, 1])
                    final_bbox = [left, top, right, bottom]
                    final_boxes += [final_bbox]
                    final_kpts += [kpt]

                    # plot points                 
                    # plt.figure(1)
                    # plt.plot(kpt[:, 0], kpt[:, 1], 'ro')
                    # plt.figure(2)
                    # plt.plot(pts_img[i][:, 0], pts_img[i][:, 1], 'ro')
                # print("Plotting landmarks")
                # plt.show()

        # del lms # attempt to prevent memory leaks
        if with_landmarks:
            return final_boxes, f'kpt{self.num_landmarks}', final_kpts
        else:
            return final_boxes, f'kpt{self.num_landmarks}'


    @torch.no_grad()
    def detect_in_crop(self, crop, center, scale):
        with torch.no_grad():
            output = self.model(crop)
            
            batch_heatmaps, batch_locs, batch_scos = self.net(crop)
            # batch_heatmaps, batch_locs, batch_scos, batch_next, batch_fback, batch_back = self.net(crop)


        return preds, score_map

