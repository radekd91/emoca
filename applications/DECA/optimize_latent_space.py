import torch
import torch.functional as F
from applications.DECA.interactive_deca_decoder import load_deca_and_data, test, plot_results
import copy
from layers.losses.EmoNetLoss import EmoNetLoss
from models.DECA import DecaModule, DECA, DecaMode
from skimage.io import imread
import numpy as np


class TargetEmotionCriterion(torch.nn.Module):

    def __init__(self,
                 target_image,
                 use_feat_1 = False,
                 use_feat_2 = True,
                 use_valence = False,
                 use_arousal = False,
                 use_expression = False,
                 ):
        super().__init__()
        self.emonet_loss = EmoNetLoss('cuda')

        if isinstance(target_image, str):
            target_image = imread(target_image)[:,:,:3]

        if isinstance(target_image, np.ndarray):
            target_image = np.transpose(target_image, [2,0,1])[None, ...]
            if target_image.dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
                target_image = target_image.astype(np.float32)
                target_image /= 255.

        # if target_image.shape[2] != self.emonet_loss.size[0] or target_image.shape[3] != self.emonet_loss.size[1]:
        #     target_image = F.interpolate(target_image)

        target_image = torch.from_numpy(target_image).cuda()
        # self.target_image = target_image
        self.register_buffer('target_image', target_image)
        self.target_emotion = self.emonet_loss.emonet_out(target_image)

        self.use_feat_1 = use_feat_1
        self.use_feat_2 = use_feat_2
        self.use_valence = use_valence
        self.use_arousal = use_arousal
        self.use_expression = use_expression

    def __call__(self, image):
        return self.forward(image)

    def forward(self, image):
        return self.compute(image)

    def compute(self, image):
        input_emotion = self.emonet_loss.emonet_out(image)
        emo_feat_loss_1 = self.emonet_loss.emo_feat_loss(input_emotion['emo_feat'], self.target_emotion['emo_feat'])
        emo_feat_loss_2 = self.emonet_loss.emo_feat_loss(input_emotion['emo_feat_2'], self.target_emotion['emo_feat_2'])
        valence_loss = self.emonet_loss.valence_loss(input_emotion['valence'], self.target_emotion['valence'])
        arousal_loss = self.emonet_loss.arousal_loss(input_emotion['arousal'], self.target_emotion['arousal'])
        expression_loss = self.emonet_loss.expression_loss(input_emotion['expression'], self.target_emotion['expression'])

        total_loss = torch.zeros_like(emo_feat_loss_1)
        if self.use_feat_1:
            total_loss = total_loss + emo_feat_loss_1

        if self.use_feat_2:
            total_loss = total_loss + emo_feat_loss_2

        if self.use_valence:
            total_loss = total_loss + valence_loss

        if self.use_arousal:
            total_loss = total_loss + arousal_loss

        if self.use_expression:
            total_loss = total_loss + expression_loss

        return total_loss



def optimize(deca,
             values,
             optimize_detail=True,
             optimize_identity=False,
             optimize_expression=True,
             optimize_pose=False,
             optimize_texture=False,
             optimize_cam=False,
             optimize_light=False,
             loss_to_use=None
             ):

    loss_to_use = loss_to_use or "loss"

    parameters = []

    # deca.deca.config.train_coarse = True
    # deca.deca.config.mode = DecaMode.DETAIL
    # # deca.deca.config.mode = DecaMode.COARSE

    if optimize_detail:
        values['detailcode'] = torch.autograd.Variable(values['detailcode'].detach().clone(), requires_grad=True)
        parameters += [values['detailcode']]

    if optimize_identity:
        values['shapecode'] = torch.autograd.Variable(values['shapecode'].detach().clone(), requires_grad=True)
        parameters += [values['shapecode']]

    if optimize_expression:
        values['expcode'] = torch.autograd.Variable(values['expcode'].detach().clone(), requires_grad=True)
        parameters += [values['expcode']]

    if optimize_pose:
        values['posecode'] = torch.autograd.Variable(values['posecode'].detach().clone(), requires_grad=True)
        parameters += [values['posecode']]

    if optimize_texture:
        values['texcode'] = torch.autograd.Variable(values['texcode'].detach().clone(), requires_grad=True)
        parameters += [values['texcode']]

    if optimize_cam:
        values['cam'] = torch.autograd.Variable(values['cam'].detach().clone(), requires_grad=True)
        parameters += [values['cam']]

    if optimize_light:
        values['lightcode'] = torch.autograd.Variable(values['lightcode'].detach().clone(), requires_grad=True)
        parameters += [values['lightcode']]

    if len(parameters) == 0:
        raise RuntimeError("No parameters are being optimized")

    # optimizer = torch.optim.Adam(parameters, lr=0.01)
    # optimizer = torch.optim.SGD(parameters, lr=0.001)
    optimizer = torch.optim.LBFGS(parameters, lr=0.01)
    if not isinstance(loss_to_use, str):
        def criterion(vals):
            return loss_to_use(vals["predicted_detailed_image"])
        # criterion = loss_to_use
    else:
        criterion = None

    best_loss = 99999999999999.
    eps = 1e-6
    verbose = True
    visualize = False
    stopping_condition = False
    # max_iters = 1000
    # max_iters = 100
    max_iters = 10
    for i in range(max_iters):

        def closure():
            optimizer.zero_grad()

            values_ = deca._decode(values, training=False)
            losses_and_metrics = deca.compute_loss(values_, training=False)

            if criterion is None:
                loss = losses_and_metrics[loss_to_use]
            else:
                loss = criterion(values_)
            loss.backward(retain_graph=True)
            return loss

        optimizer.zero_grad()

        values = deca._decode(values, training=False)
        losses_and_metrics = deca.compute_loss(values, training=False)

        if criterion is None:
            loss = losses_and_metrics[loss_to_use]
        else:
            loss = criterion(values)
        loss.backward(retain_graph=True)
        # closure()

        optimizer.step(closure=closure)

        if visualize:
            uv_detail_normals = None
            if 'uv_detail_normals' in values.keys():
                uv_detail_normals = values['uv_detail_normals']
            visualizations, grid_image = deca._visualization_checkpoint(values['verts'],
                                                                        values['trans_verts'],
                                                                        values['ops'],
                                                                        uv_detail_normals,
                                                                        values,
                                                                        0,
                                                                        "",
                                                                        "",
                                                                        save=False)
            vis_dict = deca._log_visualizations("", visualizations, values, 0, indices=0)
            plot_results(vis_dict, f"Iter {i:04d}, loss={loss:.10f}", detail=True)


        if verbose:
            print(f"Iter {i:04d}, loss={loss:.10f}")
        if loss < eps:
            stopping_condition = True
            break

        if best_loss > loss.item():
            best_iter = i
            best_loss = loss.item()
            best_values = {}
            for key, val in values.items():
                try:
                    best_values[key] = val.detach().clone()
                except Exception as e:
                    try:
                        best_values[key] = copy.deepcopy(val)
                    except Exception as e2:
                        # print(val)
                        # try:
                        best_values[key] = {k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                                        for k, v in values[key].items() }
                        # except Exception as e3:

            # best_values = { key : val.detach().clone() if torch.is_tensor(val) else copy.deepcopy(val)
            #                 for key, val in values.items() }



    if not stopping_condition:
        print(f"[WARNING] Optimization terminated after max number of iterations, not becaused it reached the desired tolerance")


    values = deca._decode(best_values, training=False)
    losses_and_metrics = deca.compute_loss(values, training=False)
    uv_detail_normals = None
    if 'uv_detail_normals' in values.keys():
        uv_detail_normals = values['uv_detail_normals']
    visualizations, grid_image = deca._visualization_checkpoint(values['verts'],
                                                                values['trans_verts'],
                                                                values['ops'],
                                                                uv_detail_normals,
                                                                values,
                                                                0,
                                                                "",
                                                                "",
                                                                save=False)
    vis_dict = deca._log_visualizations("", visualizations, values, 0, indices=0)
    plot_results(vis_dict, f"Best iter {best_iter:04d}, loss={best_loss:.10f}", detail=True)


def main():

    target_image = "~/Workspace/mount/scratch/rdanecek/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/VA_Set/" \
                   "detections/Train_Set/82-25-854x480/002400_000.png"
    loss_to_use = TargetEmotionCriterion(target_image)
    path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # run_name = '2021_03_01_11-31-57_VA_Set_videos_Train_Set_119-30-848x480.mp4_EmoNetLossB_F1F2VAECw-0.00150_CoSegmentGT_DeSegmentRend'
    run_name = '2021_03_08_22-30-55_VA_Set_videos_Train_Set_119-30-848x480.mp4CoPhotoCoLMK_IDW-0.15_Aug_early'
    stage = 'detail'
    relative_to_path = '/ps/scratch/'
    replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
    deca, dm = load_deca_and_data(path_to_models, run_name, stage, relative_to_path, replace_root_path)
    deca.deca.config.train_coarse = True
    deca.deca.config.mode = DecaMode.DETAIL
    # deca.deca.config.mode = DecaMode.COARSE
    image_index = 390 * 4 + 1
    values, visdict = test(deca, dm, image_index)
    print(values.keys())


    plot_results(visdict, "title")

    optimize(deca, values, loss_to_use=loss_to_use)



if __name__ == "__main__":
    main()
