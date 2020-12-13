from models.FLAME import FLAME
import torch
import numpy as np
import os
from applications.FLAME.config import get_config
from tqdm import tqdm

def fit_FLAME_to_registered(flame : FLAME,
                            target_mesh_verts,
                            # v_template = None,
                            unpose=False,
                            max_iters=10000,
                            eps=1e-7,
                            fit_shape=True,
                            fit_expression=True,
                            fit_pose=True,
                            fit_neck=True,
                            fit_eyes=True,
                            fit_translations=True,
                            visualize=True,
                            verbose=False):

    # personalize FLAME template
    # if v_template is not None:
    #     flame.flame_model.v_template = torch.Tensor(v_template)

    shape_params = torch.zeros(1, 300)

    expression_params = torch.zeros(1, 100)

    pose_params = torch.zeros(1, 6)

    neck_pose = torch.zeros(1, 3)

    eye_pose = torch.zeros(1, 6)

    transl = torch.zeros(1, 3)


    # for param in parameters:
    #     param.requires_grad = True

    # freeze FLAME
    for param in flame.parameters():
        param.requires_grad = False



    if not isinstance(target_mesh_verts, list):
        target_mesh_verts = [target_mesh_verts, ]

    final_verts = []
    unposed_verts = []
    shape = []
    expr = []
    pose = []
    neck = []
    eye = []
    trans = []


    for mesh_idx in tqdm(range(len(target_mesh_verts))):
        target_verts = target_mesh_verts[mesh_idx]
        target_vertices = torch.Tensor(target_verts).view(1, -1, 3)

        shape_params = torch.autograd.Variable(shape_params.detach().clone(), requires_grad=fit_shape)
        expression_params = torch.autograd.Variable(expression_params.detach().clone(), requires_grad=fit_expression)
        pose_params = torch.autograd.Variable(pose_params.detach().clone(), requires_grad=fit_pose)
        neck_pose = torch.autograd.Variable(neck_pose.detach().clone(), requires_grad=fit_neck)
        eye_pose = torch.autograd.Variable(eye_pose.detach().clone(), requires_grad=fit_eyes)
        transl = torch.autograd.Variable(transl.detach().clone(), requires_grad=fit_translations)

        parameters = [
            shape_params,
            expression_params,
            pose_params,
            neck_pose,
            eye_pose,
            transl,
        ]

        # optimizer = torch.optim.SGD(parameters, lr=1)
        optimizer = torch.optim.Adam(parameters, lr=0.1)
        criterion = torch.nn.MSELoss()

        if verbose:
            print("Optimizing for mesh %.6d" % mesh_idx)
        if visualize:
            import pyvista as pv
            import pyvistaqt as pvqt

            target_mesh = pv.PolyData(target_verts[0], np.hstack([np.ones(shape=(flame.faces.shape[0],1), dtype=np.int32 )*3, flame.faces]))

            pl = pvqt.BackgroundPlotter(auto_update=True)
            pl.add_mesh(target_mesh, opacity=0.5)
            final_mesh = target_mesh.copy(deep=True)
            pl.add_mesh(final_mesh)
            pl.show()

            text = pl.add_text("Iter: %5d" % 0)

        stopping_condition = False

        previous_loss = 99999999999999999
        previous_params = previous_loss

        # previous_verts = None
        # previous_shape = None
        # previous_expr = None
        # previous_pose = None
        # previous_neck = None
        # previous_eye = None
        # previous_transl = None
        previous_verts = flame.v_template.detach().clone()
        previous_shape = shape_params.detach()
        previous_expr = expression_params.detach()
        previous_pose = pose_params.detach()
        previous_neck = neck_pose.detach()
        previous_eye = eye_pose.detach()
        previous_transl = transl.detach()


        for i in range(max_iters):
            optimizer.zero_grad()
            vertices, landmarks = flame.forward(shape_params=shape_params, expression_params=expression_params,
                                                pose_params=pose_params, neck_pose=neck_pose, eye_pose=eye_pose, transl=transl)
            # mse = (vertices - target_vertices).square().mean()

            if visualize:
                final_mesh.points = vertices[0].detach().numpy()
                text.SetText(2, "Iter: %5d" % (i+1))

            loss = criterion(vertices, target_vertices)
            if verbose:
                print("Iter %.4d, loss=%.10f" % (i, loss))
            if loss < eps:
                stopping_condition = True
                break

            loss.backward()
            optimizer.step()

            #
            if previous_loss > loss.item():
                previous_verts[...] = vertices
                previous_shape[...] = shape_params
                previous_expr[...] = expression_params
                previous_pose[...] = pose_params
                previous_neck[...] = neck_pose
                previous_eye[...] = eye_pose
                previous_transl[...] = transl

                previous_loss = loss.item()
        #

        if not stopping_condition:
            print("[WARNING] Mesh %d did not hit the stopping conditiong but ran ouf of iterations. Iter %.5d, loss=%.10f" % (mesh_idx, i, loss))

        # if verbose:
        print("Mesh %.6d finished. Iter %.4d, loss=%.10f" % (mesh_idx, i, loss))

        vertices = previous_verts
        shape_params = previous_shape
        expression_params = previous_expr
        pose_params = previous_pose
        neck_pose = previous_neck
        eye_pose = previous_eye
        transl = previous_transl


        final_verts += [np.copy(vertices[0].detach().numpy())]
        shape += [shape_params[0].detach().numpy()]
        expr += [expression_params[0].detach().numpy()]
        pose += [pose_params[0].detach().numpy() ]
        neck += [neck_pose[0].detach().numpy()]
        eye += [eye_pose[0].detach().numpy()]
        trans += [transl[0].detach().numpy()]

        if unpose:
            # shape_params = None, expression_params = None, pose_params = None, neck_pose = None, eye_pose = None, transl = None):
            unposed_vertices, _ = flame.forward(shape_params=shape_params,
                                                expression_params=expression_params,
                                                pose_params=torch.zeros_like(pose_params),
                                                neck_pose=torch.zeros_like(neck_pose),
                                                eye_pose=torch.zeros_like(eye_pose),
                                                transl=torch.zeros_like(transl))

            unposed_verts += [np.copy(unposed_vertices[0].detach().numpy())]

        if visualize:
            pl.close()

    return final_verts, shape, expr, pose, neck, eye, trans, unposed_verts





def load_FLAME(gender : str,
               shape_params=300,
               expression_params = 100,
               use_3d_trans= True,
               use_face_contour= False,
               batch_size=1,
               v_template = None) -> FLAME:
    gender = gender.lower()
    path_to_models = os.path.join(os.path.dirname(__file__), "..", "..", "trained_models", "FLAME")

    cfg, unknown = get_config()

    cfg.static_landmark_embedding_path = os.path.join(path_to_models, 'flame_static_embedding.pkl')
    cfg.dynamic_landmark_embedding_path = os.path.join(path_to_models, 'flame_dynamic_embedding.npy')
    cfg.use_face_contour = use_face_contour
    cfg.use_3D_translation = use_3d_trans
    cfg.batch_size = batch_size
    cfg.shape_params = shape_params
    cfg.flame_expression_params = expression_params

    if gender == 'male':
        cfg.flame_model_path = os.path.join(path_to_models, 'male_model.pkl')
    elif gender == 'female':
        cfg.flame_model_path = os.path.join(path_to_models, 'female_model.pkl')
    elif gender == 'neutral':
        cfg.flame_model_path = os.path.join(path_to_models, 'generic_model.pkl')
    else:
        raise ValueError("Invalid model specifier for FLAME: '%s'" % gender)
    return FLAME(cfg, v_template=v_template)


def main():
    from datasets.MeshDataset import EmoSpeechDataModule


    # flame_male = load_FLAME('male')
    # flame_female = load_FLAME('female')
    expression_params = 100

    root_dir = "/home/rdanecek/Workspace/mount/project/emotionalspeech/EmotionalSpeech/"
    processed_dir = "/home/rdanecek/Workspace/mount/scratch/rdanecek/EmotionalSpeech/"
    subfolder = "processed_2020_Dec_09_00-30-18"
    dm = EmoSpeechDataModule(root_dir, processed_dir, subfolder)
    dm.prepare_data()
    # # dm.setup()
    #
    # fitted_vertex_array = np.memmap(dm.fitted_vertex_array_path, dtype=np.float32, mode='r+',
    #                               shape=(dm.num_samples, 3 * dm.num_verts))
    # expr_array = np.memmap(dm.expr_array_path, dtype=np.float32, mode='r+',
    #                               shape=(dm.num_samples, expression_params))
    #
    # pose_array = np.memmap(dm.pose_array_path, dtype=np.float32, mode='r+',
    #                               shape=(dm.num_samples, 6))
    #
    # neck_array = np.memmap(dm.neck_array_path, dtype=np.float32, mode='r+',
    #                               shape=(dm.num_samples, 3))
    #
    # eye_array = np.memmap(dm.eye_array_path, dtype=np.float32, mode='r+',
    #                               shape=(dm.num_samples, 6))
    #
    # translation_array = np.memmap(dm.translation_array_path, dtype=np.loat32, mode='r+',
    #                       shape=(dm.num_samples, 3))


    for id, mesh in enumerate(dm.subjects_templates):
        # verts = torch.from_numpy(mesh.points)

        print("Beginning to process mesh %d" % id)
        frames = np.where(dm.identity_array == id)[0]
        frames = frames[:100]
        # frames = frames[99:100]

        flame = load_FLAME('neutral', expression_params=expression_params, v_template=mesh.points)

        verts = dm.vertex_array[frames, ...].reshape(frames.size, -1, 3)
        target_verts = np.split(verts, verts.shape[0], 0)

        fitted_verts, shape, expr, pose, neck, eye, trans = fit_FLAME_to_registered(flame, target_verts, fit_shape=False, verbose=False, visualize=True)

        # fitted_vertex_array[frames, ...] = np.reshape(fitted_verts, newshape=(frames.size, -1, 3))
        # expr_array[frames, ...] = expr
        # pose_array[frames, ...] = pose
        # neck_array[frames, ...] = neck
        # eye_array[frames, ...] = eye
        # translation_array[frames, ...] = trans

        print("Finished processing mesh %d" % id)



    print("YEAH")


if __name__ == "__main__":
    main()
    pass
