import os
import torch
from torch import nn
from model.Allmodels import *


def generate_model(opt):
    assert opt.model in [
        'ConvMix_1',
        'Convmix_256', # for without_skull_image
        'Net_v2',
        'ConvMix_MRF_1',
        'ConvMix_MRF_CAG_1',
        'ConvMix_SWE_CWE_2',
        'ConvMix_MRF_CAG_SWE_CWE_2'
    ]

    if opt.model == 'ConvMix_1':
        # assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        model = ConvMix_1(
            dim=opt.model_dim,
            patch_size=opt.model_patch,
            depth=opt.model_depth,
            n_classes=opt.n_cla_classes)
    elif opt.model == 'Convmix_256':
        model = Convmix_256(
            dim=opt.model_dim,
            patch_size=opt.model_patch,
            kernel_size=opt.model_patch,
            depth=opt.model_depth,
            n_classes=opt.n_cla_classes)
    elif opt.model == 'Net_v2':
        model = Net_v2(
            dim=opt.model_dim,
            patch_size=opt.model_patch,
            depth=opt.model_depth,
            n_classes=opt.n_cla_classes)
    elif opt.model == 'ConvMix_MRF_1':
        model = ConvMix_MRF_1(
            dim=opt.model_dim,
            patch_size=opt.model_patch,
            depth=opt.model_depth,
            convs_k=[3, 5, 7, 9],
            n_classes=opt.n_cla_classes)
    elif opt.model == 'ConvMix_MRF_CAG_1':
        model = ConvMix_MRF_CAG_1(
            dim=opt.model_dim,
            patch_size=opt.model_patch,
            depth=opt.model_depth,
            convs_k=[3, 5, 7, 9],
            n_classes=opt.n_cla_classes)
    elif opt.model == 'ConvMix_SWE_CWE_2':
        model = ConvMix_SWE_CWE_2(
            dim=opt.model_dim,
            patch_size=opt.model_patch,
            depth=opt.model_depth,
            n_classes=opt.n_cla_classes)
    elif opt.model == 'ConvMix_MRF_CAG_SWE_CWE_2':
        model = ConvMix_MRF_CAG_SWE_CWE_2(
            dim=opt.model_dim,
            patch_size=opt.model_patch,
            depth=opt.model_depth,
            conv_kernels=[3, 5, 7, 9],
            n_classes=opt.n_cla_classes)


    if not opt.no_cuda:
        # os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()