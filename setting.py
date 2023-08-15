'''
Configs for training & testing
'''

import argparse
import torch



def Data_Setting(traindataname, testdataname, modal, m, n):

    root = r'C:/LYL_data'
    shape = [91, 109, 91]  # size

    if testdataname == 'ADNI' or testdataname == 'AIBL':
        Classes = ['NC', 'SMC', 'sMCI', 'pMCI', 'AD']
        Class_size = [231, 81, 197, 108, 199]
        root = r'{}/AD'.format(root)

    elif testdataname == 'ABIDE':
        Classes = ['ABIDE-C', 'ABIDE-A']
        Class_size = [455, 413]

    elif testdataname == 'NIFD':
        Classes = ['NIFD-C', 'NIFD-N']
        Class_size = [92, 176]

    elif testdataname == 'PPMI':
        Classes = ['PPMI-HC', 'PPMI-PD-352']
        Class_size = [152, 449]

    elif testdataname == 'PSCI':
        Classes = ['non-PSCI', 'PSCI']
        Class_size = [80, 32]

    classes = [Classes[m], Classes[n]]


    trainroot = r'{}/{}'.format(root, traindataname)
    testroot = r'{}/{}'.format(root, testdataname)
    # if len(modal)==1:
    #     trainroot = r'{}/{}/{}'.format(root, traindataname, modal)
    #     testroot = r'{}/{}/{}'.format(root, testdataname, modal)
    # elif len(modal)==2:
    #     trainroot = r'{}/{}'.format(root, traindataname)
    #     testroot = r'{}/{}'.format(root, testdataname)

    sample_class_count = torch.Tensor([Class_size[m], Class_size[n]])
    class_weight = sample_class_count.float() / (Class_size[m] + Class_size[n])
    class_weight = 1. - class_weight

    return classes, class_weight, trainroot, testroot, shape


def parse_opts_single(
        traindataname='ADNI', # ADNI, PPMI, ABIDE, NIFD
        testdataname='ADNI',
        class1=0, #['NC', 'SMC', 'sMCI', 'pMCI', 'AD']
        class2=4,
        modal=['MRI'] # MRI, PET
        ):

    classes, class_weight, trainroot, testroot, shape = Data_Setting(
        traindataname, testdataname, modal, class1, class2)

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_cla_classes', default=2, type=int, help="Number of segmentation classes")
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')# set to 0.001 when finetune
    parser.add_argument('--num_workers', default=4, type=int, help='Number of jobs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument('--phase', default='train', type=str, help='Phase of train or test')
    # parser.add_argument('--save_intervals', default=50, type=int, help='Interation for saving model')
    parser.add_argument('--n_epochs', default=500, type=int, help='Number of total epochs to run')

    parser.add_argument('--modal', default=modal[0], type=str, help='MRI|PET')
    parser.add_argument('--classes', default=classes, type=list, help="['NC','SMC','sMCI','pMCI','AD']")
    parser.add_argument('--root_traindata', default=trainroot, type=str, help='Root directory path of loaddata')
    parser.add_argument('--root_testdata', default=testroot, type=str, help='Root directory path of loaddata')
    parser.add_argument('--w', default=class_weight, type=list)
    # parser.add_argument('--img_list', default='./loaddata/train.txt', type=str, help='Path for image list file')

    parser.add_argument('--input_D', default=shape[0], type=int, help='Input size of depth')
    parser.add_argument('--input_H', default=shape[1], type=int, help='Input size of height')
    parser.add_argument('--input_W', default=shape[2], type=int, help='Input size of width')

    parser.add_argument('--pretrain_path', default=False, type=str,help='Path for pretrained model.')  # default='pretrain/resnet_34.pth',
    # parser.add_argument('--pretrain_path', default='./trails/pretrain/resnet_18.pth', type=str, help='Path for pretrained model.') # default='pretrain/resnet_34.pth',
    # parser.add_argument('--new_layer_names', default=['conv_seg'], type=list, help='New layer except for backbone')#default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
    parser.add_argument('--gpu_id', nargs='+', type=int, help='Gpu id lists')
    parser.add_argument('--model', default='Net_v2', type=str, help='(ConvMix_1 | Convmix_256 | Net_v2 | ConvMix_MRF_1 | ConvMix_MRF_CAG_1 ')
    parser.add_argument('--model_patch', default=5, type=int, help='Depth of net (3 | 5 | 7 | 9 | 11)')
    parser.add_argument('--model_depth', default=5, type=int, help='Depth of net (5 | 7 | 9 | 11)')
    parser.add_argument('--model_dim', default=1024, type=int)

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    # parser.add_argument('--ci_test', action='store_true', help='If true, ci testing is used.')

    # parser.add_argument('--save_path', default=r'./trails/model_{}/{} {} vs {}/{}({},{}) w={}'
    #                     .format(modal, testdataname, classes[0], classes[1],args.model, ), type=str, help='Path for resume model.')

    args = parser.parse_args()

    args.save_path = r'./trails/model_{}/{} {}/{}({},{}) w={}'\
        .format(args.modal,
                testdataname,  classes,
                args.model, args.model_patch, args.model_depth, args.w)

    return args




def parse_opts_dual(
        traindataname='ADNI',  # ADNI, PPMI, ABIDE, NIFD
        testdataname='ADNI',
        class1=0,  # ['NC', 'SMC', 'sMCI', 'pMCI', 'AD']
        class2=4,
        modal=['MRI', 'PET']  # MRI, PET
):
    classes, class_weight, trainroot, testroot, shape = Data_Setting(
        traindataname, testdataname, modal, class1, class2)

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_cla_classes', default=2, type=int, help="Number of segmentation classes")
    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')  # set to 0.001 when finetune
    parser.add_argument('--num_workers', default=4, type=int, help='Number of jobs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument('--phase', default='train', type=str, help='Phase of train or test')
    # parser.add_argument('--save_intervals', default=50, type=int, help='Interation for saving model')
    parser.add_argument('--n_epochs', default=500, type=int, help='Number of total epochs to run')

    parser.add_argument('--modal', default=modal, type=str, help='MRI|PET')
    parser.add_argument('--classes', default=classes, type=list, help="['NC','SMC','sMCI','pMCI','AD']")
    parser.add_argument('--root_traindata', default=trainroot, type=str, help='Root directory path of loaddata')
    parser.add_argument('--root_testdata', default=testroot, type=str, help='Root directory path of loaddata')
    parser.add_argument('--w', default=class_weight, type=list)
    # parser.add_argument('--img_list', default='./loaddata/train.txt', type=str, help='Path for image list file')

    parser.add_argument('--input_D', default=shape[0], type=int, help='Input size of depth')
    parser.add_argument('--input_H', default=shape[1], type=int, help='Input size of height')
    parser.add_argument('--input_W', default=shape[2], type=int, help='Input size of width')

    parser.add_argument('--pretrain_path', default=False, type=str,
                        help='Path for pretrained model.')  # default='pretrain/resnet_34.pth',
    # parser.add_argument('--pretrain_path', default='./trails/pretrain/resnet_18.pth', type=str, help='Path for pretrained model.') # default='pretrain/resnet_34.pth',
    # parser.add_argument('--new_layer_names', default=['conv_seg'], type=list, help='New layer except for backbone')#default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
    parser.add_argument('--gpu_id', nargs='+', type=int, help='Gpu id lists')
    parser.add_argument('--model', default='ConvMix_SWE_CWE_2', type=str,
                        help='(ConvMix_1 | Convmix_256 | Net_v2 | ConvMix_MRF_1 | ConvMix_MRF_CAG_1 ')
    parser.add_argument('--model_patch', default=5, type=int, help='Depth of net (3 | 5 | 7 | 9 | 11)')
    parser.add_argument('--model_depth', default=5, type=int, help='Depth of net (5 | 7 | 9 | 11)')
    parser.add_argument('--model_dim', default=1024, type=int)

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    # parser.add_argument('--ci_test', action='store_true', help='If true, ci testing is used.')

    # parser.add_argument('--save_path', default=r'./trails/model_{}/{} {} vs {}/{}({},{}) w={}'
    #                     .format(modal, testdataname, classes[0], classes[1],args.model, ), type=str, help='Path for resume model.')

    args = parser.parse_args()

    args.save_path = r'./trails/model_{}+{}/{} {}/{}({},{}) w={}' \
        .format(args.modal[0], args.modal[1],
                testdataname, classes,
                args.model, args.model_patch, args.model_depth, args.w)

    return args