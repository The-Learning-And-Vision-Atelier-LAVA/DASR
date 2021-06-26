import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', type=bool, default=False,
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='D:/LongguangWang/Data',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DF2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set14',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-3450/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=48,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Degradation specifications
parser.add_argument('--blur_kernel', type=int, default=21,
                    help='size of blur kernels')
parser.add_argument('--blur_type', type=str, default='iso_gaussian',
                    help='blur types (iso_gaussian | aniso_gaussian)')
parser.add_argument('--mode', type=str, default='bicubic',
                    help='downsampler (bicubic | s-fold)')
parser.add_argument('--noise', type=float, default=0.0,
                    help='noise level')
## isotropic Gaussian blur
parser.add_argument('--sig_min', type=float, default=0.2,
                    help='minimum sigma of isotropic Gaussian blurs')
parser.add_argument('--sig_max', type=float, default=4.0,
                    help='maximum sigma of isotropic Gaussian blurs')
parser.add_argument('--sig', type=float, default=4.0,
                    help='specific sigma of isotropic Gaussian blurs')
## anisotropic Gaussian blur
parser.add_argument('--lambda_min', type=float, default=0.2,
                    help='minimum value for the eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--lambda_max', type=float, default=4.0,
                    help='maximum value for the eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--lambda_1', type=float, default=0.2,
                    help='one eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--lambda_2', type=float, default=4.0,
                    help='another eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--theta', type=float, default=0.0,
                    help='rotation angle of anisotropic Gaussian blurs [0, 180]')


# Model specifications
parser.add_argument('--model', default='blindsr',
                    help='model name')
parser.add_argument('--pre_train', type=str, default= '.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs_encoder', type=int, default=100,
                    help='number of epochs to train the degradation encoder')
parser.add_argument('--epochs_sr', type=int, default=500,
                    help='number of epochs to train the whole network')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr_encoder', type=float, default=1e-3,
                    help='learning rate to train the degradation encoder')
parser.add_argument('--lr_sr', type=float, default=1e-4,
                    help='learning rate to train the whole network')
parser.add_argument('--lr_decay_encoder', type=int, default=60,
                    help='learning rate decay per N epochs')
parser.add_argument('--lr_decay_sr', type=int, default=125,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma_encoder', type=float, default=0.1,
                    help='learning rate decay factor for step decay')
parser.add_argument('--gamma_sr', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from the snapshot, and the start_epoch')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='blindsr',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=200,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', default=False,
                    help='save output results')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: float(x), args.scale.split('+')))

