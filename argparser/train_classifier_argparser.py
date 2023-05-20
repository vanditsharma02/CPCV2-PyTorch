import argparse
import torch

def argparser():
    parser = argparse.ArgumentParser(description="Training Classifier")

    # optional
    parser.add_argument('--dataset',          type=str,   metavar='', default="stl10",    help="Dataset to use (stl10, cifar10, cifar100)")
    parser.add_argument('--train_size',       type=int,   metavar='', default=5000,       help="Set the size of the training data - default STL10")
    parser.add_argument('--epochs',           type=int,   metavar='', default=110,        help="Number of Epochs for Training")
    parser.add_argument('--sched_step_size',  type=int,   metavar='', default=100,        help="Schedular Step Size")
    parser.add_argument('--sched_milestones', type=str,   metavar='', default="",         help="For using optimizer with MultiStepLR - Takes a string of comma seperated milestones '50,100,150'")
    parser.add_argument('--num_workers',      type=int,   metavar='', default=1,          help="Number of workers to be used in dataloader")
    parser.add_argument('--batch_size',       type=int,   metavar='', default=100,        help="Batch Size")
    parser.add_argument('--lr',               type=float, metavar='', default=0.1,        help="Learning Rate")
    parser.add_argument('--lr_gamma',         type=float, metavar='', default=0.1,        help="Gamma value for scheduler, i.e. how much to multiply by")
    parser.add_argument('--crop',             type=str,   metavar='', default="0-0",      help="CropSize-Padding (i.e. 64-2 would crop to 64 pixels with 2 pixels padding)")
    parser.add_argument('--image_resize',     type=int,   metavar='', default=0,          help="If changed, 'after cropping' the image will be resized to the given value ")
    parser.add_argument('--encoder',          type=str,   metavar='', default="resnet14", help="Which encoder to use (resnet14/18/28/34/41/50/92/101/143/152, wideresnet-depth-width, mobilenetV2)")
    parser.add_argument('--norm',             type=str,   metavar='', default="none",     help="What normalisation layer to use (none, batch, layer)")
    parser.add_argument('--grid_size',        type=int,   metavar='', default=7,          help="Size of the grid of patches that the image is broken down to")
    parser.add_argument('--pred_directions',  type=int,   metavar='', default=1,          help="Number of Directions that was used in CPC training")
    parser.add_argument('--test_interval',    type=int,   metavar='', default=1,          help="Interval of epochs to test at")
    parser.add_argument('--model_num',        type=str,   metavar='', default="",         help="Number of Epochs that CPC Encoder was trained for",)
    
    parser.add_argument('--fully_supervised', action='store_true',                        help="When set will train a fully supeverised model")
    parser.add_argument('--sched_plateau',    action='store_true',                        help="Use ReduceLROnPlateau lr scheduler")
    parser.add_argument('--download_dataset', action='store_true',                        help="Download the chosen dataset")
    parser.add_argument('--patch_aug',        action='store_true',                        help="Apply patch-based data augmentation as in CPC V2")
    parser.add_argument('--cpc_patch_aug',    action='store_true',                        help="Whether unsupervised training used patch-based data augmentation as in CPC V2")
    parser.add_argument('--gray',             action='store_true',                        help="Convert to grayscale")
    parser.add_argument('--sgd',              action='store_true',                        help="Use SGD instead of ADAM")

    parser.add_argument('--t1',               type=str,   metavar='', default='',         help='1st transformation applied when training CPC model')
    parser.add_argument('--t2',               type=str,   metavar='', default='',         help='2nd transformation applied when training CPC model')
    args = parser.parse_args()

    # Model number must be set when training CPC model
    if not args.fully_supervised:
        if args.model_num == "":
            raise Exception("Model number must be set when training CPC model")

    # Add to args given the input choices
    if args.dataset == "stl10":
        args.num_classes = 10
    elif args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    else:
        raise Exception("Invalid Dataset Input")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check encoder choice
    if args.encoder not in ("resnet14", "resnet18", "resnet28", 
                            "resnet34", "resnet41", "resnet50", 
                            "resnet92", "resent101", "resnet143", 
                            "resnet152", "mobilenetV2"
                            ) and args.encoder[:10] != "wideresnet":
        raise Exception("Invalid Encoder Input")

    # Change learning rate for fully supervised if it wasn't changed by user
    if args.fully_supervised and args.lr == 0.1:
        print("For fully supervised training lr has been changed to 1e-3")
        args.lr = 1e-3

    # Extract crop data
    if args.crop == "0-0":
        if args.dataset == "stl10":
            args.crop = "64-0"
        elif args.dataset in ("cifar10", "cifar100"):
            args.crop = "32-4" 
    crop_parameters = args.crop.split("-")
    args.crop_size = int(crop_parameters[0])
    args.padding = int(crop_parameters[1])

    return args
