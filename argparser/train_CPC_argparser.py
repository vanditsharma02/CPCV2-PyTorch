import argparse
import torch

def argparser():
    parser = argparse.ArgumentParser(description="Training CPC")

    # optional
    parser.add_argument('--dataset',          type=str,   metavar='', default="stl10",    help="Dataset to use (stl10, cifar10, cifar100)")
    parser.add_argument('--unsupervised_size',type=str,   metavar='', default="complete", help="Set the size of the unsupervised data - default complete")
    parser.add_argument('--epochs',           type=int,   metavar='', default=300,        help="Number of Epochs for Training")
    parser.add_argument('--trained_epochs',   type=int,   metavar='', default=0,          help="Number of epochs already trained, will load from TrainedModels")
    parser.add_argument('--num_workers',      type=int,   metavar='', default=1,          help="Number of workers to be used in dataloader")
    parser.add_argument('--batch_size',       type=int,   metavar='', default=16,         help="Batch Size")
    parser.add_argument('--lr',               type=float, metavar='', default=2e-4,       help="Learning Rate")
    parser.add_argument('--crop',             type=str,   metavar='', default="0-0",      help="CropSize-Padding (i.e. 64-2 would crop to 64 pixels with 2 pixels padding)")
    parser.add_argument('--image_resize',     type=int,   metavar='', default=0,          help="If changed, 'after cropping' the image will be resized to the given value")
    parser.add_argument('--encoder',          type=str,   metavar='', default="resnet14", help="Which encoder to use (resnet18/34/50/101/152, wideresnet-depth-width, mobilenetV2)")
    parser.add_argument('--norm',             type=str,   metavar='', default="none",     help="What normalisation layer to use (none, batch, layer)")
    parser.add_argument('--grid_size',        type=int,   metavar='', default=7,          help="Size of the grid of patches that the image is broken down to")
    parser.add_argument('--pred_steps',       type=int,   metavar='', default=5,          help="Number of Predictions Steps")
    parser.add_argument('--pred_directions',  type=int,   metavar='', default=1,          help="Number of Directions that was used in CPC training")
    parser.add_argument('--neg_samples',      type=int,   metavar='', default=16,         help="Number of Negative Samples for InfoNCE Loss")
    parser.add_argument('--print_option',     type=int,   metavar='', default=0,          help="How results are displayed whilst training (0=tqdm, 1=interval statistics, other=End of Epoch only)")
    parser.add_argument('--print_interval',   type=int,   metavar='', default=500,        help="When print_option = 1, this determines how often to print")
    parser.add_argument('--model_name_ext',   type=str,   metavar='', default="",         help="Added to the end of the model name")
    
    parser.add_argument('--download_dataset', action='store_true',                        help="Download the chosen dataset")
    parser.add_argument('--patch_aug',        action='store_true',                        help="Apply path-based data augmentation as in CPC V2")
    parser.add_argument('--gray',             action='store_true',                        help="Convert to grayscale")

    args = parser.parse_args()

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

    # Check grid_size and pred_steps combination
    if args.pred_steps > args.grid_size - 2:
        raise Exception("To many predictions steps given the size of the grid")

    # Extract crop data
    if args.crop == "0-0":
        if args.dataset == "stl10":
            args.crop = "64-0"
        elif args.dataset in ("cifar10", "cifar100"):
            args.crop = "32-4" 
    crop_parameters = args.crop.split("-")
    args.crop_size = int(crop_parameters[0])
    args.padding = int(crop_parameters[1])

    args.fully_supervised = False

    return args
    
