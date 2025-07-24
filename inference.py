import os.path
import torch
import numpy as np
import utils.func as func
import argparse


def configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='s2r_it_20k.pt')
    parser.add_argument('--data', type=str, help='whether to show result')
    parser.add_argument('--save-filename', type=str)
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--infer-size', type=int, default=None, nargs='+',
                        help='Size of the input. It should be a triple integer of (timeline, crossline, inline). '
                             'If None, the whole cube will be the input.')
    parser.add_argument('--show', action='store_true', help='whether to show result')

    return parser.parse_args()



def main(args):
    # prepare data
    
    remain_prob = False

    seismic = np.load(args.data)  # [t,h,w]

    print('seismic data size == ', seismic.shape)
    # prepare model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seg_model = torch.jit.load(args.model, map_location=device)
    seg_model.eval()
    
    pred = func.model_infer(model=seg_model, img=seismic, infer_size=args.infer_size, device=device)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    save_path = os.path.join(args.save_dir, args.save_filename)
    np.save(save_path, pred)
    print(r'Results has been saved at {}.npy'.format(save_path))
    
    if args.show: 
        func.plot_result(seismic, pred, remain_prob)


if __name__ == '__main__':
    args = configs()
    main(args)
