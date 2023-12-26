import torch
import os,sys
get_path = os.path.dirname(__file__)
sys.path.append(get_path)
current_path = os.path.dirname(__file__).split('/')
import torch
from ViT import U_Vit,ModelWapper4ODE
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms.functional as transF
import yaml
import argparse
from torchdiffeq import odeint_adjoint as odeint
from rflow_loader import pil_loader
def parse_args():
    parser = argparse.ArgumentParser(description="Your program description here")
    parser.add_argument("--input_image", type=str, help="the image to start with")
    parser.add_argument("--target_image", type=str, help="the image to be interpolated to")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for computation, default is 'cuda:0'")
    parser.add_argument("--load", type=str, default="last", help="Specify the epoch to load, either 'best' or 'last'")
    parser.add_argument("--SavedDir", type=str, help="Directory to save images")
    parser.add_argument("--ExpConfig", type=str, help="Path to the YAML file of your experiments")
    parser.add_argument("--rtol", type=float, default=0.0001, help="Acceptable relative error per step")
    parser.add_argument("--mix_depth", type=float, default=-0.02, help="Where to mix the noise, 0.0 means mixing at the complete noises, 1.0 means mixing at original images")
    parser.add_argument("--spherical", type=bool, default=True, help="Whether to use spherical interpolation")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    load = args.load
    SavedDir = args.SavedDir
    ExpDir = args.ExpConfig
    rtol = args.rtol
    path1 = args.input_image
    path2 = args.target_image
    mix_depth = args.mix_depth
    spherical = args.spherical

    with open(ExpDir,'r') as f:
        training_parameters = yaml.full_load(f)
    TrainDir = training_parameters['dataStorage']
    try:
        os.mkdir(SavedDir)
    except:
        pass
    modelName = training_parameters["framework"]
    image_size = training_parameters["image_size"]
    patch_size = training_parameters["patch_size"] 
    embed_dim = training_parameters["embed_dim"] 
    depth = training_parameters["depth"] 
    head = training_parameters["head"]    
    model = U_Vit(img_size=image_size,patch_size=patch_size,embed_dim=embed_dim,depth=depth,num_heads=head)

    if load == 'best':
        initializing = os.path.join(os.path.dirname(ExpDir),'bestloss.pkl')
        state = torch.load(initializing,map_location=device)
    elif load == 'last':
        initializing = os.path.join(os.path.dirname(ExpDir),'lastepoch.pkl')
        state = torch.load(initializing,map_location=device)
        state = state['state_dict']
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state,prefix='module.')

    model.load_state_dict(state,strict=True)
    img_size = model.img_size
    model = ModelWapper4ODE(model)
    model.to(device)
    model.eval()
    
    img1 = pil_loader(path1)
    img1 = transF.to_tensor(img1)
    img1 = transF.resize(img1,img_size,antialias=False)
    img1 = img1*2 -1


    img2 = pil_loader(path2)
    img2 = transF.to_tensor(img2)
    img2 = transF.resize(img2,img_size,antialias=False)
    img2 = img2*2 -1
    img = torch.stack([img1,img2],dim=0).float().to(device)


    noise = odeint(model,img,torch.tensor([1.,mix_depth],device=device),rtol=rtol)[1]
    noise = torch.clip_(noise,-1,1)
    noise1,noise2 = noise

    intermidiate = []
    # spherical linear interpolation between noise1 and noise2 with alpha in [1,0.9,0.75,0.5,0.25,0.1,0.0]
    for alpha in [1,0.9,0.75,0.5,0.25,0.1,0.0]:
        if spherical:
            normed_1 = noise1/torch.norm(noise1,dim=0,keepdim=True)
            normed_2 = noise2/torch.norm(noise2,dim=0,keepdim=True)
            angle = torch.acos(torch.sum(normed_1 * normed_2,dim=0,keepdim=True))
            sin_angle = torch.sin(angle)
            intermidiate.append((torch.sin(alpha*angle)/sin_angle)*noise1 + (torch.sin((1-alpha)*angle)/sin_angle)*noise2)
        else:
            intermidiate.append(alpha*noise1 + ((1-alpha**2)**0.5)*noise2)
    intermidiate = torch.stack(intermidiate,dim=0)
    intermidiate = torch.clip(intermidiate,-1,1)
    img = odeint(model,intermidiate,torch.tensor([mix_depth,1.],device=device),rtol=rtol)
    img = torch.clip_(img[1],-1,1)
    img = (img.cpu()+1)/2
    fig = plt.figure(figsize=(img_size[0]*7/8,img_size[1]/8),dpi=64)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1, 7),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    for ax, im in zip(grid,img):
        # Iterating over the grid returns the Axes.
        ax.imshow(transF.to_pil_image(im))
        #ax.text(0,8,str(l),fontsize=60, color='blue')
    plt.savefig(os.path.join(SavedDir,"img_interpolation.png"),bbox_inches='tight')
    plt.close()


