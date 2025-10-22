
import argparse
import torch 
from datasets.liferefer_dataset import LifeReferDataset
from datasets.strefer_dataset import STReferDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def create_dataset(args, split):
    if args.dataset == 'liferefer':
        return LifeReferDataset(args, split)
    elif args.dataset == 'strefer':
        return STReferDataset(args, split)
    else:
        raise ValueError("Wrong Dataset")

from models.bdetr import BeaUTyDETR
# from .ap_helper import APCalculator, parse_predictions, parse_groundtruths
from models.losses import HungarianMatcher, SetCriterion, compute_hungarian_loss

def create_model(args):
    return BeaUTyDETR(
        args=args,
        num_class=args.max_lang_num,
        input_feature_dim=3,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        self_position_embedding='loc_learned',
        contrastive_align_loss=True,
        d_model=288,
        pointnet_ckpt=None,
        resnet_ckpt=None,
        self_attend=True,
        frame_num=args.frame_num,
        butd=args.butd
    )



parser = argparse.ArgumentParser('Set config')
args = parser.parse_args()
args.batch_size = 32 
args.butd = False 
args.dataset = 'liferefer'
args.debug = False
args.dynamic = True
args.epochs = 100
args.frame_num = 2
args.img_size = 384
args.lr = 0.0001
args.lr_backbone = 0.001
args.lr_step = [45, 80]
args.max_lang_num = 100
args.max_obj_num = 100
args.num_decoder_layers = 6
args.num_queries = 256
args.num_workers = 8
args.pretrain = 'weights/liferefer_weights.pth'
args.seed = 42
args.text_encoder_lr = 1e-05
args.val_epoch = 1
args.verbose_step = 10
args.warmup_epoch = -1
args.work_dir = 'outputs/debug'


test_dataset = create_dataset(args, 'test')
generator = torch.Generator()
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, generator=generator)

model = create_model(args)


model.load_state_dict(torch.load(args.pretrain, map_location='cpu')['model'], strict=True)

# model.cuda() 
model.to("cpu")


# Evaluate the model 
model.eval()
loss = 0
total_predict_boxes = []

####################
for batch in test_loader:
    break

# Print batch information
print("\n" + "="*80)
print("BATCH INFORMATION (before taking single sample)")
print("="*80)
if 'point_clouds' in batch and isinstance(batch['point_clouds'], torch.Tensor):
    print(f"Original batch point_clouds shape: {batch['point_clouds'].shape}")
    print(f"  Batch contains {batch['point_clouds'].shape[0]} samples")
    print(f"  Each sample has {batch['point_clouds'].shape[1]} points")
    print(f"  Each point has {batch['point_clouds'].shape[2]} features (XYZ + features)")
print("="*80 + "\n")

one_test_sample_from_batch = dict()
for key in batch:
    if isinstance(batch[key], torch.Tensor):
        one_test_sample_from_batch[key] = batch[key][:1,].cuda()
    else:
        one_test_sample_from_batch[key] = batch[key][:1]


model.cuda()
from pprint import pprint 
pprint(one_test_sample_from_batch.keys())

# Print original input shapes before model processing
print("\n" + "="*80)
print("ORIGINAL INPUT SHAPES BEFORE PREPROCESSING")
print("="*80)
for key, value in one_test_sample_from_batch.items():
    if isinstance(value, torch.Tensor):
        print(f"[{key}]: {value.shape} (dtype: {value.dtype}, device: {value.device})")
    else:
        print(f"[{key}]: {type(value).__name__}")
        
# Specifically check point clouds
if 'point_clouds' in one_test_sample_from_batch:
    pc = one_test_sample_from_batch['point_clouds']
    print(f"\nPoint Cloud Details:")
    print(f"  Shape: {pc.shape}")
    print(f"  Batch size: {pc.shape[0]}")
    if len(pc.shape) >= 3:
        print(f"  Number of points per sample: {pc.shape[1]}")
        print(f"  Number of features per point: {pc.shape[2]}")
    print(f"  Min value: {pc.min().item():.4f}")
    print(f"  Max value: {pc.max().item():.4f}")
    print(f"  Mean value: {pc.mean().item():.4f}")
print("="*80 + "\n")

# Hook function to print activation shapes
def print_activation_shape(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            print(f"[{name}] Output shape: {output.shape}")
        elif isinstance(output, tuple):
            shapes = [f"{o.shape}" if isinstance(o, torch.Tensor) else f"type({type(o).__name__})" for o in output]
            print(f"[{name}] Output shapes (tuple): {shapes}")
        elif isinstance(output, dict):
            print(f"[{name}] Output is dict with keys: {list(output.keys())}")
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    print(f"  [{name}][{k}]: {v.shape}")
        else:
            print(f"[{name}] Output type: {type(output).__name__}")
    return hook

# Register hooks on all modules
print("\n" + "="*80)
print("REGISTERING FORWARD HOOKS ON ALL MODULES")
print("="*80 + "\n")

hook_handles = []
for name, module in model.named_modules():
    if name:  # Skip the root module
        handle = module.register_forward_hook(print_activation_shape(name))
        hook_handles.append(handle)

print("\n" + "="*80)
print("STARTING FORWARD PASS")
print("="*80 + "\n")

with torch.inference_mode():
    end_points = model(one_test_sample_from_batch, DEBUG=True)

# Remove hooks after inference
for handle in hook_handles:
    handle.remove()

print("\n" + "="*80)
print("FORWARD PASS COMPLETED")
print("="*80 + "\n")

print("[Difference of keys from inputs to end_points]")
from pprint import pprint 
pprint(set(end_points.keys()) - set(one_test_sample_from_batch.keys()))

for key in one_test_sample_from_batch:
        if key not in end_points:
            end_points[key] = one_test_sample_from_batch[key]

# contrast
pred_center = end_points['last_center'].detach().cpu()
pred_size = end_points["last_pred_size"].detach().cpu()
pred_boxes = torch.concat([pred_center, pred_size], dim=-1).numpy()

proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
sem_scores = torch.softmax(sem_scores_, dim=-1)

token = end_points['tokenized']
mask = token['attention_mask'].detach().cpu()
last_pos = mask.sum(1) - 2

bs = sem_scores.shape[0]
pred_box = np.zeros((bs, 7))
for i in range(bs):
    sim = 1 - sem_scores[i, :, last_pos[i]]
    max_idx = torch.argmax(sim)
    box = pred_boxes[i, max_idx.item()]
    pred_box[i, :6] = box

total_predict_boxes = []
total_predict_boxes.append(pred_box)

predict_boxes = np.vstack(total_predict_boxes)
eval_info = acc25, acc50, m_iou = test_dataset.evaluate(predict_boxes) 
print(f"{acc25=}\n{acc50=}\n{m_iou=}")

