import torch
import argparse
import os

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('CHECKPOINT_PATH', type=str)
args = parser.parse_args()


ckpt = torch.load(args.CHECKPOINT_PATH,
                  map_location='cpu',
                  weights_only=True)
weights = ckpt['state_dict']
# change dict keys names by removing 'sam2_model' prefix

new_weights = {}
for k, v in weights.items():
    new_k = k.replace('sam2_model.', '')
    new_weights[new_k] = v

# save new weights with 'weights.pt' as output file path
output_path = os.path.join(os.path.dirname(args.CHECKPOINT_PATH), 'weights.pt')
print(f'Saving new weights to {output_path}')
torch.save({'model': new_weights},
           output_path)
