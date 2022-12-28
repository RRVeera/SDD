import torch

ckpt_path='/content/model.ckpt'
sd = torch.load(ckpt_path, map_location="cuda")

if "state_dict" not in sd:
    pruned_sd = {
        "state_dict": dict(),
    }
else:
    pruned_sd = dict()
    
for k in sd.keys():
    if k != "optimizer_states":
        if "state_dict" not in sd:
            pruned_sd["state_dict"][k] = sd[k]
        else:
            pruned_sd[k] = sd[k]
              
torch.save(pruned_sd, "model2.ckpt")
