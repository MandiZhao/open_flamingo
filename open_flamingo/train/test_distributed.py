import torch
from distributed import *

# local_rank, _, _ = world_info_from_env()
local_rank = 0
torch.distributed.init_process_group(
                backend='nccl', 
                init_method="env://172.24.72.83",
)
world_size = torch.distributed.get_world_size()
rank = torch.distributed.get_rank()
print('World size, rank, local_rank:', world_size, rank, local_rank)

exit()
