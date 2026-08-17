import argparse

from cosmos_predict2._src.predict2.inference.video2world import load_sam3d_condition_cache


parser = argparse.ArgumentParser()
parser.add_argument(
    "path",
    nargs="?",
    default="/datahdd/mccxadmin/cosmos-sam3d-cache/real-smoke/legacy4k/legacy4k_00000001/condition.pt",
)
path = parser.parse_args().path
value = load_sam3d_condition_cache(path)
print("SAM3D_INFERENCE_CACHE_OK", {key: tuple(tensor.shape) for key, tensor in value.items()})
