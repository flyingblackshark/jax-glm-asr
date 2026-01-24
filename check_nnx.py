
from flax import nnx
import jax.numpy as jnp

class Model(nnx.Module):
    def __init__(self, rngs):
        self.linear = nnx.Linear(2, 2, rngs=rngs)

rngs = nnx.Rngs(0)
model = Model(rngs)
flat_state = nnx.state(model).flat_state()
flat_state_dict = dict(flat_state)
print("Keys:", flat_state_dict.keys())

try:
    # Try updating with raw dict
    nnx.update(model, flat_state_dict)
    print("Update with raw dict -> Success")
except Exception as e:
    print(f"Update with raw dict -> Failed: {e}")

try:
    # Try creating State from raw dict (if constructor allows flat)
    st = nnx.State(dict(flat_state))
    nnx.update(model, st)
    print("Update with State(dict) -> Success")
except Exception as e:
    print(f"Update with State(dict) -> Failed: {e}")

# Check from_flat_path
if hasattr(nnx.State, 'from_flat_path'):
    print("State.from_flat_path exists")
else:
    print("State.from_flat_path MISSING")
