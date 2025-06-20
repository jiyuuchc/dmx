import optax
from flax import nnx

def ema_step(
    new_model: nnx.Module,
    old_model: nnx.Module, 
    decay: float = 0.999,
) -> nnx.Module:
    old_params = nnx.split(old_model, nnx.Param, ...)[1]
    graphdef, new_params, *other_state = nnx.split(new_model, nnx.Param, ...)
    params = optax.incremental_update(new_params, old_params, decay)
    return nnx.merge(graphdef, params, *other_state)

class EMAOptimizer(nnx.Optimizer):
    
    def __init__(
        self,
        model: nnx.Module,
        tx: optax.GradientTransformation,
        wrt: nnx.filterlib.Filter = nnx.Param,
        ema_decay: float = 0.999,
    ):
        super().__init__(model, tx, wrt)
        self.ema_model = model
        self.ema_decay = ema_decay
        
    def update(self, grads, **kwargs):
        super().update(grads, **kwargs)
        self.ema_model = ema_step(self.model, self.ema_model, self.ema_decay)
