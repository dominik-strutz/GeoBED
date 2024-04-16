from torch.optim import Adam
from ..utils.misc import _DummyScheduler
from ._mi_lower_bounds import _mi_lower_bound

def flo(
    self,
    design,
    guide,
    N,
    M=-1,
    guide_kwargs={},
    K=10,
    n_batch=1,
    n_epochs=100,
    optimizer=Adam,
    optimizer_kwargs={},
    scheduler=_DummyScheduler,
    scheduler_kwargs={},
    return_guide=True,
    return_train_loss=True,
    return_test_loss=True,
    progress_bar=False,
    ):
    
    bound_kwargs={'K': K}
    
    return _mi_lower_bound(
        self,
        design,
        guide,
        N,
        'flo',
        bound_kwargs,
        M,
        guide_kwargs,
        n_batch,
        n_epochs,
        optimizer,
        optimizer_kwargs,
        scheduler,
        scheduler_kwargs,
        return_guide,
        return_train_loss,
        return_test_loss,
        progress_bar,
        )