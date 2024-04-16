class _Dummy_Cond_Dist():
    def __init__(self, nuisance_dist):
        self.nuisance_dist = nuisance_dist

    def __call__(self, x):
        if x is None:
            return self.nuisance_dist
        else:            
            return self.nuisance_dist.expand(x.shape[:-1])
        
class _DummyScheduler:
    def step(self, *args, **kwargs):
        pass