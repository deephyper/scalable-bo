from deephyper.stopper._stopper import Stopper


class ModelBaseStopper(Stopper):
    def observe(self, budget: float, objective: float):
        pass

    def stop(self) -> bool:
        pass
