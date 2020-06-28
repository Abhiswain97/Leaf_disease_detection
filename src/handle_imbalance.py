from imblearn import under_sampling, over_sampling, combine


class Imbalance:
    def __init__(self):
        self.samplers = {
            "smotetomek": combine.SMOTETomek(),
            "smote": over_sampling.SMOTE(),
            "all_knn": under_sampling.AllKNN(),
            "condensed_nearest_neighbour": under_sampling.CondensedNearestNeighbour(),
            "edited_nearest_neighbours": under_sampling.EditedNearestNeighbours(),
            "repeated_edited_nearest_neighbours": under_sampling.RepeatedEditedNearestNeighbours(),
            "random_under_sampler": under_sampling.RandomUnderSampler(),
        }

    def __call__(self, sampler):
        if sampler not in self.samplers:
            raise Exception("Sampler not implemented")
        else:
            return self.samplers[sampler]

    @staticmethod
    def _all_knn(X, y):
        return under_sampling.AllKNN().fit_resample(X=X, y=y)

    @staticmethod
    def _condensed_nearest_neighbour(X, y):
        return under_sampling.CondensedNearestNeighbour().fit_resample(X=X, y=y)

    @staticmethod
    def _edited_nearest_neighbours(X, y):
        return under_sampling.EditedNearestNeighbours().fit_resample(X=X, y=y)

    @staticmethod
    def _repeated_edited_nearest_neighbours(X, y):
        return under_sampling.RepeatedEditedNearestNeighbours().fit_resample(X=X, y=y)

    @staticmethod
    def _random_under_sampler(X, y):
        return under_sampling.RandomUnderSampler().fit_resample(X=X, y=y)

    @staticmethod
    def _smotetomek(X, y):
        return combine.SMOTETomek().fit_resample(X=X, y=y)

    @staticmethod
    def _smote(X, y):
        return over_sampling.SMOTE().fit_resample(X=X, y=y)
