from imblearn import under_sampling


class Imbalance:
    def __init__(self):
        self.under_samplers = {
            'all_knn': self._all_knn,
            'condensed_nearest_neighbour': self._condensed_nearest_neighbour,
            'edited_nearest_neighbours': self._edited_nearest_neighbours,
            'repeated_edited_nearest_neighbours': self._repeated_edited_nearest_neighbours,
            'random_under_sampler': self._random_under_sampler
        }

    def __call__(self, under_sampler, X, y):
        if under_sampler not in self.under_samplers:
            raise Exception('Under Sampler not implemented')
        else:
            return self.under_samplers[under_sampler](X, y)

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


