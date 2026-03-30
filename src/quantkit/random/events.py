import numpy as np


class RandomEvents:
    def __init__(self):
        pass

    @staticmethod
    def dice_roll(n_sides: int, n_throws: int, weights: list = None) -> dict:
        assert(n_sides > 3), "Number of sides must be greater than 3."
        assert(n_throws > 0), "Number of throws must be a positive integer."
        assert(weights is None or len(weights) == n_sides), "Weights list must be the same length as the number of sides."
        if weights is not None:
            assert(np.isclose(sum(weights), 1)), "Weights must sum to 1."
        if weights is not None:
            result = np.random.choice(np.arange(1, n_sides + 1), size=n_throws, p=weights)
            mean = np.mean(result)
            std = np.std(result)
            q25 = np.percentile(result, 25)
            q75 = np.percentile(result, 75)
            median = np.median(result)
            return {"mean": float(mean), "std": float(std), "q25": float(q25), "q75": float(q75), "median": float(median), "rolls": result}
        else:
            result = np.random.randint(1, n_sides + 1, size=n_throws)
            mean = np.mean(result)
            std = np.std(result)
            q25 = np.percentile(result, 25)
            q75 = np.percentile(result, 75)
            median = np.median(result)
            return {"mean": float(mean), "std": float(std), "q25": float(q25), "q75": float(q75), "median": float(median), "rolls": result}

    @staticmethod
    def coin_flip(n_flips: int, weights: list = None) -> dict:
        assert(n_flips > 0), "Number of flips must be a positive integer."
        assert(weights is None or len(weights) == 2), "Weights list must be of length 2."
        if weights is not None:
            assert(np.isclose(sum(weights), 1)), "Weights must sum to 1."
            result = np.random.choice([0, 1], size=n_flips, p=weights)
            mean = np.mean(result)
            std = np.std(result)
            q25 = np.percentile(result, 25)
            q75 = np.percentile(result, 75)
            median = np.median(result)
            return {"mean": float(mean), "std": float(std), "q25": float(q25), "q75": float(q75), "median": float(median), "flips": result}
        else:
            result = np.random.randint(0, 2, size=n_flips)
            mean = np.mean(result)
            std = np.std(result)
            q25 = np.percentile(result, 25)
            q75 = np.percentile(result, 75)
            median = np.median(result)
            return {"mean": float(mean), "std": float(std), "q25": float(q25), "q75": float(q75), "median": float(median), "flips": result}
