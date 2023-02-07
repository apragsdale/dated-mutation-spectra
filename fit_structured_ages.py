import numpy as np
import scipy.linalg
import scipy.optimize


def predict_spectrum(ages):
    """
    This comes from the `age_modeling.R` script from Wang et al.
    """
    alpha = np.array([13.830321, 15.180457, 14.056053, 13.923672, 13.952551, 14.947698])
    beta0 = np.array([-0.316633, -0.327940, -0.322887, -0.329628, -0.321475, -0.326378])
    beta1 = np.array([0.252819, 0.265539, 0.249886, 0.264401, 0.262430, 0.256306])
    p = np.exp(alpha + ages[0] * beta0 + ages[1] * beta1)
    return p / np.sum(p)


def clr(x):
    geom_mean = np.prod(x) ** (1 / len(x))
    return np.log(x / geom_mean)


def cost_func(ages, data, predict_spectrum, proportion=0.1):
    predicted = (1 - proportion) * predict_spectrum(
        (20, 20)
    ) + proportion * predict_spectrum(ages)
    predicted /= predicted.sum()
    dist = scipy.linalg.norm(clr(predicted) - clr(data))
    return dist

# observed inferred ages (at 10k gens ago)
obs_ages = (28, 23)

# proportion of spectrum from ghost lineage
proportion = 0.3

data = predict_spectrum(obs_ages)
args = (data, predict_spectrum, proportion)

ret = scipy.optimize.fmin_l_bfgs_b(
    cost_func, (30, 30), args=args, approx_grad=True, bounds=[[10, 100], [10, 100]]
)

print(ret[0])
