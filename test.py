from typing import List, Dict, Any, Tuple

from tqdm import tqdm
import random

import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
import torch
from pytorch_lightning.utilities.seed import seed_everything, reset_seed

from fleiss_kappa import FleissKappa

def simulate_data(diagonal_p: float, num_raters: int, num_classes: int, num_subjects: int) -> Tuple[np.ndarray, torch.Tensor]:
    off_diag = (1-diagonal_p)/(num_raters-1)
    diag_increment = diagonal_p-off_diag
    cov = np.tile(off_diag, (num_raters,num_raters))
    cov += np.eye(num_raters)* diag_increment
    
    np_logits = np.random.multivariate_normal(mean=np.zeros(num_raters), cov=cov, size=(num_subjects,num_classes))
    torch_logits = torch.from_numpy(np_logits)
    
    return np_logits, torch_logits

def np_convert_logits_to_counts(logits: np.ndarray, num_subjects: int, num_raters: int, num_classes: int) -> np.ndarray:
    dense_pred = logits.argmax(axis=1)
    rating_counts = np.zeros((num_subjects,num_classes))
    for participant in range(num_subjects):
        for rater in range(num_raters):
            single_rating = dense_pred[participant, rater]
            rating_counts[participant, single_rating] +=1
    return rating_counts

def assert_equality(covariance_agreement: np.ndarray, num_raters: int, num_classes: int, num_subjects: int, tolerance: float) -> Tuple[float, float]:
    np_logits, torch_logits = simulate_data(covariance_agreement, num_raters, num_classes, num_subjects) 
    torch_kappa_fn = FleissKappa(num_classes, num_raters)
    torch_kappa_fn.update(torch_logits, None)
    torch_kappa = float(torch_kappa_fn.compute().detach().cpu().numpy())

    np_counts = np_convert_logits_to_counts(np_logits, num_subjects, num_raters, num_classes)
    np_kappa = float(fleiss_kappa(np_counts))
    
    if abs(torch_kappa) < 0.0001 or abs(np_kappa) < 0.0001: #if values get small np.isclose fails
        assert abs(torch_kappa) < 0.0001 + tolerance and abs(np_kappa) < 0.0001 + tolerance
    else:
        assert np.isclose(torch_kappa, np_kappa, rtol=tolerance), f"Kappas unequal: Torch {torch_kappa:.4f}, Statsmodel: {np_kappa:.4f}"
    return torch_kappa, np_kappa


if __name__ == "__main__":
    """Goal of this file is to test the Fleiss Kappa implementation against the `statsmodels` package.
    """
    # make a single test with seed and print output
    seed_everything(23633)
    num_subjects, num_raters, num_classes = 64, 8, 10
    covariance_agreement = 0.6

    np_logits, torch_logits = simulate_data(covariance_agreement, num_raters, num_classes, num_subjects) 
    torch_kappa_fn = FleissKappa(num_classes, num_raters)
    torch_kappa_fn.update(torch_logits, None)
    torch_kappa = torch_kappa_fn.compute()
    print(f"Torchmetrics Kappa: {torch_kappa}")

    np_counts = np_convert_logits_to_counts(np_logits, num_subjects, num_raters, num_classes)
    np_kappa = fleiss_kappa(np_counts)
    print(f"Statsmodels Kappa: {np_kappa}")
    
    #make multiple tests and assert equality
    reset_seed()
    num_tests = 10000
    max_delta_assert = 0.0005
    
    for i in tqdm(range(num_tests), desc="Automatic Testing"):
        num_subjects = random.choice(range(10,100))
        num_raters = random.choice(range(2,20))
        num_classes  = random.choice(range(2,20))
        covariance_agreement = random.uniform(0,1)
        kappas = []
        kappas.append(assert_equality(covariance_agreement, num_raters, num_classes, num_subjects, max_delta_assert))

    print(f"No deviations found beyond a tolerance of {max_delta_assert}.")
