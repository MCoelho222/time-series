from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from rhis.hypothesis import mann_whitney

P_VALUE_TOL = 1e-4


def test_mann_whitney():
    """
    Test the Mann-Whitney hypothesis test.

    Example from the book Statistical Methods in Water Resources

    Author: Helsel & Hirsch
    Year: 2002
    Source: https://pubs.usgs.gov/twri/twri4a3/twri4a3.pdf

    Chapter 5 - Differences between two independent groups
    """
    x = [0.59, 0.87, 1.1, 1.1, 1.2, 1.3, 1.6, 1.7, 3.2, 4.0] # industrial site
    y = [0.3, 0.36, 0.5, 0.7, 0.7, 0.9, 0.92, 1., 1.3, 9.7] # residential site

    median_x = 1.25
    median_y = 0.8
    expected_p = 0.0491

    result_a = mann_whitney(x=x, y=y)
    result_b = mannwhitneyu(x=x, y=y, method='asymptotic')

    assert np.median(np.array(x)) == median_x
    assert np.median(np.array(y)) == median_y
    assert result_a.p_value == pytest.approx(expected_p, abs=P_VALUE_TOL)
    assert result_a.p_value == pytest.approx(result_b.pvalue, abs=P_VALUE_TOL)



