import numpy as np
import pytest

from eos.nodes.parse_interm_layer_elements import (
    return_index_of_uniques_given_instances,
)


def test_basic_functionality() -> None:
    uniques = np.array([10, 20, 30, 40])
    instances = np.array([20, 30, 20, 10, 40, 30])
    expected = np.array([1, 2, 1, 0, 3, 2])
    assert np.array_equal(
        return_index_of_uniques_given_instances(uniques, instances), expected
    )


def test_with_negative_numbers() -> None:
    uniques = np.array([-10, -20, -30, -40])
    instances = np.array([-30, -10, -40, -20])
    expected = np.array([2, 0, 3, 1])
    assert np.array_equal(
        return_index_of_uniques_given_instances(uniques, instances), expected
    )


def test_with_repeated_uniques_should_fail() -> None:
    uniques = np.array([10, 20, 20, 30])
    instances = np.array([20, 30, 10])
    with pytest.raises(ValueError):
        return_index_of_uniques_given_instances(uniques, instances)


def test_empty_arrays() -> None:
    uniques = np.array([])
    instances = np.array([])
    expected = np.array([])
    assert np.array_equal(
        return_index_of_uniques_given_instances(uniques, instances), expected
    )


def test_instances_not_in_uniques() -> None:
    uniques = np.array([1, 2, 3, 4])
    instances = np.array([5, 6, 7])
    with pytest.raises(ValueError):
        return_index_of_uniques_given_instances(uniques, instances)


def test_all_instances_are_the_same() -> None:
    uniques = np.array([5, 10, 15, 20])
    instances = np.array([10, 10, 10, 10])
    expected = np.array([1, 1, 1, 1])
    assert np.array_equal(
        return_index_of_uniques_given_instances(uniques, instances), expected
    )
