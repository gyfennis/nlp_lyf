from engine.vote_resolver import tally_votes, get_majority_target


def test_simple_majority():
    votes = {1: 3, 2: 3, 3: 5, 4: 5, 5: 5}
    exiled, tied = tally_votes(votes, [1, 2, 3, 4, 5])
    assert exiled == 5
    assert tied == []


def test_tie():
    votes = {1: 3, 2: 5, 3: 3, 4: 5}
    exiled, tied = tally_votes(votes, [1, 2, 3, 4])
    assert exiled is None
    assert 3 in tied
    assert 5 in tied


def test_all_abstain():
    votes = {1: None, 2: None}
    exiled, tied = tally_votes(votes, [1, 2])
    assert exiled is None
    assert tied == []


def test_empty_votes():
    exiled, tied = tally_votes({}, [1, 2])
    assert exiled is None
    assert tied == []


def test_single_voter():
    votes = {1: 4}
    exiled, tied = tally_votes(votes, [1, 2, 3, 4])
    assert exiled == 4
    assert tied == []


def test_get_majority_target():
    assert get_majority_target({1: 3, 2: 3}) == 3
    assert get_majority_target({1: 3, 2: 4}) == 4
    assert get_majority_target({1: 3, 2: 5, 3: 3}) is None
