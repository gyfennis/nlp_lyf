def tally_votes(votes: dict[int, int], alive_player_ids: list[int]) -> tuple[int | None, list[int]]:
    """
    Tally votes and return (exiled_id or None, tied_player_ids).
    Returns (None, []) if all abstain.
    Returns (None, [a, b]) for a tie requiring PK.
    """
    if not votes:
        return None, []

    vote_count: dict[int, int] = {}
    for target in votes.values():
        if target is not None:
            vote_count[target] = vote_count.get(target, 0) + 1

    if not vote_count:
        return None, []

    max_votes = max(vote_count.values())
    top_candidates = [pid for pid, count in vote_count.items() if count == max_votes]

    if len(top_candidates) > 1:
        return None, top_candidates

    return top_candidates[0], []


def get_majority_target(votes: dict[int, int]) -> int | None:
    """Get the player with the most votes. Returns None if empty or tied."""
    exiled, tied = tally_votes(votes, list(votes.keys()))
    if tied:
        return None
    return exiled
