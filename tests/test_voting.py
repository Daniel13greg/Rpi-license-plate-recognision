from carwash_lpr.config import PlatesConfig, VotingConfig
from carwash_lpr.voting import PlateVoter, observe, primary

from conftest import make_read


def obs(text, t, confidence=0.95, box=(500, 500, 760, 560)):
    return observe(make_read(text, confidence, box), PlatesConfig(accept=["moldova", "foreign"]), t)


def test_observe_normalises_and_fixes():
    o = obs("kca 12B", 0.0)
    assert o.plate == "KCA128"
    assert o.display == "KCA 128"
    assert o.confidence < 0.95  # a fixed character lowers the confidence


def test_observe_rejects_categories_and_low_confidence():
    cfg = PlatesConfig(accept=["moldova"], min_read_confidence=0.6)
    assert observe(make_read("MAI1234"), cfg, 0) is None  # moldova_special not accepted
    assert observe(make_read("STOP"), cfg, 0) is None  # unknown
    assert observe(make_read("KCA123", confidence=0.5), cfg, 0) is None
    assert observe(make_read("KCA123"), cfg, 0) is not None


def test_primary_prefers_biggest_plate():
    near = obs("KCA123", 0, box=(0, 0, 300, 60))
    far = obs("ABE456", 0, confidence=0.99, box=(0, 0, 120, 25))
    assert primary([far, near]) is near
    assert primary([]) is None


def test_voter_needs_min_reads():
    voter = PlateVoter(VotingConfig(min_reads=3, min_confidence=0.8))
    voter.add(obs("KCA123", 0.0))
    voter.add(obs("KCA123", 0.3))
    assert voter.decide() is None
    voter.add(obs("KCA123", 0.6))
    decision = voter.decide()
    assert decision.plate == "KCA123"
    assert decision.votes == 3


def test_voter_outvotes_a_misread():
    voter = PlateVoter(VotingConfig(min_reads=2, min_agreement=0.6))
    for t, text in enumerate(["KCA123", "KCA128", "KCA123", "KCA123"]):
        voter.add(obs(text, t * 0.25))
    tally = voter.tally()
    assert [c.plate for c in tally] == ["KCA123", "KCA128"]
    assert voter.decide().plate == "KCA123"


def test_voter_refuses_a_split_vote():
    voter = PlateVoter(VotingConfig(min_reads=2, min_agreement=0.6))
    for t, text in enumerate(["KCA123", "KCA128", "KCA123", "KCA128"]):
        voter.add(obs(text, t * 0.25))
    assert voter.decide() is None


def test_voter_requires_confidence():
    voter = PlateVoter(VotingConfig(min_reads=2, min_confidence=0.9))
    voter.add(obs("KCA123", 0, confidence=0.85))
    voter.add(obs("KCA123", 0.1, confidence=0.85))
    assert voter.decide() is None


def test_voter_window():
    voter = PlateVoter(VotingConfig(window_seconds=3.0, min_reads=2))
    voter.add(obs("KCA123", 0.0))
    voter.add(obs("KCA123", 5.0))
    voter.prune(5.0)
    assert len(voter) == 1
    assert voter.decide() is None
    voter.clear()
    assert len(voter) == 0
