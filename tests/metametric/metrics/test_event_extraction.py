"""Tests for event extraction metrics."""

from pytest import fixture, approx


from metametric.structures.ie import Mention, Trigger, Argument, Event, EventSet
from metametric.metrics.event_extraction import event_extraction_suite


@fixture
def data():
    """From https://aclanthology.org/P13-1008.pdf."""
    baghdad = Mention(left=4, right=11)
    cameraman = Mention(left=15, right=24)
    died = Mention(left=25, right=29)
    american_tank = Mention(left=38, right=51)
    fired = Mention(left=52, right=57)
    palestine_hotel = Mention(left=65, right=80)

    ref_die = Event(
        trigger=Trigger(mention=died, type="die"),
        args=[
            Argument(mention=baghdad, role="place"),
            Argument(mention=cameraman, role="victim"),
            Argument(mention=american_tank, role="instrument"),
        ],
    )
    ref_attack = Event(
        trigger=Trigger(mention=fired, type="attack"),
        args=[
            Argument(mention=american_tank, role="instrument"),
            Argument(mention=baghdad, role="place"),
            Argument(mention=cameraman, role="target"),
            Argument(mention=palestine_hotel, role="target"),
        ],
    )

    pred_die = Event(
        trigger=Trigger(mention=died, type="die"),
        args=[
            Argument(mention=palestine_hotel, role="place"),
            Argument(mention=cameraman, role="victim"),
        ],
    )
    pred_attack = Event(
        trigger=Trigger(mention=fired, type="attack"),
        args=[
            Argument(mention=american_tank, role="attacker"),
            Argument(mention=baghdad, role="place"),
            Argument(mention=palestine_hotel, role="target"),
        ],
    )

    pred = EventSet(events=[pred_die, pred_attack])
    ref = EventSet(events=[ref_die, ref_attack])
    return pred, ref


def test_event_extraction(data):
    """Event extraction metrics."""
    pred, ref = data
    agg = event_extraction_suite.new()
    agg.update_single(pred, ref)
    metrics = agg.compute()
    for k, v in metrics.items():
        print(f"{k}: {v}")

    assert metrics["trigger_identification-precision"] == approx(1.0, abs=0.01)
    assert metrics["trigger_identification-recall"] == approx(1.0, abs=0.01)
    assert metrics["trigger_identification-f1"] == approx(1.0, abs=0.01)
    assert metrics["trigger_classification-precision"] == approx(1.0, abs=0.01)
    assert metrics["trigger_classification-recall"] == approx(1.0, abs=0.01)
    assert metrics["trigger_classification-f1"] == approx(1.0, abs=0.01)
    assert metrics["argument_identification-precision"] == approx(0.8, abs=0.01)
    assert metrics["argument_identification-recall"] == approx(0.57, abs=0.01)
    assert metrics["argument_identification-f1"] == approx(0.67, abs=0.01)
    assert metrics["argument_classification-precision"] == approx(0.6, abs=0.01)
    assert metrics["argument_classification-recall"] == approx(0.43, abs=0.01)
    assert metrics["argument_classification-f1"] == approx(0.5, abs=0.01)
