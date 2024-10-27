from metametric.core.matching import Path


def test_path():
    assert Path.parse("@").components == ()
    assert Path.parse("@.a").components == ("a",)
    assert Path.parse("@.a.b").components == ("a", "b")
    assert Path.parse("a").components == ("a",)
    assert Path.parse("a.b").components == ("a", "b")
    assert Path.parse("a.b.c").components == ("a", "b", "c")
    assert Path.parse("a.b[1].c").components == ("a", "b", 1, "c")
    assert Path.parse("[1]").components == (1,)
    assert Path.parse("[1][2]").components == (1, 2)
    assert Path.parse("[*]").components == (-1,)
    assert Path.parse(".*").components == ("*",)
    assert Path.parse("@.*").components == ("*",)
    assert Path.parse("[*].*").components == (-1, "*")
    assert Path.parse(".*[*]").components == ("*", -1)
