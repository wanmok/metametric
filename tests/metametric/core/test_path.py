import pytest

from metametric.core.path import Path


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


def test_path_getitem():
    path_to_test = Path.parse("@.a.b.c")
    assert path_to_test[0] == "a"
    assert path_to_test[1] == "b"
    assert path_to_test[2] == "c"
    assert path_to_test[0:2] == ("a", "b")
    assert path_to_test[1:] == ("b", "c")
    assert path_to_test[-1] == "c"
    with pytest.raises(IndexError):
        path_to_test[5]


def test_path_append():
    path_to_test = Path.parse("@.a.b")
    assert path_to_test.append("c") == Path.parse("@.a.b.c")
    assert path_to_test.append(1) == Path.parse("@.a.b[1]")


def test_path_prepend():
    path_to_test = Path.parse("a.b")
    assert path_to_test.prepend("c") == Path.parse("c.a.b")
    assert path_to_test.prepend(1) == Path.parse("[1].a.b")


def test_path_is_root():
    assert Path.parse("@").is_root()
    assert not Path.parse("@.a").is_root()
    assert not Path.parse("a").is_root()


def test_path_selects():
    assert Path.parse("@.a.b").selects(Path.parse("@.a.b"))
    assert not Path.parse("@.a.b").selects(Path.parse("@.a"))
    assert not Path.parse("@.a.b").selects(Path.parse("@.a.b.c"))
    assert not Path.parse("@.a.b").selects(Path.parse("@.a.c"))
    assert Path.parse("@.a.*").selects(Path.parse("@.a.b"))
    assert Path.parse("@.a[*]").selects(Path.parse("@.a[1]"))
