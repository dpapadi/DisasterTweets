from src.utils.preprocessing import txt_cleanning


def test_cleanning():
    t = "31 This1 is a Dirt@y text found \\x22fsa here: https:\\\\fakesite.com"
    assert txt_cleanning(t, tokenize=False) == "this is a dirty text found here http"
    assert txt_cleanning(t, tokenize=True) == ["this", "is", "a", "dirty", "text", "found", "here", "http"]
