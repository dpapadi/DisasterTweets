from src.utils.model_utils import get_train_test_sets, _sentence_to_index
from src.vocabulary.base import Vocabulary


def test_get_sets():
    X = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    y = [0, 1, 2, 3, 4]

    X_train, X_test, y_train, y_test = get_train_test_sets(X, y, None, None, test_size=0.2, random_state=1)
    assert X_train == [[1, 1], [4, 4], [0, 0], [3, 3]]
    assert y_train == [1, 4, 0, 3]
    assert X_test == [[2, 2]]
    assert y_test == [2]


def test_sentence_to_index():
    voc = ["this", "is", "test", "a"]
    vocabulary = Vocabulary().add_list(voc)
    sentence_long = "this is a marvelous test and some noise"
    sentence_short = "this is a"
    assert [1, 2, 4, -1, 3] == _sentence_to_index(sentence_long, sentence_length=5, vocabulary=vocabulary)
    assert[1, 2, 4, 0, 0] == _sentence_to_index(sentence_short, sentence_length=5, vocabulary=vocabulary)
