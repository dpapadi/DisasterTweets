import re
import string


def txt_cleanning(text, tokenize=False):
    """ cleanning:
            lowercase
            remove numbers
            remove non alphabetic characters
            expand expressions
    """
    text = "".join([char.lower() for char in text])
    text = re.sub('[0-9]+', '', text)
    text = text.split()
    out_text = []
    for word in text:
        if "\\x" in word:
            continue
        if "http" in word:
            out_text.append("http")
            continue
        if word == "ain't":
            out_text += ["is", "not"]
            continue
        if word == "won't":
            out_text += ["will", "not"]
            continue
        if word and word[0] in string.punctuation:
            word = word[1:]
        if word and word[-1] in string.punctuation:
            word = word[:-1]
        if len(word) > 4 and "n't" in word[-3:]:
            out_text.append(word[:-3])
            out_text.append("not")
        elif "'" in word:
            new_words = word.split("'")
            out_text += [new_words[0], "'" + new_words[1]]
        elif "-" in word:
            out_text += word.split("-")
        elif word:
            word = re.sub("[^a-z]+", "", word)
            out_text += ("".join([char if char.isalpha()
                         else " " for char in word]).split())
    out_text = [word.strip() for word in out_text if word.strip()]
    if not tokenize:
        out_text = " ".join([str(elem) for elem in out_text])
    return out_text
