"""
A Python port of the Moses tokenizer hardcoded for German (de).
"""

import re

# Type 1: never break after these when followed by a period
_NONBREAKING = {
    # Single letters (A-Z, a-z)
    *"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    # Roman numerals
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
    "xiii",
    "xiv",
    "xv",
    "xvi",
    "xvii",
    "xviii",
    "xix",
    "xx",
    # Titles and honorifics
    "Adj",
    "Adm",
    "Adv",
    "Asst",
    "Bart",
    "Bldg",
    "Brig",
    "Bros",
    "Capt",
    "Cmdr",
    "Col",
    "Comdr",
    "Con",
    "Corp",
    "Cpl",
    "DR",
    "Dr",
    "Ens",
    "Gen",
    "Gov",
    "Hon",
    "Hosp",
    "Insp",
    "Lt",
    "MM",
    "MR",
    "MRS",
    "MS",
    "Maj",
    "Messrs",
    "Mlle",
    "Mme",
    "Mr",
    "Mrs",
    "Ms",
    "Msgr",
    "Op",
    "Ord",
    "Pfc",
    "Ph",
    "Prof",
    "Pvt",
    "Rep",
    "Reps",
    "Res",
    "Rev",
    "Rt",
    "Sen",
    "Sens",
    "Sfc",
    "Sgt",
    "Sr",
    "St",
    "Supt",
    "Surg",
    # Misc abbreviations
    "Mio",
    "Mrd",
    "bzw",
    "vs",
    "usw",
    "d.h",
    "z.B",
    "u.a",
    "etc",
    "MwSt",
    "ggf",
    "d.J",
    "D.h",
    "m.E",
    "vgl",
    "I.F",
    "z.T",
    "sogen",
    "ff",
    "u.E",
    "g.U",
    "g.g.A",
    "c.-à-d",
    "Buchst",
    "u.s.w",
    "sog",
    "u.ä",
    "Std",
    "evtl",
    "Zt",
    "Chr",
    "u.U",
    "o.ä",
    "Ltd",
    "b.A",
    "z.Zt",
    "spp",
    "sen",
    "SA",
    "k.o",
    "jun",
    "i.H.v",
    "dgl",
    "dergl",
    "Co",
    "zzt",
    "usf",
    "s.p.a",
    "Dkr",
    "bzgl",
    "BSE",
}

# Type 2: only non-breaking when followed by a digit (0-9)
_NONBREAKING_NUMERIC_ONLY = {"No", "Nos", "Art", "Nr", "pp", "ca", "Ca"}

# German ordinals: digits 1-99 — "1." means "1st", not end-of-sentence
_NONBREAKING_NUMERIC_ONLY.update(str(n) for n in range(1, 100))

_PREFIXES: dict[str, int] = {word: 1 for word in _NONBREAKING}
_PREFIXES.update({word: 2 for word in _NONBREAKING_NUMERIC_ONLY})


class Tokenizer:
    """
    Moses-style tokenizer, hardcoded for German.
    """

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize *text* and return a list of token strings.
        """
        return self._tokenize(text).split()

    def tokenize_to_string(self, text: str) -> str:
        """
        Return tokens as a newline-separated string (mirrors original stdout).
        """
        return "\n".join(self.tokenize(text))

    def tokenize_sentences(self, sentences: list[str]) -> list[str]:
        """
        Tokenize a list of sentences, returning each as a newline-joined
        token string. Drop-in replacement for ``process_by_sentence()``.
        """
        return ["\n".join(self.tokenize(s)) for s in sentences]

    def _tokenize(self, text: str) -> str:
        text = text.rstrip("\n")
        text = f" {text} "

        # Separate out all "other" special characters
        text = re.sub(r"([^\w\s.\'\`,\-])", r" \1 ", text)

        # Multi-dots stay together
        text = re.sub(r"\.([\.]+)", r" DOTMULTI\1", text)
        while "DOTMULTI." in text:
            text = re.sub(r"DOTMULTI\.([^\.])", r"DOTDOTMULTI \1", text)
            text = re.sub(r"DOTMULTI\.", "DOTDOTMULTI", text)

        # Separate out "," except if within numbers (5.300 / 5,300)
        text = re.sub(r"([^\d])[,]([^\d])", r"\1 , \2", text)
        text = re.sub(r"([\d])[,]([^\d])", r"\1 , \2", text)
        text = re.sub(r"([^\d])[,]([\d])", r"\1 , \2", text)

        # Turn ` into '
        text = text.replace("`", "'")

        # Turn '' into "
        text = re.sub(r"''", ' " ', text)

        # German: split all apostrophes
        text = re.sub(r"'", " ' ", text)

        # Word-token method: handle trailing periods
        words = text.split()
        output_words: list[str] = []
        for i, word in enumerate(words):
            m = re.match(r"^(\S+)\.$", word)
            if m:
                pre = m.group(1)
                next_word = words[i + 1] if i < len(words) - 1 else ""
                if (
                    (re.search(r"\.", pre) and re.search(r"[A-Za-z]", pre))
                    or (_PREFIXES.get(pre) == 1)
                    or (next_word and re.match(r"^[a-z]", next_word))
                ):
                    pass  # no change
                elif (
                    _PREFIXES.get(pre) == 2
                    and next_word
                    and re.match(r"^[0-9]+", next_word)
                ):
                    pass  # no change
                else:
                    word = pre + " ."
            output_words.append(word)

        text = " ".join(output_words)

        # Clean up extraneous spaces
        text = re.sub(r" +", " ", text).strip()

        # Restore multi-dots
        while "DOTDOTMULTI" in text:
            text = text.replace("DOTDOTMULTI", "DOTMULTI.")
        text = text.replace("DOTMULTI", ".")

        return text
