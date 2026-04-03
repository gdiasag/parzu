"""
Punkt sentence boundary detection for German text.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import TYPE_CHECKING

import punkt_data_german

if TYPE_CHECKING:
    from io import TextIOBase


class OrthoContext(IntFlag):
    """
    Flags that record the orthographic contexts a word type has appeared in.
    """

    BEG_UC = auto()  # sentence-initial, upper case
    MID_UC = auto()  # sentence-internal, upper case
    UNK_UC = auto()  # unknown position, upper case
    BEG_LC = auto()  # sentence-initial, lower case
    MID_LC = auto()  # sentence-internal, lower case
    UNK_LC = auto()  # unknown position, lower case

    # Convenience composites
    @classmethod
    def upper(cls) -> OrthoContext:
        return cls.BEG_UC | cls.MID_UC | cls.UNK_UC

    @classmethod
    def lower(cls) -> OrthoContext:
        return cls.BEG_LC | cls.MID_LC | cls.UNK_LC


_ORTHO_MAP: dict[tuple[str, str], OrthoContext] = {
    ("initial", "upper"): OrthoContext.BEG_UC,
    ("internal", "upper"): OrthoContext.MID_UC,
    ("unknown", "upper"): OrthoContext.UNK_UC,
    ("initial", "lower"): OrthoContext.BEG_LC,
    ("internal", "lower"): OrthoContext.MID_LC,
    ("unknown", "lower"): OrthoContext.UNK_LC,
}


class PunktLanguageVars:
    """
    Language-specific regular expressions for sentence boundary detection.

    Subclass and override attributes / properties to adapt to a new language.
    """

    sent_end_chars: tuple[str, ...] = (".", "?", "!")
    internal_punctuation: str = ",:;"

    re_boundary_realignment: re.Pattern[str] = re.compile(
        r'["\')\]}]+?(?:\s+|(?=--)|$)', re.MULTILINE
    )

    _re_word_start = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"
    _re_non_word_chars = r"(?:[?!)\";}\]\*:@\'\({\[])"
    _re_multi_char_punct = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)"

    _word_tokenize_fmt = r"""(
        %(MultiChar)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found
        (?= # Sequences marking a word's end
            \s|                                  # White-space
            $|                                   # End-of-string
            %(NonWord)s|%(MultiChar)s|           # Punctuation
            ,(?=$|\s|%(NonWord)s|%(MultiChar)s)  # Comma if at end of word
        )
        |
        \S
    )"""

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            \s+(?P<next_tok>\S+)     # or whitespace and some other token
        ))"""

    # ------------------------------------------------------------------
    # Cached compiled patterns (one per instance)
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._re_word_tokenizer: re.Pattern[str] | None = None
        self._re_period_context: re.Pattern[str] | None = None

    @property
    def _re_sent_end_chars(self) -> str:
        return "[%s]" % re.escape("".join(self.sent_end_chars))

    def _word_tokenizer_re(self) -> re.Pattern[str]:
        if self._re_word_tokenizer is None:
            self._re_word_tokenizer = re.compile(
                self._word_tokenize_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "MultiChar": self._re_multi_char_punct,
                    "WordStart": self._re_word_start,
                },
                re.UNICODE | re.VERBOSE,
            )
        return self._re_word_tokenizer

    def word_tokenize(self, s: str) -> list[str]:
        """Split a string into tokens (periods are kept attached)."""
        return self._word_tokenizer_re().findall(s)

    def period_context_re(self) -> re.Pattern[str]:
        """
        Return a compiled pattern for potential sentence-boundary contexts.
        """
        if self._re_period_context is None:
            self._re_period_context = re.compile(
                self._period_context_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "SentEndChars": self._re_sent_end_chars,
                },
                re.UNICODE | re.VERBOSE,
            )
        return self._re_period_context


@dataclass
class PunktParameters:
    """Stores learned / pre-loaded data for sentence boundary detection."""

    abbrev_types: set[str] = field(default_factory=set)
    collocations: set[tuple[str, str]] = field(default_factory=set)
    sent_starters: set[str] = field(default_factory=set)
    ortho_context: defaultdict[str, OrthoContext] = field(
        default_factory=lambda: defaultdict(lambda: OrthoContext(0))
    )

    def add_ortho_context(self, typ: str, flag: OrthoContext) -> None:
        self.ortho_context[typ] |= flag

    def clear_abbrevs(self) -> None:
        self.abbrev_types.clear()

    def clear_collocations(self) -> None:
        self.collocations.clear()

    def clear_sent_starters(self) -> None:
        self.sent_starters.clear()

    def clear_ortho_context(self) -> None:
        self.ortho_context.clear()


_RE_NON_PUNCT: re.Pattern[str] = re.compile(r"[^\W\d]", re.UNICODE)
_RE_NUMERIC: re.Pattern[str] = re.compile(
    r"^-?[\.,]?[\divxlcdm][\d,\.\-ivxlcdm]*\.?$"
)


@dataclass
class PunktToken:
    """
    A single text token, plus annotations accumulated during sentence-boundary
    detection.
    """

    tok: str
    parastart: bool = False
    linestart: bool = False
    sentbreak: bool | None = None
    abbr: bool | None = None
    ellipsis: bool | None = None

    # Compiled patterns shared across all instances
    _RE_ELLIPSIS: re.ClassVar[re.Pattern[str]] = re.compile(r"\.\.+$")
    _RE_INITIAL: re.ClassVar[re.Pattern[str]] = re.compile(
        r"[^\W\d]\.$", re.UNICODE
    )
    _RE_ALPHA: re.ClassVar[re.Pattern[str]] = re.compile(
        r"[^\W\d]+$", re.UNICODE
    )

    def __post_init__(self) -> None:
        self.type: str = _RE_NUMERIC.sub("##number##", self.tok.lower())
        self.period_final: bool = self.tok.endswith(".")

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def type_no_period(self) -> str:
        if len(self.type) > 1 and self.type[-1] == ".":
            return self.type[:-1]
        return self.type

    @property
    def type_no_sentperiod(self) -> str:
        return self.type_no_period if self.sentbreak else self.type

    @property
    def first_upper(self) -> bool:
        return self.tok[0].isupper()

    @property
    def first_lower(self) -> bool:
        return self.tok[0].islower()

    @property
    def first_case(self) -> str:
        if self.first_lower:
            return "lower"
        if self.first_upper:
            return "upper"
        return "none"

    @property
    def is_ellipsis(self) -> re.Match[str] | None:
        return self._RE_ELLIPSIS.match(self.tok)

    @property
    def is_number(self) -> bool:
        return self.type.startswith("##number##")

    @property
    def is_initial(self) -> re.Match[str] | None:
        return self._RE_INITIAL.match(self.tok)

    @property
    def is_alpha(self) -> re.Match[str] | None:
        return self._RE_ALPHA.match(self.tok)

    @property
    def is_non_punct(self) -> re.Match[str] | None:
        return _RE_NON_PUNCT.search(self.type)

    def __repr__(self) -> str:
        type_str = f" type={self.type!r}," if self.type != self.tok else ""
        props = ", ".join(
            f"{p}={getattr(self, p)!r}"
            for p in ("parastart", "linestart", "sentbreak", "abbr", "ellipsis")
            if getattr(self, p)
        )
        return f"{self.__class__.__name__}({self.tok!r},{type_str} {props})"

    def __str__(self) -> str:
        res = self.tok
        if self.abbr:
            res += "<A>"
        if self.ellipsis:
            res += "<E>"
        if self.sentbreak:
            res += "<S>"
        return res


def _pair_iter(
    it: Iterator[PunktToken],
) -> Generator[tuple[PunktToken, PunktToken | None], None, None]:
    """
    Yield consecutive overlapping pairs from *it*.  The last pair has
    ``None`` as its second element.
    """
    it = iter(it)
    prev = next(it)
    for el in it:
        yield prev, el
        prev = el
    yield prev, None


class PunktBaseClass:
    """Shared logic for both training and tokenisation."""

    def __init__(
        self,
        lang_vars: PunktLanguageVars | None = None,
        token_cls: type[PunktToken] = PunktToken,
        params: PunktParameters | None = None,
    ) -> None:
        self._lang_vars = lang_vars or PunktLanguageVars()
        self._Token = token_cls
        self._params = params or PunktParameters()

    def _tokenize_words(self, plaintext: str) -> Iterator[PunktToken]:
        """
        Split *plaintext* into :class:`PunktToken` instances, marking
        paragraph- and line-starts.
        """
        parastart = False
        for line in plaintext.split("\n"):
            if line.strip():
                line_toks = iter(self._lang_vars.word_tokenize(line))
                yield self._Token(
                    next(line_toks), parastart=parastart, linestart=True
                )
                parastart = False
                for t in line_toks:
                    yield self._Token(t)
            else:
                parastart = True

    def _annotate_first_pass(
        self, tokens: Iterator[PunktToken]
    ) -> Iterator[PunktToken]:
        """Yield tokens with type-based sentence-boundary annotations."""
        for tok in tokens:
            self._first_pass_annotation(tok)
            yield tok

    def _first_pass_annotation(self, aug_tok: PunktToken) -> None:
        """Apply type-based annotation to a single token (in-place)."""
        tok = aug_tok.tok

        if tok in self._lang_vars.sent_end_chars:
            aug_tok.sentbreak = True
        elif aug_tok.is_ellipsis:
            aug_tok.ellipsis = True
        elif aug_tok.period_final and not tok.endswith(".."):
            stripped = tok[:-1].lower()
            is_abbrev = (
                stripped in self._params.abbrev_types
                or stripped.split("-")[-1] in self._params.abbrev_types
            )
            aug_tok.abbr = is_abbrev
            if not is_abbrev:
                aug_tok.sentbreak = True


class PunktSentenceTokenizer(PunktBaseClass):
    """
    Unsupervised sentence boundary detector for German text.

    Pre-trained parameters are loaded from :mod:`punkt_data_german`.
    """

    PUNCTUATION: tuple[str, ...] = (";", ":", ",", ".", "!", "?")

    def __init__(
        self,
        lang_vars: PunktLanguageVars | None = None,
        token_cls: type[PunktToken] = PunktToken,
    ) -> None:
        super().__init__(lang_vars=lang_vars, token_cls=token_cls)

        self._params.collocations = punkt_data_german.collocations
        self._params.ortho_context = punkt_data_german.ortho_context
        self._params.abbrev_types = punkt_data_german.abbrev_types
        self._params.sent_starters = punkt_data_german.sent_starters

    def tokenize(
        self, text: str, realign_boundaries: bool = False
    ) -> list[str]:
        """Return a list of sentences found in *text*."""
        return list(self.sentences_from_text(text, realign_boundaries))

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        """
        Return ``(start, end)`` character spans for each sentence in *text*.
        """
        return [(sl.start, sl.stop) for sl in self._slices_from_text(text)]

    def sentences_from_text(
        self, text: str, realign_boundaries: bool = False
    ) -> list[str]:
        """
        Return sentences from *text*, optionally realigning closing punctuation.
        """
        sents = [text[sl] for sl in self._slices_from_text(text)]
        if realign_boundaries:
            sents = list(self._realign_boundaries(sents))
        return sents

    def tokenize_fobj(
        self,
        fobj_in: TextIOBase,
        fobj_out: TextIOBase,
    ) -> None:
        """
        Tokenize a file object, writing one sentence per line to *stdout*.
        """
        for sent in self._slices_from_fobj(fobj_in):
            sys.stdout.write(sent.replace("\n", "").strip() + "\n")

    def _slices_from_text(self, text: str) -> Iterator[slice]:
        last_break = 0
        for match in self._lang_vars.period_context_re().finditer(text):
            context = match.group() + match.group("after_tok")
            if self.text_contains_sentbreak(context):
                yield slice(last_break, match.end())
                last_break = (
                    match.start("next_tok")
                    if match.group("next_tok")
                    else match.end()
                )
        yield slice(last_break, len(text))

    def _slices_from_fobj(self, fobj: TextIOBase) -> Generator[str, None, None]:
        """Yield sentence strings from a file object, line by line."""
        buf = ""
        for line in fobj:
            buf += line.strip() + " "
            last_break = 0
            for match in self._lang_vars.period_context_re().finditer(buf):
                context = match.group() + match.group("after_tok")
                if self.text_contains_sentbreak(context):
                    yield buf[last_break : match.end()]
                    last_break = (
                        match.start("next_tok")
                        if match.group("next_tok")
                        else match.end()
                    )
            buf = buf[last_break:]
        yield buf

    def text_contains_sentbreak(self, text: str) -> bool:
        """Return ``True`` if *text* contains a sentence break."""
        found = False
        for t in self._annotate_tokens(self._tokenize_words(text)):
            if found:
                return True
            if t.sentbreak:
                found = True
        return False

    def sentences_from_tokens(
        self, tokens: Iterator[str]
    ) -> Generator[list[str], None, None]:
        """Yield lists of token strings, one list per sentence."""
        annotated = iter(self._annotate_tokens(self._Token(t) for t in tokens))
        sentence: list[str] = []
        for aug_tok in annotated:
            sentence.append(aug_tok.tok)
            if aug_tok.sentbreak:
                yield sentence
                sentence = []
        if sentence:
            yield sentence

    def _annotate_tokens(
        self, tokens: Iterator[PunktToken]
    ) -> Iterator[PunktToken]:
        tokens = self._annotate_first_pass(tokens)
        return self._annotate_second_pass(tokens)

    def _annotate_second_pass(
        self, tokens: Iterator[PunktToken]
    ) -> Iterator[PunktToken]:
        for t1, t2 in _pair_iter(tokens):
            self._second_pass_annotation(t1, t2)
            yield t1

    def _second_pass_annotation(
        self, aug_tok1: PunktToken, aug_tok2: PunktToken | None
    ) -> None:
        """
        Contextual refinement of a token pair (modifies *aug_tok1* in-place).
        """
        if aug_tok2 is None or not aug_tok1.period_final:
            return

        typ = aug_tok1.type_no_period
        next_typ = aug_tok2.type_no_sentperiod
        tok_is_initial = aug_tok1.is_initial

        # Known collocation → not a sentence break
        if (typ, next_typ) in self._params.collocations:
            aug_tok1.sentbreak = False
            aug_tok1.abbr = True
            return

        if (aug_tok1.abbr or aug_tok1.ellipsis) and not tok_is_initial:
            if self._ortho_heuristic(aug_tok2) is True:
                aug_tok1.sentbreak = True
                return
            if aug_tok2.first_upper and next_typ in self._params.sent_starters:
                aug_tok1.sentbreak = True
                return

        if tok_is_initial or typ == "##number##":
            is_sent_starter = self._ortho_heuristic(aug_tok2)

            if is_sent_starter is False:
                aug_tok1.sentbreak = False
                aug_tok1.abbr = True
                return

            if (
                is_sent_starter == "unknown"
                and tok_is_initial
                and aug_tok2.first_upper
                and not (
                    self._params.ortho_context[next_typ] & OrthoContext.lower()
                )
            ):
                aug_tok1.sentbreak = False
                aug_tok1.abbr = True

    def _ortho_heuristic(self, aug_tok: PunktToken) -> bool | str:
        """
        Decide whether *aug_tok* is likely the first token of a sentence.

        Returns ``True``, ``False``, or ``"unknown"``.
        """
        if aug_tok.tok in self.PUNCTUATION:
            return False

        ctx = self._params.ortho_context[aug_tok.type_no_sentperiod]

        if (
            aug_tok.first_upper
            and (ctx & OrthoContext.lower())
            and not (ctx & OrthoContext.MID_UC)
        ):
            return True

        if aug_tok.first_lower and (
            (ctx & OrthoContext.upper()) or not (ctx & OrthoContext.BEG_LC)
        ):
            return False

        return "unknown"

    def sentences_from_text_legacy(
        self, text: str
    ) -> Generator[str, None, None]:
        """Annotate every token (slower); should give identical results."""
        tokens = self._annotate_tokens(self._tokenize_words(text))
        yield from self._build_sentence_list(text, tokens)

    def _build_sentence_list(
        self, text: str, tokens: Iterator[PunktToken]
    ) -> Generator[str, None, None]:
        pos = 0
        ws_re = re.compile(r"\s*")
        sentence = ""

        for aug_tok in tokens:
            tok = aug_tok.tok
            ws = ws_re.match(text, pos).group()
            pos += len(ws)

            # Handle tokens that span whitespace in the source
            if text[pos : pos + len(tok)] != tok:
                pat = r"\s*".join(re.escape(c) for c in tok)
                if m := re.compile(pat).match(text, pos):
                    tok = m.group()

            assert (
                text[pos : pos + len(tok)] == tok
            ), f"Token mismatch at pos {pos}: expected {tok!r}"
            pos += len(tok)

            sentence = (sentence + ws + tok) if sentence else tok

            if aug_tok.sentbreak:
                yield sentence
                sentence = ""

        if sentence:
            yield sentence
