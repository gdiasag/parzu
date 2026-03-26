#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © 2009-2011 University of Zürich
# Author: Rico Sennrich <sennrich@cl.uzh.ch>

from __future__ import unicode_literals

import sys
import os
import shlex
import pexpect
import tempfile
import threading
import codecs
import re
from subprocess import Popen, PIPE

# root directory of ParZu if file is run as script
root_directory = sys.path[0]
# root directory of ParZu if file is loaded as a module
if "__file__" in globals():
    root_directory = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root_directory, "preprocessor"))
import punkt_tokenizer
import treetagger2prolog

sys.path.append(os.path.join(root_directory, "preprocessor", "morphology"))
import morphisto2prolog

sys.path.append(os.path.join(root_directory, "postprocessor"))
import cleanup_output

COMMENT_CHAR = "#"
OPTION_CHAR = "="


def parse_config(filename):
    options = {}
    f = open(filename)
    for line in f:
        # First, remove comments:
        if COMMENT_CHAR in line:
            # split on comment char, keep only the part before
            line, comment = line.split(COMMENT_CHAR, 1)
        # Second, find lines with an option=value:
        if OPTION_CHAR in line:
            # split on option char:
            option, value = line.split(OPTION_CHAR, 1)
            # strip spaces:
            option = option.strip()
            value = value.strip()
            # store in dictionary:
            options[option] = value
    f.close()
    return options


# sanity checks and merging config and command line options
def process_arguments():

    # get config options
    options = parse_config(os.path.join(root_directory, "parzu.ini"))

    options["tempdir"] = os.path.abspath("/tmp")

    uniquetmp = str(os.getpid())

    options["tagfilepath"] = os.path.join(
        options["tempdir"], "tags" + uniquetmp + ".pl"
    )
    options["morphinpath"] = os.path.join(
        options["tempdir"], "morphin" + uniquetmp
    )
    options["errorpath"] = os.path.join(options["tempdir"], "error" + uniquetmp)
    options["morphpath"] = os.path.join(
        options["tempdir"], "morph" + uniquetmp + ".pl"
    )

    options["taggercmd"] = shlex.split(options["taggercmd"])
    options["senderror"] = sys.stderr

    return options


class Parser:

    def __init__(self, options, timeout=10):

        # launch punkt_tokenizer for sentence splitting
        self.punkt_tokenizer = punkt_tokenizer.PunktSentenceTokenizer()
        self.punkt_tokenizer._params.collocations = (
            punkt_tokenizer.punkt_data_german.collocations
        )
        self.punkt_tokenizer._params.ortho_context = (
            punkt_tokenizer.punkt_data_german.ortho_context
        )
        self.punkt_tokenizer._params.abbrev_types = (
            punkt_tokenizer.punkt_data_german.abbrev_types
        )
        self.punkt_tokenizer._params.sent_starters = (
            punkt_tokenizer.punkt_data_german.sent_starters
        )

        # launch moses tokenizer
        tokenizer_cmd = (
            "perl "
            + os.path.join(root_directory, "preprocessor", "tokenizer.perl")
            + " -l de"
        )
        self.tokenizer = pexpect.spawn(
            tokenizer_cmd, echo=False, encoding="utf-8"
        )
        self.tokenizer.expect("Tokenizer v3\r\nLanguage: de\r\n")
        self.tokenizer.delaybeforesend = 0

        # launch clevertagger for POS tagging
        clevertagger_dir = os.path.dirname(options["taggercmd"][0])
        sys.path.append(clevertagger_dir)
        import clevertagger

        self.tagger = clevertagger.Clevertagger()

        # launch SMOR morphological analyzer
        self.morph = pexpect.spawn(
            "fst-infl2",
            ["-q", options["smor_model"]],
            echo=False,
            encoding="utf-8",
        )
        self.morph.delaybeforesend = 0

        # launch morphological preprocessing (prolog script)
        self.prolog_preprocess = pexpect.spawn(
            "swipl",
            [
                "-q",
                "-s",
                os.path.join(
                    root_directory, "preprocessor", "preprocessing.pl"
                ),
            ],
            echo=False,
            encoding="utf-8",
            timeout=timeout,
        )

        self.prolog_preprocess.expect_exact("?- ")
        self.prolog_preprocess.delaybeforesend = 0

        # launch main parser process (prolog script)
        args = ["-q", "-s", "ParZu-parser.pl", "--stack-limit=496M"]

        self.prolog_parser = pexpect.spawn(
            "swipl",
            args,
            echo=False,
            encoding="utf-8",
            cwd=os.path.join(root_directory, "core"),
            timeout=timeout,
        )

        self.prolog_parser.expect_exact("?- ")
        self.prolog_parser.delaybeforesend = 0

        # initialize parser parameters
        parser_init = (
            "retract(sentdelim(_)),"
            + "assert(sentdelim('$newline')),"
            + "retract(returnsentdelim(_)),"
            + "assert(returnsentdelim(no)),"
            + "retract(nbestmode(_)),"
            + "assert(nbestmode(0)),"
            + "retractall(morphology(_)),"
            + "assert(morphology(gertwol)),"
            + "retractall(lemmatisation(_)),"
            + "assert(lemmatisation(gertwol)),"
            + "retractall(extrainfo(_)),"
            + "assert(extrainfo(no)),"
            + "start_german."
        )

        self.prolog_parser.sendline(parser_init)
        self.prolog_parser.expect(".*true.*")

        self.lock_tokenize = threading.Lock()
        self.lock_tag = threading.Lock()
        self.lock_preprocess = threading.Lock()
        self.lock_parse = threading.Lock()
        self.lock_svg = threading.Lock()

        self.options = options

    def __del__(self):
        self.tokenizer.close()
        self.morph.close()
        self.prolog_preprocess.close()
        self.prolog_parser.close()

    def main(self, text, inputformat="plain", outputformat="conll"):
        text = text.strip()
        with self.lock_tokenize:
            sentences = self.tokenize(text, inputformat)

        # strip empty sentences
        # TODO: we may want to retain empty sentences for alignment purposes in parallel data
        # this is currently not supported in parzu_class; use the batch processing mode of ParZu for this
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        with self.lock_tag:
            sentences = self.tag(sentences)

        with self.lock_preprocess:
            preprocessed_path = self.preprocess(sentences)

        with self.lock_parse:
            parsed_path = self.parse(preprocessed_path, outputformat)

        os.remove(preprocessed_path)

        sentences = self.postprocess(parsed_path, outputformat)
        os.remove(parsed_path)

        return sentences

    # sentence splitting and tokenization
    # input: plain text
    # output: one token per line; empty lines mark sentence boundaries
    def tokenize(self, text, inputformat):
        if not text:
            return []

        sentences = self.punkt_tokenizer.tokenize(text)
        # remove line breaks
        sentences = [
            sentence.replace("\n", " ").replace("\r", "")
            for sentence in sentences
        ]

        sentences = process_by_sentence(self.tokenizer, sentences)

        return sentences

    # pos tagging
    # input: list of sentences; each sentence is one token per line
    # output: list of sentences; each sentence is one token per line (token \t tag \n)
    def tag(self, sentences):
        sentences = self.tagger.tag(sentences)

        return sentences

    # convert to prolog-readable format
    # do morphological analysis
    # identify verb complexes
    # input: list of sentences
    # output: path to file in which preprocessed text is written
    def preprocess(self, sentences):
        # convert to prolog format and get vocabulary
        sentences_out = []
        vocab = set()
        for sentence in sentences:
            sentence_out = []
            for line in sentence.splitlines():
                word, line = treetagger2prolog.format_conversion(line)
                sentence_out.append(line)

                # expand word forms for query (to also include spelling variants)
                for variant in treetagger2prolog.spelling_variations(word):
                    vocab.add(variant)

            sentence_out.append(
                "w('ENDOFSENTENCE','$newline',['._$newline'],'ENDOFSENTENCE')."
            )

            sentences_out.append("\n".join(sentence_out))

        sentences_out.append(
            "\nw('ENDOFDOC','$newline',['._$newline'],'ENDOFDOC')."
        )

        analyses = []
        # split vocab into batches to avoid filling buffer
        batch_size = 100
        vocab = list(vocab)
        for i in range(0, len(vocab), batch_size):
            subvocab = "\n".join(vocab[i : i + batch_size]) + "\n\n"

            # do morphological analysis
            self.morph.send(subvocab)

            while True:
                ret = self.morph.readline().strip()
                if ret == "no result for":
                    break
                else:
                    analyses.append(ret)

        # convert morphological analysis to prolog format
        analyses = morphisto2prolog.main(analyses)

        # having at least one entry makes sure that the preprocessing script doesn't crash
        analyses.append("gertwol('<unknown>','<unknown>',_,_,_).")

        # communication with swipl scripts is via temporary files
        morphfile = tempfile.NamedTemporaryFile(
            prefix="ParZu-morph.pl",
            dir=os.path.join(self.options["tempdir"]),
            delete=False,
        )
        morphfile.close()
        codecs.open(morphfile.name, "w", encoding="UTF-8").write(
            "\n".join(analyses)
        )

        tagfile = tempfile.NamedTemporaryFile(
            prefix="ParZu-tag.pl",
            dir=os.path.join(self.options["tempdir"]),
            delete=False,
        )
        tagfile.close()
        codecs.open(tagfile.name, "w", encoding="UTF-8").write(
            "\n".join(sentences_out)
        )

        preprocessedfile = tempfile.NamedTemporaryFile(
            prefix="ParZu-preprocessed.pl",
            dir=os.path.join(self.options["tempdir"]),
            delete=False,
        )
        preprocessedfile.close()

        # start preprocessing script and wait for it to finish
        self.prolog_preprocess.sendline(
            "retractall(gertwol(_,_,_,_,_)),retractall(lemmatisation(_)),retractall(morphology(_)),assert(lemmatisation(smor)),assert(morphology(smor)),retract(sentdelim(_)),assert(sentdelim('$newline')),start('"
            + morphfile.name
            + "','"
            + tagfile.name
            + "','"
            + preprocessedfile.name
            + "')."
        )

        while True:
            line = self.prolog_preprocess.readline()
            line = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", line)

            if re.match("(\?-\s)?true\.\r\n", line):
                break

        # clean up temporary files
        os.remove(morphfile.name)
        os.remove(tagfile.name)

        return preprocessedfile.name

    # main parsing step
    # input: path to input file (produced by preprocess())
    # output: path to output file (temporary file created by function)
    def parse(self, inpath, outputformat):
        parsedfile = tempfile.NamedTemporaryFile(
            prefix="ParZu-parsed.pl",
            dir=os.path.join(self.options["tempdir"]),
            delete=False,
        )
        parsedfile.close()

        cmd = (
            "retract(outputformat(_)),"
            + "assert(outputformat("
            + outputformat
            + ")),"
            + "go_textual('"
            + inpath
            + "','"
            + parsedfile.name
            + "').\n"
        )

        self.prolog_parser.sendline(cmd)

        while True:
            line = self.prolog_parser.readline()
            line = re.sub(
                r"\x1B\[[0-?]*[ -/]*[@-~]", "", line
            )  # remove styling tokens
            if re.match("(\?-)?(\s+\|\s+)?true\.\r\n", line):
                break

        return parsedfile.name

    # de-projectivization and removal of debugging output
    # input: path to input file (produced by parse)
    # output:
    def postprocess(self, inpath, outputformat):
        infile = codecs.open(inpath, encoding="UTF-8")

        if outputformat == "prolog":
            sentences = list(cleanup_output.cleanup_prolog(infile))
        else:
            sentences = list(cleanup_output.cleanup_conll(infile))

        return sentences


def process_by_sentence(processor, sentences):
    sentences_out = []
    for sentence in sentences:
        words = []
        processor.send(sentence + "\n")
        while True:
            word = processor.readline().strip()
            if word:
                words.append(word)
            else:
                break
        sentences_out.append("\n".join(words))

    return sentences_out
