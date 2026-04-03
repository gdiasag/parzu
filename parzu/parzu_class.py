#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © 2009-2011 University of Zürich
# Author: Rico Sennrich <sennrich@cl.uzh.ch>

from __future__ import unicode_literals

import sys
import os
import shlex
from typing import Any
import pexpect
import tempfile
import threading
import codecs
import re

import asyncio
from contextlib import asynccontextmanager

from .tokenizer import Tokenizer

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


class ParserPool:
    def __init__(self, config, pool_size=4):
        self.config = config
        self.pool_size = pool_size
        self._pool = asyncio.Queue()

    async def setup(self):
        """Initialize N parsers in parallel at startup."""
        print(f"Initializing pool of {self.pool_size} parsers...")

        def create_parser():
            return Parser(self.config)

        tasks = [
            asyncio.to_thread(create_parser) for _ in range(self.pool_size)
        ]
        parsers = await asyncio.gather(*tasks)

        for p in parsers:
            self._pool.put_nowait(p)

    @asynccontextmanager
    async def get_parser(self):
        parser = await self._pool.get()
        try:
            yield parser
        finally:
            self._pool.put_nowait(parser)

    async def parse(self, text: str):
        async with self.get_parser() as parser:
            result = await asyncio.to_thread(parser.main, text)
            return result


class Parser:
    def __init__(self, config, timeout=60):
        self.config = config
        self.punkt_tokenizer = punkt_tokenizer.PunktSentenceTokenizer()
        self.tokenizer = Tokenizer()

        # launch clevertagger for POS tagging
        sys.path.append(config.tagger_dir)
        import clevertagger

        self.tagger = clevertagger.Clevertagger()

        # launch SMOR morphological analyzer
        self.morph = pexpect.spawn(
            "fst-infl2",
            ["-q", config.smor_model],
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

    def __del__(self):
        # self.tokenizer.close()
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

        # sentences = process_by_sentence(self.tokenizer, sentences)
        sentences = self.tokenizer.tokenize_sentences(sentences)

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
            dir=os.path.join(self.config.tmp_dir),
            delete=False,
        )
        morphfile.close()
        codecs.open(morphfile.name, "w", encoding="UTF-8").write(
            "\n".join(analyses)
        )

        tagfile = tempfile.NamedTemporaryFile(
            prefix="ParZu-tag.pl",
            dir=os.path.join(self.config.tmp_dir),
            delete=False,
        )
        tagfile.close()
        codecs.open(tagfile.name, "w", encoding="UTF-8").write(
            "\n".join(sentences_out)
        )

        preprocessedfile = tempfile.NamedTemporaryFile(
            prefix="ParZu-preprocessed.pl",
            dir=os.path.join(self.config.tmp_dir),
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
            dir=os.path.join(self.config.tmp_dir),
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
