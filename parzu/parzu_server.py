#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import logging

from flask import Flask, request, Response

from parzu_class import Parser, process_arguments


class Server(object):

    def __init__(self, timeout=10, options=None):
        if options is None:
            options = process_arguments()
            options["extrainfo"] = "secedges"

        self.parser = Parser(options, timeout=timeout)
        self.app = Flask("ParZuServer")

        @self.app.route("/parse", methods=["POST"])
        def parse():
            input = request.get_json(force=True)
            text = input.get("text")

            parses = self.parser.main(text)

            result = "\n".join(parses)

            return Response(result, mimetype="text/plain")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout for each step in pipeline (total processing time may be longer).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(name)-12s %(levelname)-5s] %(message)s",
    )

    server = Server(timeout=args.timeout)

    server.app.run(port="5003", host="0.0.0.0", threaded=True)
