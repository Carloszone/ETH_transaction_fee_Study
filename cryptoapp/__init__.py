import os

from flask import Flask


def create_app(test_config=None):
    # create and configure the cryptoapp
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev')