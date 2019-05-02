import os

from flask import Flask

from spellvardetection.rest import db, routes

def create_app(test_config=None):

    app = Flask(__name__, instance_relative_config=True)
    # default config
    app.config.from_mapping(
        DATABASE=os.path.join(app.instance_path, 'resources.db')
    )

    # load config
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    db.init_app(app)
    app.register_blueprint(routes.app_blueprint)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    return app

