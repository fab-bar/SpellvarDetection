import os

from flask import Flask

from spellvardetection.rest import db, resources, routes
from spellvardetection.util.spellvarfactory import create_base_factory

def create_app(test_config=None):

    app = Flask(__name__, instance_relative_config=True)
    # default config
    app.config.from_mapping(
        RESOURCES_PATH=os.path.join(app.instance_path, 'resources'),
        DATABASE=os.path.join(app.instance_path, 'resources.db')
    )

    # load config
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # make relative paths relative to resource path
    def change_path(x):
        return x if os.path.isabs(x) else os.path.join(app.config['RESOURCES_PATH'], x)

    factory = create_base_factory()
    factory.add_factory_method(os.PathLike, change_path)

    # add factory to config -- after loading config so it cannot be overridden
    app.config['FACTORY'] = factory

    db.init_app(app)
    resources.init_app(app)
    app.register_blueprint(routes.app_blueprint)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    return app
