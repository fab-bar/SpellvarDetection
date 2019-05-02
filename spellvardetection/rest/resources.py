import os
import shutil

import click
from flask import current_app
from flask.cli import AppGroup

### CLI for management of additional resources

res_cli = AppGroup('resources', short_help='Manage additional resources used by SpellvarDetection.')

@res_cli.command('list')
def list_resources():
    "List existing resources."

    click.echo('\n'.join(os.listdir(current_app.config['RESOURCES_PATH'])))

@res_cli.command('add')
@click.argument('filename')
def list_resources(filename):
    "Add a resource."

    if os.path.exists(os.path.join(current_app.config['RESOURCES_PATH'], os.path.basename(filename))):
        click.echo('File does already exist in resource folder.')
    else:
        try:
            newname = shutil.copy(filename, current_app.config['RESOURCES_PATH'])
        except IOError as e:
            print(e)
        else:
            click.echo('Added ' + os.path.basename(newname) + ' to the resources.')

@res_cli.command('remove')
@click.argument('filename')
def list_resources(filename):
    "Remove a resources."

    try:
        os.remove(os.path.join(current_app.config['RESOURCES_PATH'], filename))
    except IOError as e:
       print(e)
    else:
        click.echo('Removed ' + filename + ' from the resources.')

def init_app(app):
    app.cli.add_command(res_cli)
