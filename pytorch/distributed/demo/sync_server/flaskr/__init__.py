import os

from flask import Flask, request, jsonify, Response, json
from .sync_manager import SyncManager
from .sync_manager import TrainerProcess
from .sync_response import SyncResponse

syncManager = SyncManager()

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # a simple page that says hello
    # flask rest api examples: http://blog.luisrei.com/articles/flaskrest.html
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    @app.route('/register', methods=['POST'])
    def sync():
        if request.method == 'POST':
            data = request.json
            trainer = TrainerProcess(
                data.get('address'),
                data.get('port'), 
                0)
            worldSize = data.get('world')
            groupId = data.get('groupId')

            response = syncManager.register(groupId, worldSize, trainer)
            js = response.toJson()
            return Response(js, status=200, mimetype='application/json')                
        else:
            message = {
                        'status': 400,
                        'message': 'Only support POST request',
                }
            resp = jsonify(message)
            resp.status_code = 400
            return resp

    @app.route('/groups')
    def listGroups():
        js = syncManager.encodeJson()
        return Response(js, status=200, mimetype='application/json')                

    return app