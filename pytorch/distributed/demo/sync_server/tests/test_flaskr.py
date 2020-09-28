import os
import tempfile

import pytest

from flask import json
from flaskr import create_app

def test_hello(client):
    response = client.get('/hello')
    assert response.data == b'Hello, World!'

def test_sync(client):

    mimetype = 'application/json'
    headers = {
        'Content-Type': mimetype,
        'Accept': mimetype
    }

    # Step 1: register from host A.
    hostAsyncRequest = {
        'address': '192.168.1.1',
        'port': 2345,
        'world': 3,
        'groupId': 'pytorch1'
    }
    response = client.post('/register', data=json.dumps(hostAsyncRequest), headers=headers)

    assert response.json['groupId'] == hostAsyncRequest['groupId']
    assert response.json['globalRank'] == 0
    assert response.json['leadServerAddress'] == hostAsyncRequest['address']
    assert response.json['leadServerPort'] == hostAsyncRequest['port']

    # Step 2: register from host B.
    hostBsyncRequest = {
        'address': '192.168.1.3',
        'port': 2345,
        'world': 3,
        'groupId': 'pytorch1'
    }    
    response = client.post('/register', data=json.dumps(hostBsyncRequest), headers=headers)

    # host B are in the same group with host A.
    assert response.json['groupId'] == hostAsyncRequest['groupId']
    # host B's globalRank is after host A.
    assert response.json['globalRank'] == 1
    # master address is host A since its global rank is 0.
    assert response.json['leadServerAddress'] == hostAsyncRequest['address']
    assert response.json['leadServerPort'] == hostAsyncRequest['port']

    # Step 3: register from host A again and nothing changes.
    hostAsyncRequest = {
        'address': '192.168.1.1',
        'port': 2345,
        'world': 3,
        'groupId': 'pytorch1'
    }
    response = client.post('/register', data=json.dumps(hostAsyncRequest), headers=headers)

    assert response.json['groupId'] == hostAsyncRequest['groupId']
    assert response.json['globalRank'] == 0
    assert response.json['leadServerAddress'] == hostAsyncRequest['address']
    assert response.json['leadServerPort'] == hostAsyncRequest['port']

    # Step 4: list all process groups and validate
    response = client.get('/groups', headers=headers)
    processGroups = response.json
 
    assert len(processGroups.items()) == 1
    assert processGroups['pytorch1']['groupId'] == hostAsyncRequest['groupId']
    assert processGroups['pytorch1']['leaderAddress'] == hostAsyncRequest['address']
    assert processGroups['pytorch1']['leaderPort'] == hostAsyncRequest['port']
    assert processGroups['pytorch1']['worldSize'] == hostAsyncRequest['world']
    assert len(processGroups['pytorch1']['processes'].items()) == 2

    # first come server (hostA) is assigned as lead server (global rank == 0)
    assert processGroups['pytorch1']['processes'][hostAsyncRequest['address']]['globalRank'] == 0
    assert processGroups['pytorch1']['processes'][hostAsyncRequest['address']]['address'] == hostAsyncRequest['address']

    assert processGroups['pytorch1']['processes'][hostBsyncRequest['address']]['globalRank'] == 1
    assert processGroups['pytorch1']['processes'][hostBsyncRequest['address']]['address'] == hostBsyncRequest['address']
        
    # Step 5 create a new process group and validate
    # Step 1: register from host A.
    hostCsyncRequest = {
        'address': '192.168.1.7',
        'port': 2345,
        'world': 3,
        'groupId': 'tensorflow1'
    }
    response = client.post('/register', data=json.dumps(hostCsyncRequest), headers=headers)

    assert response.json['groupId'] == 'tensorflow1' # hostCsyncRequest['groupId']
    assert response.json['globalRank'] == 0
    assert response.json['leadServerAddress'] == hostCsyncRequest['address']
    assert response.json['leadServerPort'] == hostCsyncRequest['port']
    
    response = client.get('/groups', headers=headers)
    processGroups = response.json
 
    # has two process groups now, 'pytorch1' and 'tensorflow1'
    assert len(processGroups.items()) == 2
    assert processGroups['tensorflow1']['groupId'] == hostCsyncRequest['groupId']
    assert processGroups['tensorflow1']['leaderAddress'] == hostCsyncRequest['address']
    assert processGroups['tensorflow1']['leaderPort'] == hostCsyncRequest['port']
    assert processGroups['tensorflow1']['worldSize'] == hostCsyncRequest['world']
    assert len(processGroups['tensorflow1']['processes'].items()) == 1
    assert len(processGroups['pytorch1']['processes'].items()) == 2
