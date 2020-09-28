from flask import json

class SyncResponse:
    """ Return object to trainer process who wants to join the process group """

    # global rank, global unique integer number to identify 
    # the trainer process in the process group.
    globalRank = 0 

    # server address for the lead server to initialize the process group 
    leadServerAddress = ""

    # lead server port.
    leadServerPort = 80

    # process group id.
    groupId = ""

    # trainers (ip) of the process group
    trainers = []

    def __init__(self, gRank: int, address: str, port: int, id: str):
        self.globalRank = gRank
        self.leadServerAddress = address
        self.leadServerPort = port
        self.groupId = id

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)
