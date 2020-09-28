
import threading
from flask import json
from .sync_response import SyncResponse

class TrainerProcess:
    """ training process, no need to consider local rank 
    since we are using kubernates, each pod only has one GPU """

    address = ""
    port = 80
    globalRank = 0

    def __init__(self, address: str, port: int, gRank: int):
        self.address = address
        self.port = port
        self.globalRank = gRank

    def toJson(self):
        return self.__dict__

class ProcessGroup:
    """ distributed training process group """

    # unique id of the group
    groupId = ""
    
    # trainer processes of the group
    processes = None

    # leader trainer server address
    leaderAddress = ""

    # leader trainer server port
    leaderPort = ""

    # total expected trainers in the group
    worldSize = 0

    def __init__(self, groupId: str, worldSize: int):
        self.groupId = groupId
        self.worldSize = worldSize
        self.processes = {}

    def add(self, trainer: TrainerProcess):
        if trainer.address in self.processes:
            return

        self.processes[trainer.address] = trainer

    def get(self, address: str):
        if not address in self.processes:
            return None

        return self.processes[address]

    def exist(self, trainer: TrainerProcess) -> bool:
        if trainer.address in self.processes:
            return True

        return False

    def size(self) -> int:
        return len(self.processes)

    def toJson(self):
        return dict(
            groupId = self.groupId,
            processes = self.processes,
            leaderAddress = self.leaderAddress,
            leaderPort = self.leaderPort,
            worldSize = self.worldSize)

class ProcessGroupEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'toJson'):
            return obj.toJson()
        else:
            return json.JSONEncoder.default(self, obj)

class SyncManager:
    """ track distributed training process group information """

    processGroups = {}
    threadLock = threading.Lock()

    def exist(self, groupId: str) -> bool:
        return groupId in self.processGroups

    def get(self, groupId: str) -> ProcessGroup:
        if not self.exist(groupId):
            return None
        
        return self.processGroups[groupId]

    def toJson(self):
        return self.__dict__

    def encodeJson(self):
        return json.dumps(self.processGroups, cls=ProcessGroupEncoder)

    # register trainer in process group and return the lead server information.
    def register(self, groupId: str, worldSize: int, trainer: TrainerProcess) -> SyncResponse:
        self.threadLock.acquire()
        
        if groupId in self.processGroups:
            # add trainer
            processGroup = self.get(groupId)

            # add trainer to process group and assign global rank
            if not processGroup.exist(trainer):
                curTrainerCnt = processGroup.size()
                trainer.globalRank = curTrainerCnt
                processGroup.add(trainer)

        else:
            # create new process group and 
            # appoint the first come trainer the leader
            processGroup = ProcessGroup(groupId, worldSize)
            processGroup.leaderAddress = trainer.address
            processGroup.leaderPort = trainer.port
            trainer.globalRank = 0
            processGroup.add(trainer)

            self.processGroups[groupId] = processGroup

        self.threadLock.release()

        group = self.processGroups[groupId]
        trainer = group.get(trainer.address)

        response = SyncResponse(trainer.globalRank, group.leaderAddress, group.leaderPort, groupId)
        return response
        


    