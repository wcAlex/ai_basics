# setup

1. install sync_server
sync_server response to organize distribute process group, it will assign the first process in the group as leader.
k apply -f helm/sync_server/deployment.yaml
k apply -f helm/sync_server/service.yaml

see more details in sync_server/readme.md

2. deploy trainer.
<TODO, add timestamp for time comparison (total time, each batch, each epoch, batch size)>

perform distributed trainer, it also could train as a single process.
k apply -f helm/trainers/pv-volume.yaml
k apply -f helm/trainers/pv-claim.yaml
k apply -f helm/trainers/pytorch/deployment.yaml

see mode details on debugging pytorch-trainer in trainer/pytorch/readme.md

>> for pytorch, there are two prerequisites:
    1) set a persistent volume, which is built by hostPath in minikube. To overcome the permission issue, we could ssh to minikube and give full permission to the volume folder. See more details in helm/trainer/pv-volume.yaml
    2) preload dataset (optional), to boost speed, we could copy the dataset to the shared volume, see more details in 