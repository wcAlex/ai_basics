# cd demo/trainer/pytorch
docker build -t alex1005/disttrainer:pytorch-1 . -f Dockerfile

# run sync server and get server ip
docker run -d -p 5000:8080 alex1005/dist_syncserver:1.0 sync_server

# run single trainer (PS, image needs be put at the end to get ENV working)
docker run -d --env TRAINER_PORT=12355 --env GROUP_ID=pytorch1 --env WORLD_SIZE=1 --env SYNC_SERVER="http://172.17.0.2:8080/" --env DATA_PATH="/app/mnt/pytorch/data" --env MODEL_PATH="/app/mnt/pytorch/models" -v /home/chi/workspace/ai_basics/pytorch/distributed/demo/mount:/app/mnt/pytorch:rw --privileged alex1005/disttrainer:pytorch-1

