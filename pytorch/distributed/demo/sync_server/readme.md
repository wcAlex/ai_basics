# Run Test
https://flask.palletsprojects.com/en/1.1.x/testing/, test with flask.
```

pytest                  // run tests
coverage run -m pytest  // run test with coverage
coverage report         // review coverage report

```

# Docker build & run
move to sync_server folder.

```
# build image sync_server
$ docker build -t alex1005/dist_syncserver:1.0 -f Dockerfile .

# run the server inside container
$ docker run -d -p 5000:8080 alex1005/dist_syncserver:1.0 sync_server

# test request
$ curl -X GET http://127.0.0.1:5000/hello

```