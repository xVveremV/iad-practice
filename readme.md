## Docker-related

### 1. install docker

```shell
open https://www.docker.com
```

### 2. go to folder with image definition

```shell
cd dev-env
```

### 3. build docker image

```shell
docker compose build
```

as a result we will have image named `dev-env-ida-jupiter-notebook`

### 4. docker look for image built

```shell
docker image ls
```

```terminaloutput
REPOSITORY                        TAG        IMAGE ID       CREATED             SIZE
dev-env-ida-jupiter-notebook      latest     1eb44c814e2a   About an hour ago   1.16GB
```

### 5. start container

```shell
docker compose up -d
```

```terminaloutput
 ✔ Network dev-env_default               Created 
 ✔ Container dev-env-jupiter-notebook-1  Started
```

### 6. check if container running

```shell
docker ps
```

```terminaloutput
CONTAINER ID   IMAGE                          COMMAND                  CREATED          STATUS          PORTS                                         NAMES
435d3380698c   dev-env-ida-jupiter-notebook   "/usr/bin/tini -- ba…"   55 seconds ago   Up 55 seconds   0.0.0.0:8888->8888/tcp, [::]:8888->8888/tcp   dev-env-jupiter-notebook-1
```

### 7. start jupiter notebook (will open in your browser)

```shell
open http://localhost:8888
```

### 8. stop container (when you are done, need to free resources or turn off your computer)
```shell
docker stop dev-env-jupiter-notebook-1
```

### 9. remove unused docker container (in the end of the course)
```shell
docker ps -a --filter "ancestor=dev-env-ida-jupiter-notebook" -q | xargs -r docker rm -f && docker rmi -f dev-env-ida-jupiter-notebook
```
