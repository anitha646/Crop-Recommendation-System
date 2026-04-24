#!/bin/bash
# ═══════════════════════════════════════════════
# run.sh — Crop Advisor AI Docker Manager
# Usage:  ./run.sh build | run | push | all
# ═══════════════════════════════════════════════
IMAGE="anitharajan/crop-advisor-ai:v1"
NAME="crop-advisor-ai"
PORT=5000

case "$1" in
  build)
    echo "Building $IMAGE ..."
    docker build -t $IMAGE .
    echo "Done! Image: $IMAGE"
    ;;
  run)
    docker stop $NAME 2>/dev/null; docker rm $NAME 2>/dev/null
    docker run -d --name $NAME -p $PORT:5000 \
      -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
      --restart unless-stopped $IMAGE
    echo "Running at http://localhost:$PORT"
    ;;
  push)
    docker login
    docker push $IMAGE
    echo "Pushed to Docker Hub: https://hub.docker.com/r/anitharajan/crop-advisor-ai"
    ;;
  stop)
    docker stop $NAME && docker rm $NAME
    ;;
  export)
    docker save $IMAGE | gzip > crop_advisor_ai.tar.gz
    echo "Exported: crop_advisor_ai.tar.gz"
    ;;
  all)
    bash $0 build && bash $0 run
    ;;
  *)
    echo "Usage: ./run.sh [build|run|push|stop|export|all]"
    ;;
esac
