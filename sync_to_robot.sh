#!/bin/bash
set -euo pipefail

LOCAL_DIR="/home/arrma/Computer_vision_in_navigation_of_unmanned_robotic_systems/scripts"
REMOTE_HOST="avt-robot"
REMOTE_DIR="/home/avt_user/PROGRAMMS"

sync() {
    rsync -a --exclude='__pycache__/' --exclude='*.pyc' "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/" || true
}

sync

while true; do
    inotifywait -q -r -t 30 -e modify,create,close_write,delete,move,attrib \
        --exclude '__pycache__|\.pyc' "$LOCAL_DIR" >/dev/null 2>&1 || true
    sleep 0.3
    sync
done
