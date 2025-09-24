#!/bin/bash
# 备份Grafana容器数据
BACKUP_DIR=/data/grafana_backup_$(date +%Y%m%d)
DOCKER_NAME=grafana
mkdir -p $BACKUP_DIR

docker cp $DOCKER_NAME:/var/lib/grafana $BACKUP_DIR/data
docker cp $DOCKER_NAME:/etc/grafana $BACKUP_DIR/config

# 压缩备份
tar -czvf grafana_backup.tar.gz $BACKUP_DIR
echo "备份已保存到: grafana_backup.tar.gz"

