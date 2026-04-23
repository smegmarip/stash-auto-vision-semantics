#!/bin/bash
set -e

# Ensure Redis data directory exists
mkdir -p /data/redis

# Start embedded Redis (daemonized)
redis-server --daemonize yes --dir /data/redis --appendonly yes --appendfsync everysec

# Wait for Redis to be ready
until redis-cli ping > /dev/null 2>&1; do
    sleep 0.1
done

echo "Redis ready"

# Start SSH server
mkdir -p /run/sshd
sed -i 's/#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
if [ -n "$SSH_PASSWORD" ]; then
    echo "root:$SSH_PASSWORD" | chpasswd
fi
ssh-keygen -A 2>/dev/null
/usr/sbin/sshd
echo "SSH ready (port 22)"

# Start the semantics service (single worker)
exec uvicorn app.main:app --host 0.0.0.0 --port 5004 --workers 1
