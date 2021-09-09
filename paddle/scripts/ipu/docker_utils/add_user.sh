# add user to docker container
addgroup --gid "$DOCKER_GRP_ID" "$DOCKER_GRP"
adduser --disabled-password --gecos '' "$DOCKER_USER" \
    --uid "$DOCKER_USER_ID" --gid "$DOCKER_GRP_ID" 2>/dev/null
usermod -aG sudo "$DOCKER_USER"
echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
cp -r /etc/skel/. /home/${DOCKER_USER}
chown -R ${DOCKER_USER}:${DOCKER_GRP} "/home/${DOCKER_USER}"
