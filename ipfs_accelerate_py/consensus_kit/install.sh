#!/bin/sh
sudo apt install -y build-essential clang cmake pkg-config libssl-dev protobuf-compiler git curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -L https://foundry.paradigm.xyz | bash 
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo apt update
sudo apt inscargo install --force cargo-maketall -y build-essential libssl-dev mesa-opencl-icd ocl-icd-opencl-dev gcc git bzr jq pkg-config curl clang hwloc libhwloc-dev wget ca-certificates gnupg -y
alias ipc-cli="cargo run -q -p ipc-cli --release --"
ipc-cli config init
