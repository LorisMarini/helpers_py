#!/usr/bin/env bash

git submodule init transcrypt

if [ -z "$1" ]; then
  echo "Decrypt password for transcrypt required."
  read -s -p "Enter password: " PASSWORD
else
  PASSWORD=$1
fi

./transcrypt/transcrypt -c aes-256-cbc -p "$PASSWORD" --force
