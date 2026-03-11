#!/bin/bash

set -ex

echo 'alias ll="ls -al"' >> ~/.bashrc
git config --global --add safe.directory "*"
git config --global user.name "Peter Großmann"
git config --global user.email petergrossmann@email.de

# Update repository
git fetch origin
git rebase --autostash origin/main

# Install dependencies
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

pkill python;
# npm run build_and_start;
