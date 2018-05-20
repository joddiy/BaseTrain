#!/bin/bash
# 一键发布工具

# 使用前需要提权 chmod +x ./git.sh

git add -A
git commit -m 'commit'
git pull
git push
