
#!/bin/bash

COMMAND=$@

GIT_TOKEN=$(< gitToken)

git $COMMAND https://$GIT_TOKEN@github.com/Brutalfacepalm/project_made.git
