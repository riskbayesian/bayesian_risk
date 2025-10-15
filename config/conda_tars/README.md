DEPRECATED
Using Docker Hub to store Docker images instead, much easier. 

This folder is specifically for holding tar files of conda envs. 
The Docker server, specified by the Dockerfile within config will host these
and the other Docker images will curl these. 

Doing this INSTEAD of a COPY command within the Dockerfile will reduce the size 
of the Docker Image CONSIDERABLY. 
Each COPY command forcibly makes a new layer within the image. However if we 
curl tar, extract tar, delete tar within the same RUN command, this will not
add size since these are all within the same layer.  
