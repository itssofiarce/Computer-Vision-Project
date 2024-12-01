# Introducción


# Setup con Docker
Levantar el contenedor siguiendo los pasos:


docker build -t opencv-webcam .
docker run -it -v $PWD:/app/ --device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY opencv-webcam bash