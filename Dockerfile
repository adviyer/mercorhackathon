FROM ubuntu:20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    xvfb \
    x11vnc \
    novnc \
    wget \
    unzip \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install pygame

# Create app directory
WORKDIR /app

# Copy the Python application
COPY bouncing_ball.py /app/

# Create startup script
RUN echo '#!/bin/bash \n\
Xvfb :1 -screen 0 1024x768x16 & \n\
export DISPLAY=:1 \n\
x11vnc -display :1 -forever -nopw -quiet & \n\
/usr/share/novnc/utils/launch.sh --vnc localhost:5900 --listen 80 & \n\
sleep 2 \n\
python3 /app/bouncing_ball.py \n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose only HTTP port
EXPOSE 80

# Run the application
CMD ["/app/start.sh"]
