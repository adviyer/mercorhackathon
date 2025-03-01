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
    x11-utils \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    xauth \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install pygame

# Create app directory
WORKDIR /app

# Copy the Python applications
COPY bouncing_ball.py /app/
COPY fallback_bouncing_ball.py /app/

# Create a wrapper script to try multiple approaches
RUN echo '#!/bin/bash \n\
echo "Attempting to run Pygame version..." \n\
python3 /app/bouncing_ball.py \n\
\n\
if [ $? -ne 0 ]; then \n\
    echo "Pygame version failed, trying Tkinter fallback..." \n\
    python3 /app/fallback_bouncing_ball.py \n\
fi \n\
' > /app/run_game.sh && chmod +x /app/run_game.sh

# Create startup script
RUN echo '#!/bin/bash \n\
# Start virtual X server with error output\n\
Xvfb :1 -screen 0 1024x768x16 -ac 2>&1 & \n\
XVFB_PID=$! \n\
export DISPLAY=:1 \n\
\n\
# Wait for X server to start properly\n\
echo "Waiting for X server to start..." \n\
sleep 3 \n\
\n\
# Check if X server is running\n\
if ! ps -p $XVFB_PID > /dev/null; then\n\
    echo "ERROR: X server failed to start" >&2\n\
    exit 1\n\
fi\n\
\n\
# Start VNC server with debugging info\n\
echo "Starting VNC server..." \n\
x11vnc -display :1 -forever -nopw -noshared -noxdamage -bg -o /tmp/x11vnc.log \n\
\n\
# Start noVNC\n\
echo "Starting noVNC..." \n\
/usr/share/novnc/utils/launch.sh --vnc localhost:5900 --listen 80 & \n\
\n\
# Wait to ensure services are up\n\
sleep 5 \n\
\n\
# Check and display some debug info\n\
echo "Display environment: $DISPLAY" \n\
echo "Checking if X11 socket exists:" \n\
ls -la /tmp/.X11-unix/ \n\
\n\
# Try a simple X command to test\n\
echo "Testing X server with xdpyinfo:" \n\
DISPLAY=:1 xdpyinfo | head -5 || echo "X server not responding to xdpyinfo" \n\
\n\
echo "Starting game application..." \n\
/app/run_game.sh 2>&1 \n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose only HTTP port
EXPOSE 80

# Run the application
CMD ["/app/start.sh"]