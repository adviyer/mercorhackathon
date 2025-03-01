FROM ubuntu:22.04

# Install necessary packages
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    tzdata \
    git \
    x11vnc \
    fluxbox \
    python3 \
    python3-pip \
    python3-tk \
    websockify

RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Clone noVNC to serve in browser
RUN git clone https://github.com/novnc/noVNC.git /opt/novnc

# Set a VNC password
RUN mkdir -p /root/.vnc && \
    x11vnc -storepasswd "vncpassword" /root/.vnc/passwd

# Add the bouncing ball script to the container
COPY bouncing_ball.py /opt/bouncing_ball.py

# Start everything: Xvfb, fluxbox, VNC server, noVNC, and bouncing ball app
CMD Xvfb :99 -screen 0 1920x1080x24 & \
    sleep 2 && \
    fluxbox & \
    sleep 2 && \
    x11vnc -display :99 -rfbport 5900 -usepw -forever & \
    /usr/bin/python3 -m websockify --web /opt/novnc 6080 localhost:5900 & \
    sleep 2 && DISPLAY=:99 python3 /opt/bouncing_ball.py
