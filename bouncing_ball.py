import pygame
import sys
import os

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bouncing Ball")

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Ball properties
ball_radius = 30
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_speed_x = 5
ball_speed_y = 4

# Font for displaying connection info
font = pygame.font.SysFont('Arial', 24)

# Create text to display IP address
def get_ip_address():
    # This will work when run on the server to show its external IP
    # Note: In a real environment, you might need to use a different approach
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unknown IP"

# Clock for controlling frame rate
clock = pygame.time.Clock()
FPS = 60

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Update ball position
    ball_x += ball_speed_x
    ball_y += ball_speed_y
    
    # Bounce off walls
    if ball_x <= ball_radius or ball_x >= WIDTH - ball_radius:
        ball_speed_x = -ball_speed_x
    if ball_y <= ball_radius or ball_y >= HEIGHT - ball_radius:
        ball_speed_y = -ball_speed_y
    
    # Clear screen
    screen.fill(BLACK)
    
    # Draw ball
    pygame.draw.circle(screen, RED, (ball_x, ball_y), ball_radius)
    
    # Draw connection info
    ip_text = font.render(f"Server IP: {get_ip_address()}", True, BLUE)
    vnc_text = font.render(f"VNC Port: 5900", True, BLUE)
    screen.blit(ip_text, (10, 10))
    screen.blit(vnc_text, (10, 40))
    
    # Update display
    pygame.display.flip()
    
    # Control frame rate
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
sys.exit()