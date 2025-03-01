import tkinter as tk
import time
import socket

class BouncingBallApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bouncing Ball (Fallback Version)")
        
        # Configure window
        self.width = 800
        self.height = 600
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="black")
        self.canvas.pack()
        
        # Ball properties
        self.ball_radius = 30
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.dx = 5
        self.dy = 4
        
        # Create ball
        self.ball = self.canvas.create_oval(
            self.ball_x - self.ball_radius, 
            self.ball_y - self.ball_radius,
            self.ball_x + self.ball_radius, 
            self.ball_y + self.ball_radius, 
            fill="red"
        )
        
        # Display connection info
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except:
            ip = "Unknown"
            
        self.ip_text = self.canvas.create_text(
            100, 20, text=f"Server IP: {ip}", fill="blue", font=("Arial", 14)
        )
        self.vnc_text = self.canvas.create_text(
            100, 50, text="VNC Port: 5900", fill="blue", font=("Arial", 14)
        )
        
        # Start animation
        self.animate()
        
    def animate(self):
        # Move ball
        self.ball_x += self.dx
        self.ball_y += self.dy
        
        # Check for wall collisions
        if self.ball_x - self.ball_radius <= 0 or self.ball_x + self.ball_radius >= self.width:
            self.dx = -self.dx
        if self.ball_y - self.ball_radius <= 0 or self.ball_y + self.ball_radius >= self.height:
            self.dy = -self.dy
            
        # Update ball position
        self.canvas.coords(
            self.ball,
            self.ball_x - self.ball_radius,
            self.ball_y - self.ball_radius,
            self.ball_x + self.ball_radius,
            self.ball_y + self.ball_radius
        )
        
        # Schedule next frame
        self.root.after(16, self.animate)  # ~60 FPS

# Create and run the application
if __name__ == "__main__":
    print("Starting Tkinter bouncing ball application...")
    try:
        root = tk.Tk()
        app = BouncingBallApp(root)
        print("Application created, starting main loop")
        root.mainloop()
    except Exception as e:
        print(f"ERROR: {e}")