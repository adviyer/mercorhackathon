import tkinter as tk

class BouncingBallApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bouncing Ball Simulation")
        self.canvas = tk.Canvas(root, width=600, height=400, bg="white")
        self.canvas.pack()

        self.ball = self.canvas.create_oval(10, 10, 50, 50, fill="red")

        self.dx = 4
        self.dy = 4

        self.animate()

    def animate(self):
        self.canvas.move(self.ball, self.dx, self.dy)
        pos = self.canvas.coords(self.ball)

        if pos[2] >= 600 or pos[0] <= 0:
            self.dx = -self.dx
        if pos[3] >= 400 or pos[1] <= 0:
            self.dy = -self.dy

        self.root.after(20, self.animate)

if __name__ == "__main__":
    root = tk.Tk()
    app = BouncingBallApp(root)
    root.mainloop()
