import os
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import requests
import argparse
import io

# Creating an argparser to parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("-u", "--url", help="Pass API URL")
args = parser.parse_args()
API_URL = args.url if args.url else "http://127.0.0.1:8000"

def predict_image(image):
    # Convert PIL image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Send image to API
    files = {'file': ('image.png', img_byte_arr, 'image/png')}
    try:
        response = requests.post(f"{API_URL}/predict-file", files=files)
        if response.status_code == 200:
            return response.json()["latex"]
        else:
            return f"Error: {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"Connection Error: {str(e)}"

# Drawing application class
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Math to LaTeX")

        # Canvas settings
        self.canvas_size = 600  # Canvas size in pixels
        self.brush_size = 3  # Size of the brush
        self.brush_color = "black"
        self.canvas_color = "white"

        # Main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)

        # Canvas for drawing
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(side="left", padx=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg=self.canvas_color, 
                                width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Creation of the image for drawing
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), self.canvas_color)
        self.draw = ImageDraw.Draw(self.image)
        
        # Text display area for LaTeX result
        self.result_frame = tk.Frame(self.main_frame)
        self.result_frame.pack(side="right", padx=10, fill="both", expand=True)
        
        self.result_label = tk.Label(self.result_frame, text="LaTeX Result:")
        self.result_label.pack(anchor="w")
        
        self.latex_display = tk.Text(self.result_frame, height=10, width=40)
        self.latex_display.pack(fill="both", expand=True)

        # Control buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)
        
        self.predict_button = tk.Button(self.button_frame, text="Convert to LaTeX", 
                                        command=self.predict_drawing, padx=10)
        self.predict_button.pack(side="left", padx=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear Canvas", 
                                      command=self.clear_canvas, padx=10)
        self.clear_button.pack(side="left", padx=5)
        
        self.save_button = tk.Button(self.button_frame, text="Save Image", 
                                     command=self.save_image, padx=10)
        self.save_button.pack(side="left", padx=5)
        
        self.load_button = tk.Button(self.button_frame, text="Load Image", 
                                     command=self.load_image, padx=10)
        self.load_button.pack(side="left", padx=5)

        # Brush size control
        self.brush_frame = tk.Frame(root)
        self.brush_frame.pack(pady=5)
        
        self.brush_label = tk.Label(self.brush_frame, text="Brush Size:")
        self.brush_label.pack(side="left")
        
        self.brush_slider = tk.Scale(self.brush_frame, from_=1, to=10, orient="horizontal", 
                                    command=self.update_brush_size)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side="left", padx=5)

        # Drawing events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        # For smooth line drawing
        self.old_x = None
        self.old_y = None

    def update_brush_size(self, val):
        self.brush_size = int(val)

    def paint(self, event):
        x, y = event.x, event.y
        
        if self.old_x and self.old_y:
            # Draw on the canvas (screen)
            self.canvas.create_line(self.old_x, self.old_y, x, y, 
                                   width=self.brush_size*2, fill=self.brush_color,
                                   capstyle=tk.ROUND, smooth=True)
            
            # Draw on the Image object
            self.draw.line([self.old_x, self.old_y, x, y], 
                          fill=self.brush_color, width=self.brush_size*2)
        
        self.old_x = x
        self.old_y = y

    def reset(self, event):
        self.old_x = None
        self.old_y = None

    def predict_drawing(self):
        # Convert to grayscale for the model
        gray_image = self.image.convert('L')
        
        # Get prediction from API
        latex = predict_image(gray_image)
        
        # Display result
        self.latex_display.delete(1.0, tk.END)
        self.latex_display.insert(tk.END, latex)

    def clear_canvas(self):
        # Clears the canvas and creates a new image
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), self.canvas_color)
        self.draw = ImageDraw.Draw(self.image)
        self.latex_display.delete(1.0, tk.END)

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                               filetypes=[("PNG files", "*.png"), 
                                                         ("All files", "*.*")])
        if file_path:
            self.image.save(file_path)
            messagebox.showinfo("Success", "Image saved successfully!")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg"), 
                                                         ("All files", "*.*")])
        if file_path:
            try:
                loaded_image = Image.open(file_path)
                
                # Resize if needed
                if loaded_image.size != (self.canvas_size, self.canvas_size):
                    loaded_image = loaded_image.resize((self.canvas_size, self.canvas_size))
                
                # Update canvas and image
                self.image = loaded_image
                self.draw = ImageDraw.Draw(self.image)
                
                # Display on canvas
                self.canvas.delete("all")
                tk_image = ImageTk.PhotoImage(self.image)
                self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
                self.canvas.image = tk_image  # Keep a reference
                
                messagebox.showinfo("Success", "Image loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

# Application initialization
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()