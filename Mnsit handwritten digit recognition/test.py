import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw
import io
import torch
from torchvision.transforms import functional as F
import torchvision

# Load your trained model
model = torchvision.models.regnet.regnet_x_400mf()
model.load_state_dict(torch.load('MNIST_5_acc_0.9717.pth', map_location=torch.device('cpu')))
model.eval()

class DigitRecognizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Recognizer")

        self.canvas = Canvas(master, bg="white", width=200, height=200)
        self.canvas.pack()

        self.button_recognize = Button(master, text="Recognize", command=self.recognize_digit)
        self.button_recognize.pack()

        self.button_redraw = Button(master, text="Redraw", command=self.redraw_canvas)
        self.button_redraw.pack()

        self.label_result = tk.Label(master, text="")
        self.label_result.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)

    def recognize_digit(self):
        # Convert the drawn content to a PIL Image
        image = self.convert_canvas_to_image()

        # Get the predicted digit
        digit = self.predict_digit(image)

        # Display the result
        self.label_result.config(text=f"Predicted Digit: {digit}")

    def redraw_canvas(self):
        # Clear the canvas
        self.canvas.delete("all")

        # Clear the result label
        self.label_result.config(text="")

    def convert_canvas_to_image(self):
        # Create a blank image with a white background
        image = Image.new("L", (200, 200), 255)
        draw = ImageDraw.Draw(image)

        # Draw the content of the canvas onto the image
        image_data = self.canvas.postscript(colormode="gray")
        image = Image.open(io.BytesIO(image_data.encode("utf-8")))

        # Resize the image to the desired dimensions with LANCZOS filter
        image = image.resize((32, 32), resample=Image.LANCZOS)

        return image

    def predict_digit(self, image):
        # Convert the PIL Image to grayscale
        image = image.convert("L")

        # Replicate the single-channel image to create a 3-channel image
        image = Image.merge("RGB", [image]*3)

        # Preprocess the image
        image = F.to_tensor(image)
        image = F.normalize(image, [0.5], [0.5])

        # Add batch dimension
        image = image.unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = model(image)

        # Get predicted digit
        _, predicted_class = output.max(1)
        return predicted_class.item()


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()
