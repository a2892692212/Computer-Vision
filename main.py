import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils.blend import multiband_blending, simple_blending
from utils.panorama import generate_panorama


# select algorithm to generate panorama
option_var = None

def main():
    global option_var
    # Create the main UI window
    window = tk.Tk()
    window.title('Panorama Generation')
    window.geometry('300x150')

    # Initialize the option variable and set its default value
    option_var = tk.StringVar(window)
    option_var.set('Option 2')  # Default option

    # Define the options available for panorama blending
    options = ['Option 1', 'Option 2']

    # Create a button to select a video file
    btn_select_file = tk.Button(window, text='Select Video File', command=select_video_file)
    btn_select_file.pack(pady=20)

    # Create an option menu for selecting the blending method
    option_menu = tk.OptionMenu(window, option_var, *options)
    option_menu.pack(pady=10)

    # Start the Tkinter event loop
    window.mainloop()


def select_video_file():
    """
    Open a file dialog to select a video file, generate a panorama from it, and display the panorama in a new Tkinter window.
    """
    # Open file dialog to choose a video file
    video_path = filedialog.askopenfilename()
    if video_path:
        # Get the selected blending option
        option = option_var.get()
        # Generate the panorama using the selected video and option
        panorama = generate_panorama(video_path, option)

        # Convert the panorama to a format compatible with Tkinter
        panorama_img = cv2.cvtColor(panorama.astype(np.uint8), cv2.COLOR_BGR2RGB)
        panorama_img = Image.fromarray(panorama_img)
        panorama_img = ImageTk.PhotoImage(panorama_img)

        # Create a new window to display the panorama
        window = tk.Toplevel()
        window.title('Panorama')

        # Create a canvas widget in the window and add scrollbars
        canvas = tk.Canvas(window, width=panorama_img.width(), height=panorama_img.height())
        scrollbar_x = tk.Scrollbar(window, orient="horizontal", command=canvas.xview)
        scrollbar_y = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)

        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Place the image on the canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=panorama_img)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        # Display the panorama using matplotlib for comparison (optional)
        plt.figure()
        plt.imshow(cv2.cvtColor(panorama.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.show()

        window.mainloop()


if __name__ == '__main__':
    main()
