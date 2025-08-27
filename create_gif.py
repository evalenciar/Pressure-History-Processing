from PIL import Image
import glob
import os
from tkinter import Tk, filedialog, simpledialog

def create_gif(input_folder, output_file, duration=500):
    """
    Create a GIF from PNG images in the specified folder.
    
    Args:
        input_folder (str): Path to folder containing PNG images
        output_file (str): Path for output GIF file
        duration (int): Duration for each frame in milliseconds
    """
    # Get all PNG files in the folder
    images = []
    for file in sorted(glob.glob(os.path.join(input_folder, "*.png"))):
        img = Image.open(file)
        # Convert to RGB if necessary (handles PNGs with transparency)
        if img.mode in ('RGBA', 'LA'):
            img = img.convert('RGB')
        images.append(img)
    
    if not images:
        print("No PNG files found in the specified folder!")
        return
    
    # Save the GIF
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # 0 means loop forever
    )
    print(f"GIF created successfully: {output_file}")

if __name__ == "__main__":
    # Prompt the user to select the input folder and output file
    root = Tk()
    root.withdraw()
    input_folder = filedialog.askdirectory(title="Select Folder Containing PNG Images")
    if not input_folder:
        print("No folder selected. Exiting.")
        exit()
    else:
        print(f"Selected folder: {input_folder}")
    output_file = filedialog.asksaveasfilename(
        title="Save GIF As",
        defaultextension=".gif",
        filetypes=[("GIF files", "*.gif")]
    )
    create_gif(input_folder, output_file)