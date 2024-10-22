# Import necessary libraries
from gpt4all import GPT4All  # GPT4All model for text generation
import tkinter as tk  # Tkinter library for creating GUI applications
from PIL import ImageTk  # PIL library for image processing and handling in Tkinter
from PIL import Image  # PIL library to open and save images

import torch  # PyTorch library for deep learning models and tensor computations
from diffusers import FluxPipeline  # Diffusers library for stable diffusion-based image generation pipelines
torch.cuda.empty_cache()  # Clear CUDA memory to optimize performance

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler  # Importing necessary schedulers and models

# Function to generate a video (currently commented out for future development)
# def video_gen():
#     # Set up model path for video generation
#     # model_path = '/Saved/rain1011/pyramid-flow-sd3'  # The local directory to save downloaded checkpoint
#     # Load checkpoint for video model
#     # snapshot_download("rain1011/pyramid-flow-sd3", local_dir=model_path, local_dir_use_symlinks=False,
#     #                   repo_type='model')
#
#     # Set the GPU device to be used
#     torch.cuda.set_device(0)
#     # Set data type for the model
#     model_dtype, torch_dtype = 'bf16', torch.bfloat16  # Use bf16 (not support fp16 yet)
#
#     # Load video generation model
#     model = PyramidDiTForVideoGeneration(
#         'PATH',  # The directory where model checkpoints are stored
#         model_dtype,
#         model_variant='diffusion_transformer_768p',   # Model variant for 768p video generation
#     )

#     # Move model components to GPU
#     model.vae.to("cuda")
#     model.dit.to("cuda")
#     model.text_encoder.to("cuda")
#     model.vae.enable_tiling()

# Function to perform image-to-image generation using the Pix2Pix pipeline
def img_2_img():
    # Load the Pix2Pix pipeline model from the specified path
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("Saved/timbrooks/instruct-pix2pix", torch_dtype=torch.float16,
                                                                  safety_checker=None)

    # Move the pipeline to GPU for faster computation
    pipe.to("cuda")

    # Use Euler Ancestral Discrete Scheduler for image generation
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Load the initial image from file
    image = Image.open("image.png")

    # Get the prompt from the Tkinter text entry widget
    prompt = entry.get("1.0",tk.END)
    # Generate images based on the prompt and input image with specific settings
    images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
    # Save the first generated image
    images[0].save("image2.png")

    # Display the generated image in the Tkinter window
    img2 = ImageTk.PhotoImage(Image.open("image2.png"))
    label_img.configure(image=img2)
    label_img.image = img2

# Function to generate images from scratch using the FluxPipeline model
def image_generation():
    # Load the FluxPipeline model from the specified path
    pipe = FluxPipeline.from_pretrained("Saved/black-forest-labs/FLUX.1-schnell",
                                        torch_dtype=torch.bfloat16)
    # to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
    # Enable optimizations for low VRAM GPUs
    pipe.enable_sequential_cpu_offload()  # Offload computation to CPU sequentially
    pipe.vae.enable_slicing()  # Enable slicing to save memory during VAE operations
    pipe.vae.enable_tiling()  # Enable tiling for memory efficiency


    # Option to load LoRA (Low-Rank Adaptation) weights for enhanced image generation (commented out for now)
    # pipe.load_lora_weights("Saved/Shakker-Labs/FLUX.1-dev-LoRA-blended-realistic-illustration",
    #                        weight_name="FLUX-dev-lora-add_details.safetensors")
    # pipe.fuse_lora(lora_scale=1.0)
    # Move the pipeline to GPU
    # pipe.to("cuda")

    pipe.to(
        torch.float16)  # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once

    # Get the prompt from the Tkinter text entry widget
    prompt = entry.get('1.0',tk.END)
    # Generate the image using the specified settings
    out = pipe(
        prompt=prompt,
        guidance_scale=0.,
        height=768,
        width=1360,
        num_inference_steps=4,
        max_sequence_length=256,
    ).images[0]
    # Save the generated image
    out.save("image.png")

    # Load the generated image and display it in the Tkinter window
    img2 = ImageTk.PhotoImage(Image.open("image.png")) # Keep a reference to the image

    label_img.configure(image=img2)
    label_img.image = img2 # Keep a reference to the image


def text_generation():
    # Load the GPT4All model for text generation
    model_name="Llama-3.2-3B-Instruct-Q8_0.gguf"
    model=GPT4All(model_name,model_path="C:/Users/david/PycharmProjects/LocalGPT/Text Generation",allow_download=False)

    # Get the prompt from the Tkinter text entry widget
    prompt = entry.get('1.0',tk.END)

    try:
        # Start a new chat session with the model
        with model.chat_session():
            ai_gen=model.generate(prompt, max_tokens=2048)
            print(f"Reply: {ai_gen}")  # Output the generated text
    except Exception as e:
        # TODO: Add functionality to handle very long prompts
        pass

# Create the main Tkinter window
window = tk.Tk()
window.title("Local AI Wrapper") # Set the title of the window

# Add a heading label to the window
heading=tk.Label(window, text="Chat:")
heading.grid(column=0, row=1, columnspan=2)

# Create a text entry widget for user input
entry=tk.Text(window,height=10,width=60)
entry.grid(column=0, row=2, columnspan=2,padx=10, pady=10)

# Create buttons for text and image generation
button_text=tk.Button(window, text="Generate Text", command=text_generation)
button_img=tk.Button(window, text="Generate Image", command=image_generation)

# Position the buttons in the window
button_text.grid(column=0, row=3,padx=5, pady=10)
button_img.grid(column=1, row=3,padx=5, pady=10)

# Load and display the initial image in the window
img=tk.PhotoImage(file="image.png")
label_img=tk.Label(window, image=img)
label_img.grid(column=0, row=0,columnspan=2)

# Create a button to regenerate an image based on the prompt and an input image
button_img_2_img=tk.Button(window, text="Regenerate Image", command=img_2_img)
button_img_2_img.grid(column=1, row=4)

# (Optional) button for future video generation feature (currently commented out)
# button_video=tk.Button(window, text="Generate Video", command=video_gen)
# button_video.grid(column=0, row=4,padx=5, pady=10)

# Start the Tkinter event loop
tk.mainloop()








