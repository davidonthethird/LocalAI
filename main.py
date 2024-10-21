from gpt4all import GPT4All
import tkinter as tk
import textwrap
from PIL import ImageTk,Image
import time
import torch
from PIL import Image
from diffusers.utils import load_image, export_to_video

import torch
from diffusers import FluxPipeline
torch.cuda.empty_cache()
from huggingface_hub import snapshot_download

import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler




def img_2_img():
    # model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("C:/Users/david/PycharmProjects/LocalGPT/Saved/timbrooks/instruct-pix2pix", torch_dtype=torch.float16,
                                                                  safety_checker=None)
    # StableDiffusionInstructPix2PixPipeline.save_pretrained(pipe,save_directory="C:/Users/david/PycharmProjects/LocalGPT/Saved/timbrooks/instruct-pix2pix")

    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    image = Image.open("image.png")

    prompt = entry.get("1.0",tk.END)
    images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
    images[0].save("image2.png")

    img2 = ImageTk.PhotoImage(Image.open("image2.png"))

    label_img.configure(image=img2)
    label_img.image = img2

def image_generation():
    # can replace schnell with dev
    pipe = FluxPipeline.from_pretrained("C:/Users/david/PycharmProjects/LocalGPT/Saved/black-forest-labs/FLUX.1-schnell",
                                        torch_dtype=torch.bfloat16)
    # to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()



    #lora
    # pipe.load_lora_weights("C:/Users/david/PycharmProjects/LocalGPT/Saved/Shakker-Labs/FLUX.1-dev-LoRA-blended-realistic-illustration",
    #                        weight_name="FLUX-dev-lora-add_details.safetensors")
    # pipe.fuse_lora(lora_scale=1.0)
    # pipe.to("cuda")


    pipe.to(
        torch.float16)  # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once

    prompt = entry.get('1.0',tk.END)
    out = pipe(
        prompt=prompt,
        guidance_scale=0.,
        height=768,
        width=1360,
        num_inference_steps=4,
        max_sequence_length=256,
    ).images[0]
    out.save("image.png")

    time.sleep(2)
    img2 = ImageTk.PhotoImage(Image.open("image.png"))
    # label_img.config(image=img2)
    # canvas.itemconfig(img_container, image=img2)

    label_img.configure(image=img2)
    label_img.image = img2


def text_generation():
    # model_name="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    model_name="Llama-3.2-3B-Instruct-Q8_0.gguf"
    model=GPT4All(model_name,model_path="C:/Users/david/PycharmProjects/LocalGPT/Text Generation",allow_download=False)

    # f=open("aisort.txt")
    # code=f.read()

    # prompt = f"Role: Your task is to summarize data. Guidelines: Please include important information (if such information exists) such as dates, any reasons for denial or delay, and any types of doctors visited. Please exclude general data that defines a denial or delay or names of people or specific laws. Using all the guidelines, please summarize the following and keep your entire reply to one sentence: {ai_notes}"
    prompt = entry.get('1.0',tk.END)
    # prompt=f"Please create a summary of the purpose of the following code as well as skills used in the following code for a resume description please, here is the code: "

    try:
        with model.chat_session():
            ai_gen=model.generate(prompt, max_tokens=2048)
            print(f"Reply: {ai_gen}")
    except Exception as e:
        #TODO: add abbility to use very long prompts
        pass

window = tk.Tk()
# window.geometry("800x600")
window.title("Local AI Wrapper")
heading=tk.Label(window, text="Chat:")
heading.grid(column=0, row=1, columnspan=2)
entry=tk.Text(window,height=10,width=60)
entry.grid(column=0, row=2, columnspan=2,padx=10, pady=10)
button_text=tk.Button(window, text="Generate Text", command=text_generation)
button_img=tk.Button(window, text="Generate Image", command=image_generation)
# canvas = tk.Canvas(window, width = 1360, height = 768)
# canvas.grid(column=0, row=2, columnspan=2)
button_text.grid(column=0, row=3,padx=5, pady=10)
button_img.grid(column=1, row=3,padx=5, pady=10)
# img = ImageTk.PhotoImage(Image.open("image.png"))
# img_container=canvas.create_image(0, 0,anchor="nw", image=img)
img=tk.PhotoImage(file="image.png")
label_img=tk.Label(window, image=img)
label_img.grid(column=0, row=0,columnspan=2)

button_img_2_img=tk.Button(window, text="Regenerate Image", command=img_2_img)
button_img_2_img.grid(column=1, row=4)

# button_video=tk.Button(window, text="Generate Video", command=video_gen)
# button_video.grid(column=0, row=4,padx=5, pady=10)





tk.mainloop()








