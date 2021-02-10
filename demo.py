import torch
import clip
import sys
from PIL import Image
from time import perf_counter 
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

assert len(sys.argv) > 1

def _infer(engr_prompts=["a diagram", "a dog", "a cat"], filename=None):
    if type(filename) == type(None):
        image = preprocess(Image.open(sys.argv[1])).unsqueeze(0).to(device)
    else:
        image = preprocess(Image.open(filename)).unsqueeze(0).to(device)

    start = perf_counter() 
    text = clip.tokenize(engr_prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    print("Elapsed time:", str(perf_counter() - start) + " seconds") 
    print("Label probs:", [engr_prompts[i] + ": " + str(int(p*100)) + "%" for i, p in enumerate(probs[0])])
    return probs
    
# build query given ioi is "scramble.jpg"
questions = []
questions.append("Is it organic?")
questions.append("Is it alive?")
questions.append("Is it big?")
questions.append("Is it fiction?")
questions.append("Is it a machine?")
