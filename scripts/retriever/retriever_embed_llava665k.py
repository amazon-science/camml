import data
import torch
from camml.model.imagebind_models import imagebind_model
from camml.model.imagebind_models.imagebind_model import ModalityType
import os
import json
from tqdm import tqdm
import numpy as np

source_json_file = "data/llava/llava_665k/llava_v1_5_mix665k.json"
with open(source_json_file, "r") as json_file:
    conversations = json.load(json_file)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

vision_embeddings = []
text_embeddings = []

jsons_post = []

# import pdb;pdb.set_trace()
for conv in tqdm(conversations):
    id = conv["id"]
    conversation = conv["conversations"]
    rounds = len(conversation) // 2
    context_string = []
    for r_i in range(rounds):
        assert conversation[r_i*2]["from"] == "human"
        question = conversation[r_i*2]["value"].replace("<image>", "").strip()
        assert conversation[r_i*2+1]["from"] == "gpt"
        answer = conversation[r_i*2+1]["value"]
        context_string.append(question)
        context_string.append(answer)
    context_string = "".join(context_string)

    if "image" in conv.keys():
        image_path = conv["image"]
    else:
        continue

    text_list = [context_string]

    image_path = os.path.join("data/llava/llava_665k", image_path)

    if os.path.exists(image_path):
        real_image_path = image_path
    elif os.path.exists(image_path.replace(".jpg", ".gif")):
        real_image_path = image_path.replace(".jpg", ".gif")
    elif os.path.exists(image_path.replace(".jpg", ".png")):
        real_image_path = image_path.replace(".jpg", ".png")
    else:
        print(image_path)
        raise ValueError()

    image_paths = [real_image_path]
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    vision_embedding = embeddings[ModalityType.VISION]

    vision_embedding /= vision_embedding.norm(dim=-1, keepdim=True)

    vision_embeddings.append(vision_embedding)
    jsons_post.append(
        {
            "id": id,
            "image": image_paths[0],
            "text": context_string,
        }
    )

vision_embeddings = torch.cat(vision_embeddings).cpu().detach().numpy().astype("float32")

np.save("data/llava/llava_665k/vision_emb/llava_665k_vision.npy", vision_embeddings)

with open("data/llava/llava_665k/llava_665k_memory_metadata.json", "w") as f:
    json.dump(jsons_post, f)



