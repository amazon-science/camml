import argparse
import torch

from camml.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from camml.conversation import conv_templates, SeparatorStyle
from camml.model.builder import load_pretrained_model, load_pretrained_model_665k
from camml.utils import disable_torch_init
from camml.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import os
from PIL import Image
import json
import requests
from PIL import Image
from io import BytesIO
import faiss
from camml.model.imagebind_models import imagebind_model
from camml.model.imagebind_models.imagebind_model import ModalityType
import camml.model.imagebind_models.data as imagebind_data

# retriever = faiss.read_index("../imagebind/index/sqa_vision.index") # sqa_vision
# vision_index = faiss.read_index("../imagebind/index/sqa_vision_flatIP.index")
# text_index = faiss.read_index("../imagebind/index/sqa_text_flatIP.index")





def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    index_path = "data/llava/llava_665k/index/llava_665k_vision_flatIP.index"
    retriever_json_file = "data/llava/llava_665k/llava_665k_memory_metadata.json"

    retriever = faiss.read_index(index_path)

    # sqa_llava_json_file = "../imagebind/sqa_train_post_memory_answer.json"
    with open(retriever_json_file, "r") as json_file:
        retriever_metadata = json.load(json_file)
    enc_model = imagebind_model.imagebind_huge(pretrained=True)  # .to(torch.float16)
    enc_model = enc_model.to("cuda")
    enc_model.eval()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model_665k(args.model_path, args.model_base, model_name)
    model.llava_tokenizer = tokenizer
    qs = args.query

    qs_query = qs
    text_list = [qs_query]

    image_path = [args.image_file, ]
    if args.image_file is not None:
        inputs = {
            ModalityType.TEXT: imagebind_data.load_and_transform_text(text_list, device="cuda"),
            ModalityType.VISION: imagebind_data.load_and_transform_vision_data(image_path, device="cuda"),

        }
    else:
        inputs = {
            ModalityType.TEXT: imagebind_data.load_and_transform_text(text_list, device="cuda"),
        }
    with torch.no_grad():
        embeddings = enc_model(inputs)

    text_embedding = embeddings[ModalityType.TEXT]
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding.cpu().detach().numpy().astype('float32')

    if args.image_file is not None:
        vision_embedding = embeddings[ModalityType.VISION]
        vision_embedding /= vision_embedding.norm(dim=-1, keepdim=True)
        vision_embedding = vision_embedding.cpu().detach().numpy().astype('float32')

    retrieved_num = 10

    if args.image_file is not None:
        D, I = retriever.search(vision_embedding, retrieved_num)
    else:
        D, I = retriever.search(text_embedding, retrieved_num)

    text_ret_list = []
    image_ret_path_list = []

    select_num = args.shots
    for numi in range(select_num):
        text_ret = retriever_metadata[int(I[0][numi])]["text"]
        image_ret_path = retriever_metadata[int(I[0][numi])]["image"]
        text_ret_list.append(text_ret)
        image_ret_path_list.append(image_ret_path)

    context = []
    for ti, text_ret in enumerate(text_ret_list):
        context.append(text_ret)

    if getattr(model.config, 'mm_use_im_start_end', False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        if args.image_file is not None:
            qs = "This is the useful context information for this question: <image>, which contains " \
                 "previous question answering examples, please use this context information to answer" \
                 "the following question. The image for this question: " + DEFAULT_IMAGE_TOKEN + '\n' + \
                 "The question is: " + qs
        else:
            qs = "This is the useful context information for this question: <image>, which contains " \
                 "previous question answering examples, please use this context information to answer" \
                 "the following question." + '\n' + \
                 "The question is: " + qs

    conv_mode = "vicuna_v1"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    image_ret_list = []
    for image_ret_path in image_ret_path_list:
        image_ret = Image.open(image_ret_path)
        img_ret = image_processor.preprocess(image_ret, return_tensors='pt')['pixel_values'][0]
        image_ret_list.append(img_ret.unsqueeze(0).half().cuda())
    images_ret = torch.cat(image_ret_list)

    image_file = args.image_file
    if image_file is not None:
        image = Image.open(image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        images = image_tensor.unsqueeze(0).half().cuda()
    else:
        images = torch.zeros_like(images_ret[0].unsqueeze(0))

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    temp = 0.2
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            images_ret=[images_ret],
            context=[context],
            do_sample=True if temp > 0 else False,
            temperature=temp,
            max_new_tokens=1024,
            use_cache=False,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="checkpoints/camml-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--shots", type=int, default=3)
    args = parser.parse_args()

    eval_model(args)