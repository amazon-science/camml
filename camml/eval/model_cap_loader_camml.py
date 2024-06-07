import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from camml.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from camml.conversation import conv_templates, SeparatorStyle
from camml.model.builder import load_pretrained_model, load_pretrained_model_old, load_pretrained_model_665k
from camml.utils import disable_torch_init
from camml.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from camml.model.imagebind_models import imagebind_model
from camml.model.imagebind_models.imagebind_model import ModalityType
import camml.model.imagebind_models.data as imagebind_data

from PIL import Image
import math
import faiss


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]

        qs = line["text"]

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = "This is the useful context information for this question: <image>, which contains " \
                 "previous question answering examples, please use this context information to answer" \
                 "the following question. The image for this question: " + DEFAULT_IMAGE_TOKEN + '\n' + \
                 "The question is: " + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()

    if args.memory == "665k":
        index_path = "data/llava_665k_vision_flatIP.index"
        sqa_llava_json_file = "data/llava_665k_memory_metadata.json"
    else:
        raise ValueError()

    vision_index = faiss.read_index(index_path)

    with open(sqa_llava_json_file, "r") as json_file:
        retriever_metadata = json.load(json_file)
    enc_model = imagebind_model.imagebind_huge(pretrained=True)  # .to(torch.float16)
    enc_model = enc_model.to("cuda")
    enc_model.eval()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model_665k(model_path, args.model_base, model_name)
    model.llava_tokenizer = tokenizer
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    out_dicts = []
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["image_id"]
        cur_prompt = line["text"]

        qs = line["text"]

        qs_query = qs.replace("<image>", "").strip()

        text_list = [qs_query]

        inputs = {
            ModalityType.TEXT: imagebind_data.load_and_transform_text(text_list, device="cuda")
        }
        with torch.no_grad():
            embeddings = enc_model(inputs)

        text_embedding = embeddings[ModalityType.TEXT]
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().detach().numpy().astype('float32')

        retrieved_num = 4
        D, I = vision_index.search(text_embedding, retrieved_num)

        text_ret_list = []
        image_ret_path_list = []

        select_num = args.icl_num
        for numi in range(select_num):
            text_ret = retriever_metadata[int(I[0][numi])]["text"]
            image_ret_path = retriever_metadata[int(I[0][numi])]["image"]
            text_ret_list.append(text_ret)
            image_ret_path_list.append(image_ret_path)

        context = []
        for ti, text_ret in enumerate(text_ret_list):
            context.append(text_ret)

        image_ret_list = []
        for image_ret_path in image_ret_path_list:
            image_ret = Image.open(image_ret_path).convert("RGB")
            image_ret_tensor = process_images([image_ret], image_processor, model.config)[0]
            image_ret_list.append(image_ret_tensor.unsqueeze(0).half().cuda())

        images_ret = torch.cat(image_ret_list)

        context = [c[0] for c in context]

        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                images_ret=[images_ret],
                context=[context],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        out_dict = {"image_id": idx, "caption": outputs}

        ans_file.write(json.dumps(out_dict) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--icl_num", type=int, default=3)
    parser.add_argument("--memory", type=str, default="665k")
    args = parser.parse_args()

    eval_model(args)