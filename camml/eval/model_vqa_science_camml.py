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
from camml.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import faiss
from camml.model.imagebind_models import imagebind_model
from camml.model.imagebind_models.imagebind_model import ModalityType
import camml.model.imagebind_models.data as imagebind_data


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


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
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        qs_query = qs
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


        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()

            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = "This is the useful context information for this question: <image>, which contains " \
                      "previous question answering examples, please use this context information to answer" \
                      "the following question. The image for this question: " + DEFAULT_IMAGE_TOKEN + '\n' + \
                      "The question is: " + qs
            # if getattr(model.config, 'mm_use_im_start_end', False):
            #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            # else:
            #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
            image_ret_list = []
            for image_ret_path in image_ret_path_list:
                img_ret = image_processor.preprocess(image_ret, return_tensors='pt')['pixel_values'][0]
                image_ret_list.append(img_ret.unsqueeze(0).half().cuda())
            images_ret = torch.cat(image_ret_list)

        else:
            # images = None
            qs = "This is the useful context information for this question: <image>, which contains " \
                 "previous question answering examples, please use this context information to answer" \
                 "the following question. There is no image for this question. The question is: " + qs
            image_ret_list = []
            for image_ret_path in image_ret_path_list:
                image_ret = Image.open(image_ret_path.replace("/home/ubuntu/fsx/dataset/", "./data/"))
                img_ret = image_processor.preprocess(image_ret, return_tensors='pt')['pixel_values'][0]
                image_ret_list.append(img_ret.unsqueeze(0).half().cuda())
            images_ret = torch.cat(image_ret_list)

            images = torch.zeros_like(images_ret[0].unsqueeze(0))

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                images_ret=[images_ret],
                context=[context],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # prompt for answer
        if args.answer_prompter:
            outputs_reasoning = outputs
            input_ids = tokenizer_image_token(prompt + outputs_reasoning + ' ###\nANSWER:', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=64,
                    use_cache=True,
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
            outputs = outputs_reasoning + '\n The answer is ' + outputs

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--icl_num", type=int, default=3)
    # parser.add_argument("--index_path", type=str, default="../imagebind/llava_665k/index/llava_665k_vision_flatIP.index")
    # parser.add_argument("--meta_path", type=str,
    #                     default="../imagebind/llava_665k/index/llava_665k_vision_flatIP.index")
    parser.add_argument("--memory", type=str, default="665k")
    args = parser.parse_args()

    eval_model(args)
