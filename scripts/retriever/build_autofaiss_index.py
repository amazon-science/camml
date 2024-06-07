from autofaiss import build_index


emb_folder = "data/llava/llava_665k/vision_emb/"
index_folder = "data/llava/llava_665k/index/"
index_name = "llava_665k_vision_flatIP"
max_index_memory_usage = "300G"
current_memory_available = "400G"
nb_cores = None

build_index(embeddings=emb_folder, index_key="Flat",
            index_path=index_folder + "/" + index_name + ".index",
            index_infos_path=index_folder + "/" + index_name + ".json",
            max_index_memory_usage=max_index_memory_usage,
            current_memory_available=current_memory_available)
