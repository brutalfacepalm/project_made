import gensim
import compress_fasttext

class FastTextEmbedding:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, words):
        embedding = [
            self.model.get_vector(word)
            for word in words
        ]
        return embedding


class RusVectoresEmb(FastTextEmbedding):
    def __init__(self, model_path):
        self.model = gensim.models.fasttext.FastTextKeyedVectors.load(model_path)


class CompressFastTextEmb(FastTextEmbedding):
    def __init__(self, 
                 model_path='/tmp/compress_fast_text.model',
                 load_url='https://github.com/avidale/compress-fasttext/releases/download/v0.0.4/08_11_3_6_model_400K_100K_pq300.bin'):
        try:
            self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(model_path)
        except:
            self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(load_url)
            self.model.save(model_path)

    
