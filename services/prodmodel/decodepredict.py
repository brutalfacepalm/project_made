import pickle
from sklearn.preprocessing import OrdinalEncoder
from utils import load_files_from_s3


class DecoderPredict:

    def __init__(self, path):
        self.path = path
        self.enc = self.__get_encoder()


    def __get_encoder(self):
        with open(self.path + '/dataformodel/ordinalencoder.pkl', 'rb') as enc_file:
            return pickle.load(enc_file) 

    def decode_answer(self, prediction):
        
        return self.enc.inverse_transform([[prediction]])[0][0]

        
def get_decoder(path):
    load_files_from_s3('models/baseline/ordinalencoder.pkl', path, '/dataformodel/ordinalencoder.pkl')

    decoder_predict = DecoderPredict(path)

    return decoder_predict