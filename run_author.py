import tensorflow as tf
import collections
import os
import os.path
import sys
import re
import time
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.contrib import seq2seq
import AuthorConfig
import author_model
import read_author




def main():
    conf = AuthorConfig.AuthorConfig
    #read data in 'file_path' from config
    b_reader = read_author.BooksReader(conf.file_path)
    authMod = author_model.AuthorModel(conf,b_reader.vocabularySize)
    authMod.close_sess()
    return





if __name__ == "__main__":
    main()
