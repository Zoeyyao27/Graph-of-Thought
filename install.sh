git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install cython==0.29 --upgrade
pip install -e .
pip install stanford-openie
pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

cd ..
pip install -r requirements.txt
