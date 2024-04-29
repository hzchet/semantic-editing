FROM registry.cerebra.kz/ml/cerebra

RUN apt update && apt install -y sudo
RUN apt-get install -y git
RUN apt-get install -y wget

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install black
