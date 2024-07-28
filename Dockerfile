FROM python:3.12.4-bullseye

RUN apt update -y 
RUN apt-get install -y python-dev-is-python3 graphviz libgraphviz-dev pkg-config

RUN pip install --upgrade pip

WORKDIR /cstrees

COPY . .

RUN pip install -r requirements.txt
RUN pip install -e .

# For vscode, also install packages Python and Python Extension Pack.