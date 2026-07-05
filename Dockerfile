FROM python:3.12.4-bullseye

RUN apt update -y 
RUN apt-get install -y python-dev-is-python3 graphviz libgraphviz-dev pkg-config

RUN pip install --upgrade pip

RUN pip install cslearn==1.3.1

# To install from local source instead:
#WORKDIR /cstrees
#COPY . .
#RUN pip install -e ".[expt]"


# For vscode, also install packages Python and Python Extension Pack.