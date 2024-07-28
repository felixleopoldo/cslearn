FROM python:3.12.4-bullseye

RUN apt update -y 
RUN apt-get install -y python-dev-is-python3 graphviz libgraphviz-dev pkg-config

RUN pip install --upgrade pip

RUN pip install cstrees==1.3.0

# This would be the way to install the package from the source code:
#WORKDIR /cstrees
#COPY . .
#RUN pip install -r requirements.txt
#RUN pip install -e .


# For vscode, also install packages Python and Python Extension Pack.