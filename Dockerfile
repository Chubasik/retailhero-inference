FROM tensorflow/serving


RUN echo "deb mirror://mirrors.ubuntu.com/mirrors.txt bionic main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb mirror://mirrors.ubuntu.com/mirrors.txt bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb mirror://mirrors.ubuntu.com/mirrors.txt bionic-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install build-essential ca-certificates libopenblas-dev gfortran libhdf5-dev python3-dev \
    --no-install-recommends  python3=3.6.7-1~18.04 python3-pip libatlas3-base python3-setuptools gunicorn -y && \
    mkdir /workdir/ && \
    export PATH=$PATH:~/.local/bin/

COPY requirements.txt /workdir/

RUN pip3 install -U pip

EXPOSE 8888

WORKDIR /workdir/

RUN pip3 install -r requirements.txt

VOLUME [ "/workdir" ]

COPY ./models /models

COPY ./server.py /workdir/server.py
COPY ./helpers.py /workdir/helpers.py
COPY ./vocab.txt /workdir/vocab.txt

EXPOSE 8000

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV FLASK_APP=/workdir/server.py

EXPOSE 8000

COPY ./run_docker.sh /bin/run_docker.sh
RUN chmod +x /bin/run_docker.sh

ENTRYPOINT ["/bin/run_docker.sh" ]