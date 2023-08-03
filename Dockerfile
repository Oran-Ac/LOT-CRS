FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
RUN apt update
RUN apt-get install -y tmux
RUN apt-get install -y wget
# RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
# RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
# RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.0+cu1161.html
# RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
# RUN pip install torch-geometric
RUN pip install accelerate
RUN pip install transformers==4.21.3
RUN pip install pandas
RUN pip install datasets
RUN pip install wandb
RUN pip install simcse
RUN pip install faiss-gpu