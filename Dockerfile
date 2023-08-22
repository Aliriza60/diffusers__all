FROM nvidia/cuda:12.1.0-base-ubuntu20.04
RUN apt update
RUN apt install -y git python3 python3-pip wget
WORKDIR /diffusers_all
COPY text_to_image.py /diffusers_all/
COPY img_to_img.py /diffusers_all/
COPY inpainting.py /diffusers_all/
COPY depth_to_image.py /diffusers_all/
COPY super_resolution.py /diffusers_all/
COPY super_resolution_1.py /diffusers_all/
COPY requirements.txt /diffusers_all/
RUN pip install scipy
RUN pip install super-image
RUN pip install -r requirements.txt
CMD ["python","text_to_image.py","img_to_img.py","inpainting.py","depth_to_image.py","super_resolution.py","super_resolution_1.py"]
