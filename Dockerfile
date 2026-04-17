FROM ubuntu:latest

ADD watermark.py /smd/watermark.py

CMD ["python3", "/smd/watermark.py"]
