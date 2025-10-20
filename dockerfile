# Lightweight base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y git wget curl && apt-get clean

# Set working directory
WORKDIR /app

# Copy your dependency list
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy your chatbot app
COPY finetuneChatbot.py /app/

# Expose Gradio port
EXPOSE 7860

# Run your app
CMD ["python", "finetuneChatbot.py"]
