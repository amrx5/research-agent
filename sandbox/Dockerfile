# Sandbox image for the python tool.
#   docker build -t research-agent-sandbox sandbox/
# The agent runs it with: --network none --read-only --memory 256m --cpus 1 --pids-limit 64 --user 65534
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.1.3 sympy==1.13.3 pandas==2.2.3
WORKDIR /tmp
USER 65534:65534
