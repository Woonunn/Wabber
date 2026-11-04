# Dockerfile - AgentCore MCP 服务器容器镜像
#    平台必须为 linux/arm64（AgentCore 运行环境为 ARM64）
#    final ARM64 stage avoids RUN instructions so x86 hosts do not need QEMU.
ARG VERSION=3.11

FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:${VERSION}-slim AS installer

ARG VERSION=3.11
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG ATSUITE_SYSTEM_PACKAGES=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

WORKDIR /build
COPY requirements.txt .

# pip evaluates dependency markers on the x86 host even with --platform; uv uses the ARM target.
RUN pip install -q uv==0.13.6 && uv pip install \
    --python-platform aarch64-manylinux_2_28 \
    --python-version ${VERSION} \
    --only-binary=:all: \
    --index-url ${PIP_INDEX_URL} \
    --index-strategy unsafe-best-match \
    --target /pkg \
    -r requirements.txt

# Download ARM packages on the build host; the final stage still needs no QEMU.
RUN mkdir /system && if [ -n "${ATSUITE_SYSTEM_PACKAGES}" ]; then \
    dpkg --add-architecture arm64 && apt-get update && \
    apt-get -o Dir::State::status=/dev/null -o APT::Architecture=arm64 --download-only --no-install-recommends -y install ${ATSUITE_SYSTEM_PACKAGES} && \
    for package in /var/cache/apt/archives/*.deb; do dpkg-deb -x "$package" /system; done; \
    fi

FROM --platform=linux/arm64 public.ecr.aws/docker/library/python:${VERSION}-slim

ARG VERSION=3.11
ARG ATSUITE_NODE_MODULE
ARG ATSUITE_MCP_NAME
ARG PROVIDER
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ENV PROVIDER=${PROVIDER}
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}
ENV PROJECT="uaibs-879512791065"

ENV PYTHON_VERSION=${VERSION}

# Keep the final ARM64 stage free of RUN instructions so hosts without stable
# ARM64 binfmt/QEMU can still cross-build AgentCore images. Python dependencies
# are prepared in the amd64 installer stage and copied below. Benchmarks that
# require OS packages such as git should use an AgentCore base image that already
# contains them, or enable a separate native ARM64/system-package build path.

WORKDIR /app
COPY --from=installer /pkg /app/deps
COPY --from=installer /system /

COPY . /app

ENV PYTHONPATH=/app/deps:/app
ENV ATSUITE_NODE_MODULE=${ATSUITE_NODE_MODULE}
ENV ATSUITE_MCP_NAME=${ATSUITE_MCP_NAME}

ENV ATSUITE_RUNTIME=mcp
ENV ATSUITE_MCP_PORT=8000
ENV ATSUITE_MCP_PATH=/mcp/
ENV ATSUITE_MCP_TRANSPORT=http
ENV ATSUITE_MCP_STATELESS=true

EXPOSE 8000

CMD ["python", "/app/atsuite_sdk/mcp_server.py"]
