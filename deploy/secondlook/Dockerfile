# Secondlook on Frigate 0.18.2: the official image plus this fork's Python
# package and web UI. Build from the repository root:
#
#   docker build -f deploy/secondlook/Dockerfile \
#     --build-arg SECONDLOOK_REVISION=$(git rev-parse --short HEAD) \
#     -t frigate-secondlook:0.18.2 .
#
# Only frigate/ and the web UI are replaced; native libraries, s6, nginx,
# FFmpeg and Python dependencies are the stock 0.18.2 ones. The base is pinned
# to the official 0.18.2 multi-platform manifest digest.

FROM node:20 AS web-build
WORKDIR /work
COPY web/package.json web/package-lock.json ./
RUN npm install
COPY web/ ./
# Same steps as the stock web build (docker/main/Dockerfile).
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN npm run build \
    && mv dist/BASE_PATH/monacoeditorwork/* dist/assets/ \
    && rm -rf dist/BASE_PATH

FROM ghcr.io/blakeblackshear/frigate:0.18.2@sha256:fd60f99ffd4eb6c42360ed4e14953e270586f239309ab3983e55b2e97ffac09a
ARG SECONDLOOK_REVISION=unknown
LABEL io.secondlook.source-revision=$SECONDLOOK_REVISION \
      io.secondlook.upstream-base="v0.18.2-cfa08c56b23d23280c3d0a218e5a43182c97445b"
RUN rm -rf /opt/frigate/frigate /opt/frigate/web
COPY frigate/ /opt/frigate/frigate/
COPY --from=web-build /work/dist/ /opt/frigate/web/
COPY deploy/secondlook/preflight.py /opt/frigate/secondlook_preflight.py
RUN echo "VERSION = \"0.18.2-secondlook-${SECONDLOOK_REVISION}\"" > /opt/frigate/frigate/version.py \
    && python3 -m compileall -q /opt/frigate/frigate/secondlook
