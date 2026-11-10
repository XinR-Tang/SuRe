export CUDA_VISIBLE_DEVICES=7

ROOT_DIR="Path/to/SuRe"

EXAMPLE="${EXAMPLE:-single-example1}"
EXAMPLE_DIR="${ROOT_DIR}/example/${EXAMPLE}"

MODEL="${MODEL:-sd1_5}"

OUTPUT_PATH="${OUTPUT_PATH:-${EXAMPLE_DIR}/${MODEL}.png}"

cd "$ROOT_DIR"
python -m sure.main \
    --model "${MODEL}" \
    --style_image "${EXAMPLE_DIR}/sty.png" \
    --content_image "${EXAMPLE_DIR}/cnt.png" \
    --output_path "${OUTPUT_PATH}" \
    "$@"
