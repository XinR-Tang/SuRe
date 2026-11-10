export CUDA_VISIBLE_DEVICES=0

ROOT_DIR="Path/to/SuRe"

EXAMPLE="${EXAMPLE:-multi-example}"
EXAMPLE_DIR="${ROOT_DIR}/example/${EXAMPLE}"

MODEL="${MODEL:-sd1_5}"

cd "$ROOT_DIR"
for i in 1 3 5; do
    STYLE_DIR="${EXAMPLE_DIR}/sty-${i}"
    python -m sure.main \
        --model "${MODEL}" \
        --style_image_folder "${STYLE_DIR}" \
        --content_image "${EXAMPLE_DIR}/cnt.jpg" \
        --output_path "${EXAMPLE_DIR}/${MODEL}_sty${i}.png" \
        "$@"
done
