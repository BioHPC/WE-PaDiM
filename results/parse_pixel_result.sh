for dir in Phase1_ReducedParamSearch_*; do
    model=$(echo "$dir" | sed -E 's/Phase1_ReducedParamSearch_([a-z0-9\-]+)_.*/\1/')
    python parse_by_pixel.py --results_dir "$dir" --model "$model"
done
