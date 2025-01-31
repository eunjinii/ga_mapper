# MODELS=("test_layer_BERT_qk" "test_layer_BERT_ffn" "test_layer_VGG16_conv1" "test_model_MobileNetv3" "test_model_MobileViT" "test_model_YOLOv8" "test_model_VGG16" "test_model_InceptionV3")
MODELS=("test_model_VGG16")

for LAYER_INDEX in {0..5}; do
    for MODEL in "${MODELS[@]}"; do
        for i in {0..9}
        do
            echo "Running iteration $i..."
            python run_ga_magneto.py --generation 100 --population 50 --model_name $MODEL --hwconfig mobile --power_budget_mw 1000 --layer $LAYER_INDEX
        done

        # move the results csv to out/magneto/
        mkdir -p out/magneto/$MODEL/layer${LAYER_INDEX}
        mv magneto_${MODEL}.csv out/magneto/$MODEL/layer${LAYER_INDEX}

        # move the fitness csv fitness_${MODEL}_*.csv to out/magneto/$MODEL/
        mkdir -p out/magneto/$MODEL/layer${LAYER_INDEX}
        mv fitness_${MODEL}_*.csv out/magneto/$MODEL/layer${LAYER_INDEX}

        # delete the log.txt
        if [ -f log.txt ]; then
            rm log.txt
        fi
    done
done