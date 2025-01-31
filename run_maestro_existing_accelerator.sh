#!/bin/bash

ACCELERATORS=("nvdla" "shidiannao" "eyeriss")
MODELS=("test_layer_BERT_qk" "test_layer_BERT_ffn" "test_layer_VGG16_conv1" "test_model_MobileNetv3" "test_model_MobileViT" "test_model_YOLOv8" "test_model_VGG16" "test_model_InceptionV3")

for ACCLERATOR in "${ACCELERATORS[@]}"; do

    # Set dataflow based on accelerator
    if [ "$ACCLERATOR" == "nvdla" ]; then
        DATAFLOW="ykp_os"
    elif [ "$ACCLERATOR" == "shidiannao" ]; then
        DATAFLOW="kcp_ws"
    elif [ "$ACCLERATOR" == "eyeriss" ]; then
        DATAFLOW="rs"
    else
        echo "Error: Invalid ACCLERATOR. Choose from 'nvdla', 'shidiannao', or 'eyeriss'."
        exit 1
    fi

    for MODEL in "${MODELS[@]}"; do
        MAPPING_OUTPUT_NAME="${ACCLERATOR}_${MODEL}"
        MAPPING_FILE="${ACCLERATOR}_${MODEL}_${DATAFLOW}.m"
        HW_FILE="data/hw/mobile.m"

        # create mapping file
        cd tools/frontend || exit 1
        python modelfile_to_mapping.py --model_file "$MODEL" --dataflow "$DATAFLOW" --outfile "$MAPPING_OUTPUT_NAME"

        # if nvdla, fix the TemporalMap bug
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # MacOS
            sed -i '' 's/TemporalMap(Sz(R),7) R;/TemporalMap(Sz(R),Sz(R)) R;/g' "../../data/mapping/$MAPPING_FILE"
            sed -i '' 's/TemporalMap(Sz(S),7) S;/TemporalMap(Sz(S),Sz(S)) S;/g' "../../data/mapping/$MAPPING_FILE"
        else
            # Linux
            sed -i 's/TemporalMap(Sz(R),7) R;/TemporalMap(Sz(R),Sz(R)) R;/g' "../../data/mapping/$MAPPING_FILE"
            sed -i 's/TemporalMap(Sz(S),7) S;/TemporalMap(Sz(S),Sz(S)) S;/g' "../../data/mapping/$MAPPING_FILE"
        fi

        # run MAESTRO
        cd ../.. || exit 1
        ./maestro --HW_file="$HW_FILE" \
                --Mapping_file="data/mapping/$MAPPING_FILE" \
                --print_res=false \
                --print_res_csv_file=true \
                --print_log_file=false
    done
done