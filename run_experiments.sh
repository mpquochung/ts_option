#!/bin/bash

# Script để chạy experiments với nhiều config khác nhau
# Sẽ test trên 3 file CSV: CMSN2406, CSTB2410, CVRE2407
# Với các model type: lstm, gru, rnn
# Và các hyperparameters khác nhau

# Tạo thư mục để lưu kết quả
RESULTS_DIR="experiment_results"
mkdir -p "$RESULTS_DIR"

# File CSV để lưu kết quả
RESULTS_CSV="$RESULTS_DIR/experiment_results_5day_$(date +%Y%m%d_%H%M%S).csv"

# Tạo header cho CSV
echo "experiment_id,csv_file,model_type,hidden_size,num_layers,batch_size,learning_rate,random_noise,epochs,mse,mae,rmse,mape,train_months,test_month" > "$RESULTS_CSV"

# Danh sách các file CSV để test
CSV_FILES=(
    "data/CVRE2407.csv"
    "data/CMSN2406.csv"
    "data/CSTB2410.csv"
)

# Danh sách các model type
MODEL_TYPES=("lstm" "gru" "rnn")

# Danh sách các hidden_size để test
HIDDEN_SIZES=(32 48 64)

# Danh sách các num_layers để test
NUM_LAYERS=(1 2)

# Danh sách các batch_size để test
BATCH_SIZES=(8 16 32)

# Danh sách các learning_rate để test
LEARNING_RATES=(0.001 0.0002)

# Số epochs (có thể giảm để test nhanh hơn)
NUM_EPOCHS=10

# Window size

RANDOM_NOISE=(true false)

# Counter cho experiment ID
EXPERIMENT_ID=1

# Tổng số experiments
TOTAL_EXPERIMENTS=$((${#CSV_FILES[@]} * ${#MODEL_TYPES[@]} * ${#HIDDEN_SIZES[@]} * ${#NUM_LAYERS[@]} * ${#BATCH_SIZES[@]} * ${#LEARNING_RATES[@]} * ${#RANDOM_NOISE[@]}))

echo "=========================================="
echo "Bắt đầu chạy experiments"
echo "Tổng số experiments: $TOTAL_EXPERIMENTS"
echo "Kết quả sẽ được lưu vào: $RESULTS_CSV"
echo "=========================================="
echo ""

# Lặp qua tất cả các combinations
for csv_file in "${CSV_FILES[@]}"; do
    # Lấy tên file CSV (không có path và extension)
    csv_name=$(basename "$csv_file" .csv)
    
    for model_type in "${MODEL_TYPES[@]}"; do
        for hidden_size in "${HIDDEN_SIZES[@]}"; do
            for num_layers in "${NUM_LAYERS[@]}"; do
                for batch_size in "${BATCH_SIZES[@]}"; do
                    for lr in "${LEARNING_RATES[@]}"; do
                        for random_noise in "${RANDOM_NOISE[@]}"; do
                            # Điều chỉnh số epochs theo learning rate
                            if (( $(echo "$lr == 0.0002" | bc -l) )); then
                                CURRENT_EPOCHS=15
                            else
                                CURRENT_EPOCHS=$NUM_EPOCHS
                            fi
                            
                            echo "----------------------------------------"
                            echo "Experiment $EXPERIMENT_ID / $TOTAL_EXPERIMENTS"
                            echo "CSV: $csv_name"
                            echo "Model: $model_type"
                            echo "Hidden Size: $hidden_size"
                            echo "Num Layers: $num_layers"
                            echo "Batch Size: $batch_size"
                            echo "Learning Rate: $lr"
                            echo "Random Noise: $random_noise"
                            echo "----------------------------------------"
                            
                            # Tạo config file tạm thời
                            TEMP_CONFIG="$RESULTS_DIR/temp_config_${EXPERIMENT_ID}.yaml"
                            
                            # Copy config mẫu và thay đổi các tham số
                            cat configs/5_day.yaml > "$TEMP_CONFIG"
                            
                            sed -i "s|csv_path:.*|csv_path: \"$csv_file\"|" "$TEMP_CONFIG"

                            sed -i "s|type:.*# \"lstm\" hoặc \"rnn\"|type: \"$model_type\"|" "$TEMP_CONFIG"
                            sed -i "s|hidden_size:.*|hidden_size: $hidden_size|" "$TEMP_CONFIG"
                            sed -i "s|num_layers:.*|num_layers: $num_layers|" "$TEMP_CONFIG"

                            sed -i "s|batch_size:.*|batch_size: $batch_size|" "$TEMP_CONFIG"
                            sed -i "s|num_epochs:.*|num_epochs: $CURRENT_EPOCHS|" "$TEMP_CONFIG"
                            sed -i "s|learning_rate:.*|learning_rate: $lr|" "$TEMP_CONFIG"

                            sed -i "s|experiment_name:.*|experiment_name: \"exp_${EXPERIMENT_ID}_${csv_name}_${model_type}\"|" "$TEMP_CONFIG"
                            sed -i "s|filename:.*|filename: \"model_exp${EXPERIMENT_ID}.pt\"|" "$TEMP_CONFIG"

                            sed -i "s|random_noise:.*|random_noise: $random_noise|" "$TEMP_CONFIG"

                            
                            # Chạy training và capture output
                            OUTPUT=$(python3 main.py --config "$TEMP_CONFIG" 2>&1)
                            STATUS=$?
                            
                            # Kiểm tra exit code
                            if [ $STATUS -eq 0 ]; then
                                echo "✓ Training thành công"
                                
                                # Parse kết quả từ output
                                MSE=$(echo "$OUTPUT" | grep "MSE:" | tail -1 | awk '{print $2}')
                                MAE=$(echo "$OUTPUT" | grep "MAE:" | tail -1 | awk '{print $2}')
                                RMSE=$(echo "$OUTPUT" | grep "RMSE:" | tail -1 | awk '{print $2}')
                                MAPE=$(echo "$OUTPUT" | grep "MAPE:" | tail -1 | awk '{print $2}')
                                TRAIN_MONTHS_LINE=$(echo "$OUTPUT" | grep "Train months:")
                                TEST_MONTH_LINE=$(echo "$OUTPUT" | grep "Test month:")
                                
                                TRAIN_MONTHS=$(echo "$TRAIN_MONTHS_LINE" | sed 's/Train months: *//' \
                                    | tr -d "[]'," \
                                    | xargs \
                                    | tr ' ' ';')

                                TEST_MONTH=$(echo "$TEST_MONTH_LINE" | sed 's/Test month: *//' | xargs)

                                
                                # Nếu không parse được thì đặt giá trị mặc định
                                MSE=${MSE:-"N/A"}
                                MAE=${MAE:-"N/A"}
                                RMSE=${RMSE:-"N/A"}
                                MAPE=${MAPE:-"N/A"}
                                TRAIN_MONTHS=${TRAIN_MONTHS:-"N/A"}
                                TEST_MONTH=${TEST_MONTH:-"N/A"}
                                
                                # Ghi kết quả vào CSV
                                echo "$EXPERIMENT_ID,$csv_name,$model_type,$hidden_size,$num_layers,$batch_size,$lr,$random_noise,$CURRENT_EPOCHS,$MSE,$MAE,$RMSE,$MAPE,$TRAIN_MONTHS,$TEST_MONTH" >> "$RESULTS_CSV"
                            else
                                echo "✗ Training thất bại (exit code $STATUS)"

                                # Ghi lỗi vào CSV
                                echo "$EXPERIMENT_ID,$csv_name,$model_type,$hidden_size,$num_layers,$batch_size,$lr,$random_noise,$CURRENT_EPOCHS,,,,,," >> "$RESULTS_CSV"
                            fi
                            
                            # Xóa temp config
                            rm -f "$TEMP_CONFIG"
                            
                            # Tăng experiment ID
                            EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
                            
                            echo ""
                        done
                    done
                done
            done
        done
    done
done

echo "=========================================="
echo "=========================================="
echo "Hoàn thành tất cả experiments!"
echo "Kết quả đã được lưu vào: $RESULTS_CSV"
echo "=========================================="
# Hiển thị top 10 best results theo MSE
echo ""
echo "Top 10 experiments với MSE thấp nhất:"
echo "=========================================="
head -1 "$RESULTS_CSV"
tail -n +2 "$RESULTS_CSV" | grep -v "ERROR" | sort -t',' -k9 -n | head -10
echo "=========================================="