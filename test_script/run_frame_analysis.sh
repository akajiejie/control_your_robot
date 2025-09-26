#!/bin/bash

# 逐帧差异分析工具快速启动脚本
# 运行环境: telerobot

set -e  # 遇到错误时退出

echo "=================================================="
echo "逐帧差异分析工具 - 快速启动脚本"
echo "=================================================="

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "telerobot" ]]; then
    echo "警告: 当前不在 telerobot 环境中"
    echo "请先运行: conda activate telerobot"
    echo ""
    read -p "是否要继续运行? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 默认参数
SOURCE_FOLDER="./save/pick_place_cup/"
MODEL1_FOLDER="./test/reload_model_actions/act_pick_place_cup/"
MODEL2_FOLDER="./test/reload_model_actions/pi0_pick_place_cup/"
OUTPUT_DIR="./save/output/frame_wise_analysis_$(date +%Y%m%d_%H%M%S)/"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--source)
            SOURCE_FOLDER="$2"
            shift 2
            ;;
        -m1|--model1)
            MODEL1_FOLDER="$2"
            shift 2
            ;;
        -m2|--model2)
            MODEL2_FOLDER="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法:"
            echo "  $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -s, --source    源数据文件夹 (默认: $SOURCE_FOLDER)"
            echo "  -m1, --model1   模型1文件夹 (默认: $MODEL1_FOLDER)"
            echo "  -m2, --model2   模型2文件夹 (默认: $MODEL2_FOLDER)"
            echo "  -o, --output    输出目录 (默认: 带时间戳的目录)"
            echo "  -h, --help      显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0  # 使用默认参数"
            echo "  $0 -s ./save/my_data/ -o ./my_output/"
            echo ""
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 显示配置
echo "运行配置:"
echo "  源数据文件夹: $SOURCE_FOLDER"
echo "  模型1文件夹:   $MODEL1_FOLDER"
echo "  模型2文件夹:   $MODEL2_FOLDER"
echo "  输出目录:     $OUTPUT_DIR"
echo ""

# 检查输入文件夹
echo "检查输入文件夹..."
for folder in "$SOURCE_FOLDER" "$MODEL1_FOLDER" "$MODEL2_FOLDER"; do
    if [[ ! -d "$folder" ]]; then
        echo "错误: 文件夹不存在 - $folder"
        exit 1
    else
        file_count=$(find "$folder" -name "*.hdf5" -o -name "*.h5" | wc -l)
        echo "  ✓ $folder ($file_count 个HDF5文件)"
    fi
done

echo ""
read -p "是否继续运行分析? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消运行"
    exit 0
fi

echo ""
echo "开始运行逐帧差异分析..."
echo "=================================================="

# 运行分析
python test_script/frame_wise_difference_analysis.py \
    --source_folder "$SOURCE_FOLDER" \
    --model1_folder "$MODEL1_FOLDER" \
    --model2_folder "$MODEL2_FOLDER" \
    --label_source "Dataset" \
    --label_model1 "ACT" \
    --label_model2 "PI0" \
    --output "$OUTPUT_DIR" \
    --verbose

# 检查结果
echo ""
echo "=================================================="
echo "分析完成！检查输出文件..."

if [[ -d "$OUTPUT_DIR" ]]; then
    echo "输出目录: $OUTPUT_DIR"
    
    # 检查生成的文件
    files=(
        "frame_wise_difference_tables.png"
        "frame_wise_mean_differences.csv"
        "frame_wise_variance_differences.csv"
    )
    
    for file in "${files[@]}"; do
        filepath="$OUTPUT_DIR/$file"
        if [[ -f "$filepath" ]]; then
            size=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null || echo "unknown")
            echo "  ✓ $file ($size bytes)"
        else
            echo "  ✗ $file (未生成)"
        fi
    done
    
    echo ""
    echo "可以使用以下命令查看结果:"
    echo "  # 查看图片"
    echo "  open '$OUTPUT_DIR/frame_wise_difference_tables.png'"
    echo "  # 或者"
    echo "  eog '$OUTPUT_DIR/frame_wise_difference_tables.png'"
    echo ""
    echo "  # 查看CSV数据"
    echo "  cat '$OUTPUT_DIR/frame_wise_mean_differences.csv'"
    echo "  cat '$OUTPUT_DIR/frame_wise_variance_differences.csv'"
    
else
    echo "错误: 输出目录未创建"
    exit 1
fi

echo ""
echo "=================================================="
echo "逐帧差异分析完成！"
echo "=================================================="
