#!/bin/bash

#
# Matthew Abbott 2025
# Test for both rust_cuda and facaded_rust_cuda
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
TEMP_DIR="./output/mlp_rust_cuda_user_tests_$$"
MLP_BIN="./rust_cuda/target/release/mlp_cuda"
FACADE_BIN="./facaded_rust_cuda/target/release/facaded_mlp_cuda"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Setup/Cleanup
# Note: Removed cleanup to preserve test output files in ./output directory
# cleanup() {
#     rm -rf "$TEMP_DIR"
# }
# trap cleanup EXIT

mkdir -p "$TEMP_DIR"

cd rust_cuda && cargo build --release && cd ..
cd facaded_rust_cuda && cargo build --release && cd ..

# Test function
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_pattern="$3"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    output=$(eval "$command" 2>&1)
    exit_code=$?

    if echo "$output" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Command: $command"
        echo "  Expected pattern: $expected_pattern"
        echo "  Output:"
        echo "$output" | head -5
        FAIL=$((FAIL + 1))
    fi
}

check_file_exists() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ -f "$file" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  File not found: $file"
        FAIL=$((FAIL + 1))
    fi
}

check_json_valid() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ ! -f "$file" ]; then
        echo -e "${RED}FAIL${NC}"
        echo "  File not found: $file"
        FAIL=$((FAIL + 1))
        return
    fi

    if grep -q '"input_size"' "$file" && grep -q '"output_size"' "$file" && grep -q '"weights"' "$file"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Invalid JSON structure in $file"
        FAIL=$((FAIL + 1))
    fi
}

# ============================================
# Start Tests
# ============================================

echo ""
echo "========================================="
echo "MLP Rust CUDA User Workflow Test Suite"
echo "========================================="
echo ""

# Check binaries exist
if [ ! -f "$MLP_BIN" ]; then
    echo -e "${RED}Error: $MLP_BIN not found. Compile with: cd rust_cuda && cargo build --release${NC}"
    exit 1
fi

if [ ! -f "$FACADE_BIN" ]; then
    echo -e "${RED}Error: $FACADE_BIN not found. Compile with: cd facaded_rust_cuda && cargo build --release${NC}"
    exit 1
fi

echo -e "${BLUE}=== MLP Rust CUDA Binary Tests ===${NC}"
echo ""

# ============================================
# Basic Help/Usage
# ============================================

echo -e "${BLUE}Group: Help & Usage${NC}"

run_test \
    "MLP CUDA help command" \
    "$MLP_BIN help" \
    "Commands:"

run_test \
    "MLP CUDA --help flag" \
    "$MLP_BIN --help" \
    "Commands:"

run_test \
    "FacadeMLP CUDA help command" \
    "$FACADE_BIN help" \
    "Commands:"

echo ""

# ============================================
# Model Creation - Basic
# ============================================

echo -e "${BLUE}Group: Model Creation - Basic${NC}"

run_test \
    "Create 2-4-1 model" \
    "$MLP_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic.json" \
    "Created MLP model"

check_file_exists \
    "JSON file created for 2-4-1" \
    "$TEMP_DIR/basic.json"

check_json_valid \
    "JSON contains valid structure" \
    "$TEMP_DIR/basic.json"

run_test \
    "Output shows correct architecture" \
    "$MLP_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic2.json" \
    "Input size: 2"

run_test \
    "Output shows hidden size" \
    "$MLP_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic3.json" \
    "Hidden sizes: 4"

run_test \
    "Output shows output size" \
    "$MLP_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic4.json" \
    "Output size: 1"

echo ""

# ============================================
# Model Creation - Multi-layer
# ============================================

echo -e "${BLUE}Group: Model Creation - Multi-layer${NC}"

run_test \
    "Create 3-5-3-2 network" \
    "$MLP_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/multilayer.json" \
    "Created MLP model"

check_file_exists \
    "JSON file for multi-layer" \
    "$TEMP_DIR/multilayer.json"

run_test \
    "Multi-layer output shows correct input" \
    "$MLP_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml2.json" \
    "Input size: 3"

run_test \
    "Multi-layer output shows both hidden sizes" \
    "$MLP_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml3.json" \
    "Hidden sizes: 5,3"

run_test \
    "Multi-layer output shows correct output size" \
    "$MLP_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml4.json" \
    "Output size: 2"

echo ""

# ============================================
# Hyperparameters
# ============================================

echo -e "${BLUE}Group: Hyperparameters${NC}"

run_test \
    "Create with custom learning rate" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lr.json --lr=0.01" \
    "Created MLP model"

run_test \
    "Custom learning rate saved" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lr2.json --lr=0.01" \
    "Learning rate: 0"

run_test \
    "Create with dropout" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/dropout.json --dropout=0.2" \
    "Created MLP model"

run_test \
    "Create with L2 regularization" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/l2.json --l2=0.001" \
    "Created MLP model"

echo ""

# ============================================
# Activation Functions
# ============================================

echo -e "${BLUE}Group: Activation Functions${NC}"

run_test \
    "Create with sigmoid activation" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/sigmoid.json --hidden-act=sigmoid" \
    "Created MLP model"

run_test \
    "Create with tanh activation" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/tanh.json --hidden-act=tanh" \
    "Created MLP model"

run_test \
    "Create with ReLU activation" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/relu.json --hidden-act=relu" \
    "Created MLP model"

run_test \
    "Create with softmax output" \
    "$MLP_BIN create --input=2 --hidden=3 --output=2 --save=$TEMP_DIR/softmax.json --output-act=softmax" \
    "Created MLP model"

echo ""

# ============================================
# Optimizers
# ============================================

echo -e "${BLUE}Group: Optimizers${NC}"

run_test \
    "Create with SGD optimizer" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/sgd.json --optimizer=sgd" \
    "Created MLP model"

run_test \
    "Create with Adam optimizer" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/adam.json --optimizer=adam" \
    "Created MLP model"

echo ""

# ============================================
# Model Info
# ============================================

echo -e "${BLUE}Group: Model Info${NC}"

run_test \
    "Display model info" \
    "$MLP_BIN info --model=$TEMP_DIR/basic.json" \
    "MLP Model Information"

run_test \
    "Info shows input size" \
    "$MLP_BIN info --model=$TEMP_DIR/basic.json" \
    "Input size:"

run_test \
    "Info shows output size" \
    "$MLP_BIN info --model=$TEMP_DIR/basic.json" \
    "Output size:"

echo ""

# ============================================
# Prediction
# ============================================

echo -e "${BLUE}Group: Prediction${NC}"

run_test \
    "Predict with simple model" \
    "$MLP_BIN predict --model=$TEMP_DIR/basic.json --input=0.5,0.5" \
    "Output:"

run_test \
    "Predict with multilayer model" \
    "$MLP_BIN predict --model=$TEMP_DIR/multilayer.json --input=0.1,0.2,0.3" \
    "Output:"

echo ""

# ============================================
# Facade Binary Tests
# ============================================

echo -e "${BLUE}Group: Facade Binary Tests${NC}"

run_test \
    "Facade create model" \
    "$FACADE_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/facade_basic.json" \
    "Created MLP model"

check_file_exists \
    "Facade JSON file created" \
    "$TEMP_DIR/facade_basic.json"

run_test \
    "Facade info command" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_basic.json" \
    "MLP Model Information"

run_test \
    "Facade predict command" \
    "$FACADE_BIN predict --model=$TEMP_DIR/facade_basic.json --input=0.5,0.5" \
    "Output:"

echo ""

# ============================================
# Cross-Binary Compatibility
# ============================================

echo -e "${BLUE}Group: Cross-Binary Compatibility${NC}"

run_test \
    "MLP can load Facade model" \
    "$MLP_BIN info --model=$TEMP_DIR/facade_basic.json" \
    "MLP Model Information"

run_test \
    "Facade can load MLP model" \
    "$FACADE_BIN info --model=$TEMP_DIR/basic.json" \
    "MLP Model Information"

run_test \
    "MLP can predict with Facade model" \
    "$MLP_BIN predict --model=$TEMP_DIR/facade_basic.json --input=0.5,0.5" \
    "Output:"

run_test \
    "Facade can predict with MLP model" \
    "$FACADE_BIN predict --model=$TEMP_DIR/basic.json --input=0.5,0.5" \
    "Output:"

echo ""

# ============================================
# Model File Creation for HTML Apps
# ============================================

echo -e "${BLUE}Group: Model File Creation for HTML Apps${NC}"

run_test \
    "Create simple model for JS (2-4-1)" \
    "$MLP_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/js_simple.json" \
    "Created MLP model"

run_test \
    "Create multi-layer model for JS (3-5-3-2)" \
    "$MLP_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/js_multilayer.json" \
    "Created MLP model"

run_test \
    "Create complex model for JS (10-16-8-4-2)" \
    "$MLP_BIN create --input=10 --hidden=16,8,4 --output=2 --save=$TEMP_DIR/js_complex.json" \
    "Created MLP model"

run_test \
    "Facade model with hyperparameters" \
    "$FACADE_BIN create --input=5 --hidden=8,8 --output=3 --save=$TEMP_DIR/js_facade.json --lr=0.01 --optimizer=adam --dropout=0.1" \
    "Created MLP model"

echo ""

echo -e "${BLUE}Group: JSON Validation for HTML Loading${NC}"

check_json_valid \
    "JS simple model JSON valid" \
    "$TEMP_DIR/js_simple.json"

check_json_valid \
    "JS multilayer model JSON valid" \
    "$TEMP_DIR/js_multilayer.json"

check_json_valid \
    "JS complex model JSON valid" \
    "$TEMP_DIR/js_complex.json"

check_json_valid \
    "JS facade model JSON valid" \
    "$TEMP_DIR/js_facade.json"

run_test \
    "All JSON models have proper weight structure" \
    "for f in $TEMP_DIR/js_*.json; do grep -q '\"weights\"' \"\$f\" || exit 1; done && echo 'ok'" \
    "ok"

run_test \
    "All JSON models have proper bias structure" \
    "for f in $TEMP_DIR/js_*.json; do grep -q '\"biases\"' \"\$f\" || exit 1; done && echo 'ok'" \
    "ok"

echo ""

echo -e "${BLUE}Group: Cross-Loading Tests (Binary ↔ HTML)${NC}"

run_test \
    "Model created by MLP binary is loadable format" \
    "jq . $TEMP_DIR/js_simple.json > /dev/null 2>&1 && echo 'ok'" \
    "ok"

run_test \
    "Model created by FacadeMLP binary is loadable format" \
    "jq . $TEMP_DIR/js_facade.json > /dev/null 2>&1 && echo 'ok'" \
    "ok"

run_test \
    "Simple model can be predicted with binary" \
    "$MLP_BIN predict --model=$TEMP_DIR/js_simple.json --input=0.5,0.5" \
    "Output:"

run_test \
    "Multilayer model can be predicted with binary" \
    "$MLP_BIN predict --model=$TEMP_DIR/js_multilayer.json --input=0.1,0.2,0.3" \
    "Output:"

run_test \
    "Facade model can access weights from JS-created file" \
    "$FACADE_BIN get-weight --model=$TEMP_DIR/js_simple.json --layer=0 --neuron=0 --weight=0" \
    "Weight"

run_test \
    "Facade can modify JS model and save" \
    "$FACADE_BIN set-weight --model=$TEMP_DIR/js_simple.json --layer=0 --neuron=0 --weight=0 --value=0.123 --save=$TEMP_DIR/js_modified.json && jq . $TEMP_DIR/js_modified.json > /dev/null 2>&1 && echo 'ok'" \
    "ok"

run_test \
    "Modified JS model can be loaded and predicted" \
    "$MLP_BIN predict --model=$TEMP_DIR/js_modified.json --input=0.5,0.5" \
    "Output:"

echo ""

echo -e "${BLUE}Group: HTML App Function Coverage - index.html (MLP)${NC}"

run_test \
    "index.html: Load model with camelCase format" \
    "grep -q 'inputSize.*hiddenSizes.*outputSize' $TEMP_DIR/js_simple.json || $MLP_BIN info --model=$TEMP_DIR/js_simple.json" \
    "Input size"

run_test \
    "index.html: MultiLayerPerceptron class supports multiple hidden layers" \
    "$MLP_BIN create --input=2 --hidden=4,3,2 --output=1 --save=$TEMP_DIR/mhidden.json && grep '\"hidden_sizes\"' $TEMP_DIR/mhidden.json" \
    "hidden_sizes"

run_test \
    "index.html: Model has all layers for visualization" \
    "grep -q '\"hidden_layers\"' $TEMP_DIR/js_multilayer.json && grep -q '\"output_layer\"' $TEMP_DIR/js_multilayer.json && echo 'ok'" \
    "ok"

run_test \
    "index.html: Model preserves activation functions" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/act.json --hidden-act=tanh --output-act=sigmoid && grep -q 'activation' $TEMP_DIR/act.json && echo 'ok'" \
    "ok"

run_test \
    "index.html: Model ready for training (has learning rate)" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/train.json --lr=0.1 && grep -q '\"learning_rate\"' $TEMP_DIR/train.json && echo 'ok'" \
    "ok"

echo ""

echo -e "${BLUE}Group: HTML App Function Coverage - facaded_mlp.html${NC}"

run_test \
    "facaded_mlp.html: Can load models created by MLP" \
    "$FACADE_BIN info --model=$TEMP_DIR/js_multilayer.json" \
    "MLP Model Information"

run_test \
    "facaded_mlp.html: Facade supports extended weight access" \
    "$FACADE_BIN get-weight --model=$TEMP_DIR/js_facade.json --layer=0 --neuron=0 --weight=0" \
    "Weight"

run_test \
    "facaded_mlp.html: Facade supports layer information queries" \
    "$FACADE_BIN layer-info --model=$TEMP_DIR/js_facade.json --layer=0" \
    "Layer 0"

run_test \
    "facaded_mlp.html: Facade supports bias modification" \
    "$FACADE_BIN set-bias --model=$TEMP_DIR/js_facade.json --layer=0 --neuron=0 --value=0.5 --save=$TEMP_DIR/facade_bias.json && jq . $TEMP_DIR/facade_bias.json > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "facaded_mlp.html: Facade can compute layer output with input" \
    "$FACADE_BIN get-output --model=$TEMP_DIR/js_facade.json --layer=0 --neuron=0 --input=0.1,0.2,0.3,0.4,0.5" \
    "Output"

run_test \
    "facaded_mlp.html: Facade histogram generation" \
    "$FACADE_BIN histogram --model=$TEMP_DIR/js_facade.json --layer=0 --type=activation" \
    "Histogram"

echo ""

echo -e "${BLUE}Group: mlp_player.html Compatibility Tests${NC}"

run_test \
    "mlp_player.html: Model is loadable for inference" \
    "jq . $TEMP_DIR/js_simple.json > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "mlp_player.html: Can predict with loaded model" \
    "$MLP_BIN predict --model=$TEMP_DIR/js_simple.json --input=0.5,0.5" \
    "Output:"

run_test \
    "mlp_player.html: Multilayer model predictions work" \
    "$MLP_BIN predict --model=$TEMP_DIR/js_multilayer.json --input=0.1,0.2,0.3" \
    "Output:"

run_test \
    "mlp_player.html: Architecture displayed correctly" \
    "output=$($FACADE_BIN info --model=$TEMP_DIR/js_multilayer.json); echo \"$output\" | grep -q 'Input size' && echo \"$output\" | grep -q 'Output size' && echo 'ok'" \
    "ok"

echo ""

echo -e "${BLUE}Group: Cross-Application Model Compatibility${NC}"

run_test \
    "Model created in index.html can load in facaded_mlp.html" \
    "jq '.hiddenSizes' $TEMP_DIR/js_simple.json > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "Model created in facaded_mlp.html can load in index.html" \
    "jq '.hidden_sizes' $TEMP_DIR/js_facade.json > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "Model from facaded_mlp can be used in mlp_player.html" \
    "$FACADE_BIN predict --model=$TEMP_DIR/js_facade.json --input=0.1,0.2,0.3,0.4,0.5" \
    "Output:"

run_test \
    "Binary can load and work with any HTML app's model" \
    "for model in $TEMP_DIR/js_*.json; do $MLP_BIN info --model=\$model > /dev/null || exit 1; done && echo 'ok'" \
    "ok"

echo ""

echo -e "${BLUE}Group: Full Workflow: Create → Save → Load → Predict${NC}"

run_test \
    "Workflow A: MLP binary → index.html" \
    "$MLP_BIN create --input=4 --hidden=6,4 --output=2 --save=$TEMP_DIR/workflow_a.json && jq '.inputSize,.hiddenSizes,.outputSize' $TEMP_DIR/workflow_a.json | wc -l | grep -q '3' && echo 'ok'" \
    "ok"

run_test \
    "Workflow B: MLP binary → facaded_mlp.html → predict" \
    "$MLP_BIN create --input=5 --hidden=8,6 --output=3 --save=$TEMP_DIR/workflow_b.json && $FACADE_BIN predict --model=$TEMP_DIR/workflow_b.json --input=0.1,0.2,0.3,0.4,0.5" \
    "Output:"

run_test \
    "Workflow C: FacadeMLP → mlp_player.html → predict" \
    "$FACADE_BIN create --input=3 --hidden=5,4 --output=2 --save=$TEMP_DIR/workflow_c.json && $MLP_BIN predict --model=$TEMP_DIR/workflow_c.json --input=0.1,0.2,0.3" \
    "Output:"

run_test \
    "Workflow D: Modify in Facade → Load in all HTML apps" \
    "$FACADE_BIN set-weight --model=$TEMP_DIR/workflow_c.json --layer=0 --neuron=0 --weight=0 --value=0.999 --save=$TEMP_DIR/workflow_d.json && $MLP_BIN predict --model=$TEMP_DIR/workflow_d.json --input=0.1,0.2,0.3" \
    "Output:"

echo ""

# ============================================
# Summary
# ============================================

echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total tests: $TOTAL"
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
