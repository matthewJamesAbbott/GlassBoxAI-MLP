use std::env;
use std::process;

use crate::mlp::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TCommand {
    CmdNone, CmdCreate, CmdTrain, CmdPredict, CmdInfo, CmdHelp,
    CmdGetWeight, CmdSetWeight, CmdGetBias, CmdSetBias,
    CmdGetOutput, CmdGetError, CmdLayerInfo, CmdHistogram,
    CmdGetOptimizer, CmdGetWeights, CmdGetAllOutputs,
}

fn PrintUsage() {
    println!("Facaded MLP CUDA (Rust) - Command-line Multi-Layer Perceptron (w/ Facade)");
    println!();
    println!("Commands:");
    println!("  create         Create a new MLP model");
    println!("  train          Train an existing model with data");
    println!("  predict        Make predictions with a trained model");
    println!("  info           Display model information");
    println!("  get-weight     Get a single weight value");
    println!("  set-weight     Set a single weight value");
    println!("  get-weights    Get all weights for a neuron");
    println!("  get-bias       Get bias value for a neuron");
    println!("  set-bias       Set bias value for a neuron");
    println!("  get-output     Get neuron output value");
    println!("  get-outputs    Get all outputs for all layers");
    println!("  get-error      Get neuron error value");
    println!("  layer-info     Display layer information");
    println!("  histogram      Display activation or gradient histogram");
    println!("  get-optimizer  Get optimizer state values (M, V for Adam/RMSProp)");
    println!("  help           Show this help message");
    println!();
    println!("Create Options:");
    println!("  --input=N              Input layer size (required)");
    println!("  --hidden=N,N,...       Hidden layer sizes (required)");
    println!("  --output=N             Output layer size (required)");
    println!("  --save=FILE            Save model to file (required)");
    println!("  --lr=VALUE             Learning rate (default: 0.1)");
    println!("  --optimizer=TYPE       sgd|adam|rmsprop (default: sgd)");
    println!("  --hidden-act=TYPE      sigmoid|tanh|relu|softmax (default: sigmoid)");
    println!("  --output-act=TYPE      sigmoid|tanh|relu|softmax (default: sigmoid)");
    println!("  --dropout=VALUE        Dropout rate 0-1 (default: 0)");
    println!("  --l2=VALUE             L2 regularization (default: 0)");
    println!("  --beta1=VALUE          Adam beta1 (default: 0.9)");
    println!("  --beta2=VALUE          Adam beta2 (default: 0.999)");
    println!();
    println!("Train Options:");
    println!("  --model=FILE           Model file to load (required)");
    println!("  --data=FILE            Training data CSV file (required)");
    println!("  --save=FILE            Save trained model to file (required)");
    println!("  --epochs=N             Number of training epochs (default: 100)");
    println!("  --batch=N              Batch size (default: 1)");
    println!("  --lr=VALUE             Override learning rate");
    println!("  --lr-decay             Enable learning rate decay");
    println!("  --lr-decay-rate=VALUE  LR decay rate (default: 0.95)");
    println!("  --lr-decay-epochs=N    Epochs between decay (default: 10)");
    println!("  --early-stop           Enable early stopping");
    println!("  --patience=N           Early stopping patience (default: 10)");
    println!("  --normalize            Normalize input data");
    println!("  --verbose              Show training progress");
    println!();
    println!("Predict Options:");
    println!("  --model=FILE           Model file to load (required)");
    println!("  --input=v1,v2,...      Input values (required)");
    println!();
    println!("Facade Options:");
    println!("  --layer=L              Layer index (required for facade commands)");
    println!("  --neuron=N             Neuron index (required for most facade commands)");
    println!("  --weight=W             Weight index within neuron (for weight commands)");
    println!("  --value=V              Value to set (required for set-* commands)");
    println!("  --bins=N               Number of histogram bins (default: 20)");
    println!("  --type=TYPE            Histogram type: activation or gradient (default: activation)");
    println!("  --run-input=v1,v2,...  Input values for get-output/layer-info/histogram");
    println!();
    println!("Examples:");
    println!("  facaded_mlp_cuda create --input=2 --hidden=8 --output=1 --save=xor.json");
    println!("  facaded_mlp_cuda train --model=xor.json --data=xor.csv --epochs=1000 --save=xor_trained.json");
    println!("  facaded_mlp_cuda predict --model=xor_trained.json --input=1,0");
    println!("  facaded_mlp_cuda info --model=xor_trained.json");
    println!("  facaded_mlp_cuda get-weight --model=xor.json --layer=1 --neuron=0 --weight=0");
    println!("  facaded_mlp_cuda set-weight --model=xor.json --layer=1 --neuron=0 --weight=0 --value=0.5 --save=xor_mod.json");
    println!("  facaded_mlp_cuda get-output --model=xor.json --layer=1 --run-input=1,0");
    println!("  facaded_mlp_cuda layer-info --model=xor.json --layer=1 --run-input=1,0");
    println!("  facaded_mlp_cuda histogram --model=xor.json --layer=1 --type=activation --run-input=1,0");
}

pub fn run() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        PrintUsage();
        process::exit(0);
    }

    let cmd_str = &args[1];
    let command = match cmd_str.as_str() {
        "create" => TCommand::CmdCreate,
        "train" => TCommand::CmdTrain,
        "predict" => TCommand::CmdPredict,
        "info" => TCommand::CmdInfo,
        "help" | "--help" | "-h" => TCommand::CmdHelp,
        "get-weight" => TCommand::CmdGetWeight,
        "set-weight" => TCommand::CmdSetWeight,
        "get-weights" => TCommand::CmdGetWeights,
        "get-bias" => TCommand::CmdGetBias,
        "set-bias" => TCommand::CmdSetBias,
        "get-output" => TCommand::CmdGetOutput,
        "get-outputs" => TCommand::CmdGetAllOutputs,
        "get-error" => TCommand::CmdGetError,
        "layer-info" => TCommand::CmdLayerInfo,
        "histogram" => TCommand::CmdHistogram,
        "get-optimizer" => TCommand::CmdGetOptimizer,
        _ => {
            eprintln!("Unknown command: {}", cmd_str);
            PrintUsage();
            process::exit(1);
        }
    };

    if command == TCommand::CmdHelp {
        PrintUsage();
        process::exit(0);
    }

    let mut input_size: i32 = 0;
    let mut output_size: i32 = 0;
    let mut hidden_sizes: TIntArray = Vec::new();
    let mut input_values: Darray = Vec::new();
    let mut model_file = String::new();
    let mut save_file = String::new();
    let mut data_file = String::new();
    let mut learning_rate: f64 = 0.1;
    let mut optimizer = TOptimizerType::otSGD;
    let mut hidden_act = TActivationType::atSigmoid;
    let mut output_act = TActivationType::atSigmoid;
    let mut dropout_rate: f64 = 0.0;
    let mut l2_lambda: f64 = 0.0;
    let mut beta1: f64 = 0.9;
    let mut beta2: f64 = 0.999;
    let mut epochs: i32 = 100;
    let mut batch_size: i32 = 1;
    let mut lr_decay = false;
    let mut lr_decay_rate: f64 = 0.95;
    let mut lr_decay_epochs: i32 = 10;
    let mut early_stop = false;
    let mut patience: i32 = 10;
    let mut normalize = false;
    let mut verbose = false;
    let mut lr_override = false;
    let mut layer_idx: i32 = -1;
    let mut neuron_idx: i32 = -1;
    let mut weight_idx: i32 = -1;
    let mut set_value: f64 = 0.0;
    let mut has_set_value = false;
    let mut histogram_type = String::from("activation");
    let mut histogram_bins: usize = 20;
    let mut run_input: Darray = Vec::new();

    for i in 2..args.len() {
        let arg = &args[i];
        if arg == "--lr-decay" { lr_decay = true; continue; }
        if arg == "--early-stop" { early_stop = true; continue; }
        if arg == "--normalize" { normalize = true; continue; }
        if arg == "--verbose" { verbose = true; continue; }

        if let Some(eq) = arg.find('=') {
            let key = &arg[..eq];
            let value = &arg[eq + 1..];
            match key {
                "--input" => {
                    if command == TCommand::CmdPredict { input_values = ParseDoubleArray(value); }
                    else { input_size = value.parse().unwrap_or(0); }
                }
                "--hidden" => hidden_sizes = ParseIntArray(value),
                "--output" => output_size = value.parse().unwrap_or(0),
                "--model" => model_file = value.to_string(),
                "--save" => save_file = value.to_string(),
                "--data" => data_file = value.to_string(),
                "--lr" => { learning_rate = value.parse().unwrap_or(0.1); lr_override = true; }
                "--optimizer" => optimizer = ParseOptimizer(value),
                "--hidden-act" => hidden_act = ParseActivation(value),
                "--output-act" => output_act = ParseActivation(value),
                "--dropout" => dropout_rate = value.parse().unwrap_or(0.0),
                "--l2" => l2_lambda = value.parse().unwrap_or(0.0),
                "--beta1" => beta1 = value.parse().unwrap_or(0.9),
                "--beta2" => beta2 = value.parse().unwrap_or(0.999),
                "--epochs" => epochs = value.parse().unwrap_or(100),
                "--batch" => batch_size = value.parse().unwrap_or(1),
                "--lr-decay-rate" => lr_decay_rate = value.parse().unwrap_or(0.95),
                "--lr-decay-epochs" => lr_decay_epochs = value.parse().unwrap_or(10),
                "--patience" => patience = value.parse().unwrap_or(10),
                "--layer" => layer_idx = value.parse().unwrap_or(-1),
                "--neuron" => neuron_idx = value.parse().unwrap_or(-1),
                "--weight" => weight_idx = value.parse().unwrap_or(-1),
                "--value" => { set_value = value.parse().unwrap_or(0.0); has_set_value = true; }
                "--type" => histogram_type = value.to_string(),
                "--bins" => histogram_bins = value.parse().unwrap_or(20),
                "--run-input" => run_input = ParseDoubleArray(value),
                _ => eprintln!("Unknown option: {}", key),
            }
        }
    }

    match command {
        TCommand::CmdCreate => {
            if input_size <= 0 { eprintln!("Error: --input is required"); process::exit(1); }
            if hidden_sizes.is_empty() { eprintln!("Error: --hidden is required"); process::exit(1); }
            if output_size <= 0 { eprintln!("Error: --output is required"); process::exit(1); }
            if save_file.is_empty() { eprintln!("Error: --save is required"); process::exit(1); }

            let mut mlp = match TMultiLayerPerceptronCUDA::new(input_size, &hidden_sizes, output_size, hidden_act, output_act) {
                Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); }
            };
            mlp.LearningRate = learning_rate;
            mlp.Optimizer = optimizer;
            mlp.DropoutRate = dropout_rate;
            mlp.L2Lambda = l2_lambda;
            mlp.Beta1 = beta1;
            mlp.Beta2 = beta2;
            if let Err(e) = mlp.Save(&save_file) { eprintln!("Error: {}", e); process::exit(1); }

            println!("Created MLP model (CUDA/Rust):");
            println!("  Input size: {}", input_size);
            print!("  Hidden sizes: "); for (i, &s) in hidden_sizes.iter().enumerate() { print!("{}{}", if i > 0 { "," } else { "" }, s); } println!();
            println!("  Output size: {}", output_size);
            println!("  Hidden activation: {}", ActivationToStr(hidden_act));
            println!("  Output activation: {}", ActivationToStr(output_act));
            println!("  Optimizer: {}", OptimizerToStr(optimizer));
            println!("  Learning rate: {:.4}", learning_rate);
            println!("  Saved to: {}", save_file);
        }
        TCommand::CmdTrain => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if data_file.is_empty() { eprintln!("Error: --data is required"); process::exit(1); }
            if save_file.is_empty() { eprintln!("Error: --save is required"); process::exit(1); }

            let mut mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) {
                Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); }
            };
            if lr_override { mlp.LearningRate = learning_rate; }
            mlp.EnableLRDecay = lr_decay;
            mlp.LRDecayRate = lr_decay_rate;
            mlp.LRDecayEpochs = lr_decay_epochs;
            mlp.EnableEarlyStopping = early_stop;
            mlp.EarlyStoppingPatience = patience;

            let mut data = LoadDataCSV(&data_file, mlp.GetInputSize(), mlp.GetOutputSize());
            if data.is_empty() { eprintln!("Error: No valid data loaded"); process::exit(1); }
            println!("Loaded {} training samples", data.len());
            if normalize { NormalizeData(&mut data); println!("Data normalized"); }

            for epoch in 1..=epochs {
                ShuffleData(&mut data);
                for dp in &data { mlp.Train(&dp.Input, &dp.Target).unwrap(); }
                if verbose && (epoch % 10 == 0 || epoch == 1) {
                    let mut total_loss = 0.0;
                    for dp in &data {
                        let output = mlp.Predict(&dp.Input).unwrap();
                        total_loss += mlp.ComputeLoss(&output, &dp.Target);
                    }
                    println!("Epoch {}/{} - Loss: {:.6}", epoch, epochs, total_loss / data.len() as f64);
                }
            }

            let mut total_loss = 0.0;
            for dp in &data {
                let output = mlp.Predict(&dp.Input).unwrap();
                total_loss += mlp.ComputeLoss(&output, &dp.Target);
            }
            println!("Final loss: {:.6}", total_loss / data.len() as f64);
            if let Err(e) = mlp.Save(&save_file) { eprintln!("Error: {}", e); process::exit(1); }
            println!("Model saved to: {}", save_file);
        }
        TCommand::CmdPredict => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if input_values.is_empty() { eprintln!("Error: --input is required"); process::exit(1); }

            let mut mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) {
                Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); }
            };
            if input_values.len() as i32 != mlp.GetInputSize() {
                eprintln!("Error: Expected {} input values, got {}", mlp.GetInputSize(), input_values.len());
                process::exit(1);
            }
            let output = mlp.Predict(&input_values).unwrap();
            print!("Input: "); for (i, &v) in input_values.iter().enumerate() { print!("{}{:.4}", if i > 0 { ", " } else { "" }, v); } println!();
            print!("Output: "); for (i, &v) in output.iter().enumerate() { print!("{}{:.6}", if i > 0 { ", " } else { "" }, v); } println!();
            if output.len() > 1 { println!("Max index: {}", MaxIndex(&output)); }
        }
        TCommand::CmdInfo => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            let mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) {
                Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); }
            };
            println!("MLP Model Information (CUDA/Rust)");
            println!("=================================");
            println!("Input size: {}", mlp.GetInputSize());
            println!("Output size: {}", mlp.GetOutputSize());
            println!("Hidden layers: {}", mlp.GetHiddenLayerCount());
            print!("Hidden sizes: "); for (i, &s) in mlp.GetHiddenSizes().iter().enumerate() { print!("{}{}", if i > 0 { "," } else { "" }, s); } println!();
            print!("Layer sizes: {}", mlp.GetInputSize()); for &s in mlp.GetHiddenSizes() { print!(" -> {}", s); } println!(" -> {}", mlp.GetOutputSize());
            println!();
            println!("Hyperparameters:");
            println!("  Learning rate: {:.6}", mlp.LearningRate);
            println!("  Optimizer: {}", OptimizerToStr(mlp.Optimizer));
            println!("  Hidden activation: {}", ActivationToStr(mlp.HiddenActivation));
            println!("  Output activation: {}", ActivationToStr(mlp.OutputActivation));
            println!("  Dropout rate: {:.4}", mlp.DropoutRate);
            println!("  L2 lambda: {:.6}", mlp.L2Lambda);
            println!("  Beta1: {:.4}", mlp.Beta1);
            println!("  Beta2: {:.4}", mlp.Beta2);
            println!("  Timestep: {}", mlp.Timestep);
            println!();
            println!("Total layers: {}", mlp.GetNumLayers());
            for i in 0..mlp.GetNumLayers() as usize { println!("  Layer {}: {} neurons", i, mlp.GetLayerSize(i)); }
        }
        TCommand::CmdGetWeight => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if layer_idx < 0 { eprintln!("Error: --layer is required"); process::exit(1); }
            if neuron_idx < 0 { eprintln!("Error: --neuron is required"); process::exit(1); }
            if weight_idx < 0 { eprintln!("Error: --weight is required"); process::exit(1); }
            let mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            let w = mlp.GetNeuronWeight(layer_idx, neuron_idx, weight_idx);
            println!("Weight[layer={}, neuron={}, weight={}] = {:.10}", layer_idx, neuron_idx, weight_idx, w);
        }
        TCommand::CmdSetWeight => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if layer_idx < 0 { eprintln!("Error: --layer is required"); process::exit(1); }
            if neuron_idx < 0 { eprintln!("Error: --neuron is required"); process::exit(1); }
            if weight_idx < 0 { eprintln!("Error: --weight is required"); process::exit(1); }
            if !has_set_value { eprintln!("Error: --value is required"); process::exit(1); }
            if save_file.is_empty() { eprintln!("Error: --save is required"); process::exit(1); }
            let mut mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            let old_val = mlp.GetNeuronWeight(layer_idx, neuron_idx, weight_idx);
            mlp.SetNeuronWeight(layer_idx, neuron_idx, weight_idx, set_value);
            mlp.Save(&save_file).unwrap();
            println!("Weight[layer={}, neuron={}, weight={}]: {:.10} -> {:.10}", layer_idx, neuron_idx, weight_idx, old_val, set_value);
            println!("Saved to: {}", save_file);
        }
        TCommand::CmdGetWeights => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if layer_idx < 0 { eprintln!("Error: --layer is required"); process::exit(1); }
            if neuron_idx < 0 { eprintln!("Error: --neuron is required"); process::exit(1); }
            let mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            let weights = mlp.GetNeuronWeights(layer_idx, neuron_idx);
            println!("Weights[layer={}, neuron={}] ({} weights):", layer_idx, neuron_idx, weights.len());
            for (i, &w) in weights.iter().enumerate() { println!("  [{}] = {:.10}", i, w); }
        }
        TCommand::CmdGetBias => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if layer_idx < 0 { eprintln!("Error: --layer is required"); process::exit(1); }
            if neuron_idx < 0 { eprintln!("Error: --neuron is required"); process::exit(1); }
            let mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            let b = mlp.GetNeuronBias(layer_idx, neuron_idx);
            println!("Bias[layer={}, neuron={}] = {:.10}", layer_idx, neuron_idx, b);
        }
        TCommand::CmdSetBias => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if layer_idx < 0 { eprintln!("Error: --layer is required"); process::exit(1); }
            if neuron_idx < 0 { eprintln!("Error: --neuron is required"); process::exit(1); }
            if !has_set_value { eprintln!("Error: --value is required"); process::exit(1); }
            if save_file.is_empty() { eprintln!("Error: --save is required"); process::exit(1); }
            let mut mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            let old_val = mlp.GetNeuronBias(layer_idx, neuron_idx);
            mlp.SetNeuronBias(layer_idx, neuron_idx, set_value);
            mlp.Save(&save_file).unwrap();
            println!("Bias[layer={}, neuron={}]: {:.10} -> {:.10}", layer_idx, neuron_idx, old_val, set_value);
            println!("Saved to: {}", save_file);
        }
        TCommand::CmdGetOutput => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if layer_idx < 0 { eprintln!("Error: --layer is required"); process::exit(1); }
            let mut mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            if !run_input.is_empty() {
                if run_input.len() as i32 != mlp.GetInputSize() { eprintln!("Error: --run-input needs {} values", mlp.GetInputSize()); process::exit(1); }
                let _ = mlp.Predict(&run_input);
            }
            let outputs = mlp.GetLayerOutputs(layer_idx);
            println!("Outputs[layer={}] ({} neurons):", layer_idx, outputs.len());
            for (i, &v) in outputs.iter().enumerate() { println!("  [{}] = {:.10}", i, v); }
        }
        TCommand::CmdGetAllOutputs => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            let mut mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            if !run_input.is_empty() {
                if run_input.len() as i32 != mlp.GetInputSize() { eprintln!("Error: --run-input needs {} values", mlp.GetInputSize()); process::exit(1); }
                let _ = mlp.Predict(&run_input);
            }
            for l in 0..mlp.GetNumLayers() {
                let outputs = mlp.GetLayerOutputs(l);
                println!("Layer {} ({} neurons):", l, outputs.len());
                for (i, &v) in outputs.iter().enumerate() { println!("  [{}] = {:.6}", i, v); }
            }
        }
        TCommand::CmdGetError => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if layer_idx < 0 { eprintln!("Error: --layer is required"); process::exit(1); }
            let mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            let errors = mlp.GetLayerErrors(layer_idx);
            println!("Errors[layer={}] ({} neurons):", layer_idx, errors.len());
            for (i, &v) in errors.iter().enumerate() { println!("  [{}] = {:.10}", i, v); }
        }
        TCommand::CmdLayerInfo => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if layer_idx < 0 { eprintln!("Error: --layer is required"); process::exit(1); }
            let mut mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            let num_neurons = mlp.GetLayerSize(layer_idx as usize);
            let num_weights = if layer_idx > 0 && num_neurons > 0 { mlp.GetWeightsPerNeuron(layer_idx, 0) } else { 0 };
            println!("Layer {} Information:", layer_idx);
            println!("  Neurons: {}", num_neurons);
            println!("  Weights per neuron: {}", num_weights);
            println!("  Activation: {}", ActivationToStr(mlp.GetLayerActivation(layer_idx)));
            println!();
            if !run_input.is_empty() {
                if run_input.len() as i32 != mlp.GetInputSize() {
                    println!("Warning: --run-input needs {} values, skipping outputs", mlp.GetInputSize());
                } else {
                    let _ = mlp.Predict(&run_input);
                    let outputs = mlp.GetLayerOutputs(layer_idx);
                    println!("  Neuron outputs (after prediction):");
                    for (i, &v) in outputs.iter().enumerate() { println!("    [{}] = {:.6}", i, v); }
                }
            }
            println!();
            println!("  Neuron details:");
            for n in 0..num_neurons.min(10) {
                let bias = mlp.GetNeuronBias(layer_idx, n);
                println!("    Neuron {}: bias={:.6}", n, bias);
            }
            if num_neurons > 10 { println!("    ... ({} more neurons)", num_neurons - 10); }
        }
        TCommand::CmdHistogram => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if layer_idx < 0 { eprintln!("Error: --layer is required"); process::exit(1); }
            let mut mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            if !run_input.is_empty() {
                if run_input.len() as i32 != mlp.GetInputSize() { eprintln!("Error: --run-input needs {} values", mlp.GetInputSize()); process::exit(1); }
                let _ = mlp.Predict(&run_input);
            }
            let histogram = if histogram_type == "gradient" || histogram_type == "error" {
                println!("Gradient Histogram [layer={}] ({} bins):", layer_idx, histogram_bins);
                mlp.GetGradientHistogram(layer_idx, histogram_bins)
            } else {
                println!("Activation Histogram [layer={}] ({} bins):", layer_idx, histogram_bins);
                mlp.GetActivationHistogram(layer_idx, histogram_bins)
            };
            let max_count = *histogram.iter().max().unwrap_or(&1);
            for (i, &c) in histogram.iter().enumerate() {
                let bar_len = if max_count > 0 { (c * 40 / max_count) as usize } else { 0 };
                println!("  [{:2}] {:4} |{}", i, c, "#".repeat(bar_len));
            }
        }
        TCommand::CmdGetOptimizer => {
            if model_file.is_empty() { eprintln!("Error: --model is required"); process::exit(1); }
            if layer_idx < 0 { eprintln!("Error: --layer is required"); process::exit(1); }
            if neuron_idx < 0 { eprintln!("Error: --neuron is required"); process::exit(1); }
            let mlp = match TMultiLayerPerceptronCUDA::Load(&model_file) { Ok(m) => m, Err(e) => { eprintln!("Error: {}", e); process::exit(1); } };
            println!("Optimizer state [layer={}, neuron={}]:", layer_idx, neuron_idx);
            println!("  Optimizer: {}", OptimizerToStr(mlp.Optimizer));
            println!("  Timestep: {}", mlp.Timestep);
            println!("  Bias M: {:.10}", mlp.GetBiasM(layer_idx, neuron_idx));
            println!("  Bias V: {:.10}", mlp.GetBiasV(layer_idx, neuron_idx));
            if weight_idx >= 0 {
                println!();
                println!("  Weight[{}]:", weight_idx);
                println!("    M: {:.10}", mlp.GetWeightM(layer_idx, neuron_idx, weight_idx));
                println!("    V: {:.10}", mlp.GetWeightV(layer_idx, neuron_idx, weight_idx));
            } else {
                let num_weights = mlp.GetWeightsPerNeuron(layer_idx, neuron_idx);
                println!();
                println!("  All weights ({}):", num_weights);
                for w in 0..num_weights.min(10) {
                    println!("    [{}] M={:.6} V={:.6}", w, mlp.GetWeightM(layer_idx, neuron_idx, w), mlp.GetWeightV(layer_idx, neuron_idx, w));
                }
                if num_weights > 10 { println!("    ... ({} more)", num_weights - 10); }
            }
        }
        _ => {}
    }
}
