pub const CUDA_KERNEL_SRC: &str = r#"
extern "C" {

__device__ double d_Sigmoid(double x) {
    if (x < -500.0) return 0.0;
    else if (x > 500.0) return 1.0;
    else return 1.0 / (1.0 + exp(-x));
}

__device__ double d_DSigmoid(double x) {
    return x * (1.0 - x);
}

__device__ double d_TanhActivation(double x) {
    return tanh(x);
}

__device__ double d_DTanh(double x) {
    return 1.0 - (x * x);
}

__device__ double d_ReLU(double x) {
    return (x > 0.0) ? x : 0.0;
}

__device__ double d_DReLU(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}

__device__ double d_ApplyActivation(double x, int actType) {
    switch (actType) {
        case 0: return d_Sigmoid(x);
        case 1: return d_TanhActivation(x);
        case 2: return d_ReLU(x);
        default: return d_Sigmoid(x);
    }
}

__device__ double d_ApplyActivationDerivative(double x, int actType) {
    switch (actType) {
        case 0: return d_DSigmoid(x);
        case 1: return d_DTanh(x);
        case 2: return d_DReLU(x);
        default: return d_DSigmoid(x);
    }
}

__global__ void FeedForwardKernel(
    double* outputs, double* weights, double* biases,
    double* prevOutputs, int numNeurons, int numInputs, int prevSize, int actType
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        double sum = biases[i];
        for (int j = 0; j < prevSize; j++) {
            sum += prevOutputs[j] * weights[i * numInputs + j];
        }
        outputs[i] = d_ApplyActivation(sum, actType);
    }
}

__global__ void FeedForwardSoftmaxSumKernel(
    double* sums, double* weights, double* biases,
    double* prevOutputs, int numNeurons, int numInputs, int prevSize
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        double sum = biases[i];
        for (int j = 0; j < prevSize; j++) {
            sum += prevOutputs[j] * weights[i * numInputs + j];
        }
        sums[i] = sum;
    }
}

__global__ void SoftmaxKernel(double* sums, double* outputs, int n, double maxVal, double sumExp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double val = exp(sums[i] - maxVal) / sumExp;
        if (val < 1e-15) val = 1e-15;
        else if (val > 1.0 - 1e-15) val = 1.0 - 1e-15;
        outputs[i] = val;
    }
}

__global__ void ApplyDropoutKernel(
    double* outputs, unsigned char* dropoutMask, int numNeurons,
    double dropoutRate, double scale, unsigned long seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        if (dropoutRate <= 0.0) {
            dropoutMask[i] = 1;
            return;
        }
        unsigned long state = seed + i * 1099087573UL;
        state = state * 1103515245UL + 12345UL;
        float randVal = (float)(state % 10000) / 10000.0f;
        if (randVal > dropoutRate) {
            dropoutMask[i] = 1;
            outputs[i] = outputs[i] * scale;
        } else {
            dropoutMask[i] = 0;
            outputs[i] = 0.0;
        }
    }
}

__global__ void BackPropOutputKernel(
    double* errors, double* outputs, double* target,
    int numNeurons, int actType, int isSoftmax
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        if (isSoftmax) {
            errors[i] = target[i] - outputs[i];
        } else {
            errors[i] = d_ApplyActivationDerivative(outputs[i], actType) * (target[i] - outputs[i]);
        }
    }
}

__global__ void BackPropHiddenKernel(
    double* errors, double* outputs, unsigned char* dropoutMask,
    double* nextErrors, double* nextWeights,
    int numNeurons, int nextNumNeurons, int nextNumInputs, int actType
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        if (!dropoutMask[i]) {
            errors[i] = 0.0;
            return;
        }
        double errorSum = 0.0;
        for (int j = 0; j < nextNumNeurons; j++) {
            errorSum += nextErrors[j] * nextWeights[j * nextNumInputs + i];
        }
        errors[i] = d_ApplyActivationDerivative(outputs[i], actType) * errorSum;
    }
}

__global__ void UpdateWeightsSGDKernel(
    double* weights, double* biases, double* errors,
    double* prevOutputs, int numNeurons, int numInputs, int prevSize,
    double learningRate, double l2Lambda
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        for (int j = 0; j < prevSize; j++) {
            double gradient = errors[i] * prevOutputs[j];
            if (l2Lambda > 0.0)
                gradient = gradient - l2Lambda * weights[i * numInputs + j];
            weights[i * numInputs + j] += learningRate * gradient;
        }
        biases[i] += learningRate * errors[i];
    }
}

__global__ void UpdateWeightsAdamKernel(
    double* weights, double* biases, double* errors,
    double* prevOutputs, double* M, double* V, double* MBias, double* VBias,
    int numNeurons, int numInputs, int prevSize,
    double* params
) {
    double learningRate = params[0];
    double l2Lambda = params[1];
    double beta1 = params[2];
    double beta2 = params[3];
    int timestep = (int)params[4];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        double eps = 1e-8;
        double beta1_t = pow(beta1, (double)timestep);
        double beta2_t = pow(beta2, (double)timestep);

        for (int j = 0; j < prevSize; j++) {
            int idx = i * numInputs + j;
            double gradient = -errors[i] * prevOutputs[j];
            if (l2Lambda > 0.0)
                gradient += l2Lambda * weights[idx];

            M[idx] = beta1 * M[idx] + (1.0 - beta1) * gradient;
            V[idx] = beta2 * V[idx] + (1.0 - beta2) * gradient * gradient;

            double mHat = M[idx] / (1.0 - beta1_t);
            double vHat = V[idx] / (1.0 - beta2_t);

            weights[idx] -= learningRate * mHat / (sqrt(vHat) + eps);
        }

        double gradient = -errors[i];
        MBias[i] = beta1 * MBias[i] + (1.0 - beta1) * gradient;
        VBias[i] = beta2 * VBias[i] + (1.0 - beta2) * gradient * gradient;
        double mHat = MBias[i] / (1.0 - beta1_t);
        double vHat = VBias[i] / (1.0 - beta2_t);
        biases[i] -= learningRate * mHat / (sqrt(vHat) + eps);
    }
}

__global__ void UpdateWeightsRMSPropKernel(
    double* weights, double* biases, double* errors,
    double* prevOutputs, double* V, double* VBias,
    int numNeurons, int numInputs, int prevSize,
    double learningRate, double l2Lambda
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        double eps = 1e-8;
        double decay = 0.9;

        for (int j = 0; j < prevSize; j++) {
            int idx = i * numInputs + j;
            double gradient = -errors[i] * prevOutputs[j];
            if (l2Lambda > 0.0)
                gradient += l2Lambda * weights[idx];

            V[idx] = decay * V[idx] + (1.0 - decay) * gradient * gradient;
            weights[idx] -= learningRate * gradient / (sqrt(V[idx]) + eps);
        }

        double gradient = -errors[i];
        VBias[i] = decay * VBias[i] + (1.0 - decay) * gradient * gradient;
        biases[i] -= learningRate * gradient / (sqrt(VBias[i]) + eps);
    }
}

__global__ void BatchNormForwardKernel(
    double* outputs, double* gamma, double* beta,
    double* runningMean, double* runningVar,
    double* batchMean, double* batchVar,
    int numNeurons, int isTraining, double momentum, double epsilon
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        double mean, var;
        if (isTraining) {
            mean = batchMean[i];
            var = batchVar[i];
            runningMean[i] = (1.0 - momentum) * runningMean[i] + momentum * mean;
            runningVar[i] = (1.0 - momentum) * runningVar[i] + momentum * var;
        } else {
            mean = runningMean[i];
            var = runningVar[i];
        }
        double normalized = (outputs[i] - mean) / sqrt(var + epsilon);
        outputs[i] = gamma[i] * normalized + beta[i];
    }
}

__global__ void BatchNormBackwardKernel(
    double* errors, double* outputs, double* gamma,
    double* dGamma, double* dBeta,
    double* batchMean, double* batchVar,
    int numNeurons, double epsilon
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        double normalized = (outputs[i] - batchMean[i]) / sqrt(batchVar[i] + epsilon);
        dGamma[i] = errors[i] * normalized;
        dBeta[i] = errors[i];
        errors[i] = errors[i] * gamma[i] / sqrt(batchVar[i] + epsilon);
    }
}

__global__ void ComputeBatchStatsKernel(
    double* outputs, double* batchMean, double* batchVar,
    int numNeurons
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNeurons) {
        batchMean[i] = outputs[i];
        batchVar[i] = 0.0;
    }
}

} // extern "C"
"#;

pub const KERNEL_NAMES: &[&str] = &[
    "FeedForwardKernel",
    "FeedForwardSoftmaxSumKernel",
    "SoftmaxKernel",
    "ApplyDropoutKernel",
    "BackPropOutputKernel",
    "BackPropHiddenKernel",
    "UpdateWeightsSGDKernel",
    "UpdateWeightsAdamKernel",
    "UpdateWeightsRMSPropKernel",
    "BatchNormForwardKernel",
    "BatchNormBackwardKernel",
    "ComputeBatchStatsKernel",
];
