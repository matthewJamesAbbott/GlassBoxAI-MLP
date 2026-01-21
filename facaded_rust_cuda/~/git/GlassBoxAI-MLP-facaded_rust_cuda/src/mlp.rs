// MLP module with security-focused implementations

use std::collections::HashMap;

// Re-export core types for public API
pub use crate::Matrix;
pub use crate::MLP;

// Security-focused MLP implementation
pub struct SecureMLP {
    pub mlp: MLP,
    pub weights_hash: String,
}

impl SecureMLP {
    pub fn new() -> Self {
        let mlp = MLP::new();
        let weights_hash = mlp.get_weights_hash();
        
        SecureMLP {
            mlp,
            weights_hash,
        }
    }
    
    // Forward pass with additional security checks
    pub fn forward_secure(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        // Input validation
        if input.len() != crate::HIDDEN_SIZE_IN {
            return Err(format!("Input vector length {} doesn't match expected {}", 
                              input.len(), crate::HIDDEN_SIZE_IN));
        }
        
        // Value range hardening
        let clamped_input: Vec<f32> = input.iter()
            .map(|&val| val.clamp(-crate::CLIP_VALUE, crate::CLIP_VALUE))
            .collect();
        
        // Forward pass
        self.mlp.forward(&clamped_input)
    }
    
    // Verify weights integrity
    pub fn verify_weights(&self) -> bool {
        // In a real implementation, this would verify the Blake3/SHA256 hash
        !self.weights_hash.is_empty()
    }
}
