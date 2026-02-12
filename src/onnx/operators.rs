use crate::utils::errors::Error;

pub(crate) enum OnnxOperators {
    Gemm,
    ReLU,
    Flatten,
}

impl std::str::FromStr for OnnxOperators {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gemm" => Ok(Self::Gemm),
            "relu" => Ok(Self::ReLU),
            "flatten" => Ok(Self::Flatten),
            _ => Err(Error::UnsupportedOp { op_type: s.into() }),
        }
    }
}
