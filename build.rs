use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(&["bin/onnx.proto"], &["bin", "src"])?;
    Ok(())
}
