//! # Example of how to use a PyTorch exported script module in Rust
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Nov-15
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** None known at this time.
//! - **Todo:** Further development tasks to be determined.


use tch::{jit, Tensor, Device, Kind};
use std::error::Error;
use std::fs::File;
use std::io::{Write, BufWriter};

/// Run the PyTorch model and save the output to a CSV file.
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: example-app <path-to-exported-script-module>");
        std::process::exit(1);
    }

    let module_path = &args[1];
    let module = jit::CModule::load(module_path)?;

    let num_steps = 40000;
    let mut data = Vec::with_capacity(num_steps);
    for i in 0..num_steps {
        data.push(i as f32 * 0.001);
    }

    // Create an input tensor.
    let input_tensor = Tensor::from_slice(&data).view([num_steps as i64, 1]);
    // input_tensor.print(); // print the tensor
    println!("Input: {:?}", &input_tensor);

    let input_ivalue = tch::IValue::Tensor(input_tensor);
    let output_ivalue = module.forward_is(&[input_ivalue])?;
    let output = if let tch::IValue::Tensor(output) = output_ivalue {
        output
    } else {
        return Err("Expected Tensor".into());
    };

    let output = output.to_device(Device::Cpu).to_kind(Kind::Float);
    println!("Output: {:?}", &output);

    let mut file = BufWriter::new(File::create("output.csv")?);

    let sizes = output.size();
    let num_rows = sizes[0] as usize;
    let num_cols = sizes[1] as usize;

    for i in 0..num_rows {
        for j in 0..num_cols {
            write!(file, "{:.6}", output.double_value(&[i as i64, j as i64]))?;
            if j != num_cols - 1 {
                write!(file, ",")?;
            }
        }
        writeln!(file)?;
    }

    println!("ok");
    Ok(())
}
